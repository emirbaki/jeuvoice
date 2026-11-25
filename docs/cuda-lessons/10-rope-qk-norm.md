# Ders 10: RoPE ve kafa başına RMSNorm (Qwen3)

OmniVoice'ın dil modeli Qwen3'tür. Whisper'dan iki farkı bu dersin konusu: her dikkat kafası, sorgu ve anahtar vektörünü konuma göre **döndürür** (RoPE), ve her kafanın kendi **RMSNorm**'u vardır (q/k norm). İkisi de 128 değerlik satırlar üzerinde çalışır ve her katmanda iki kez koşar.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`rope_half_f16`, `rope_half_f16_shared`, `rope_half_f16_rows`, `rope_f16`, `rmsnorm_rows_f16`, `rmsnorm_rows_f16_small`, `block_sum_4`), `crates/jeuvoice-gpu/src/ops/rope.rs`.

Ölçümler: RTX 3050 Ti Laptop, bellek sınırı 192 GB/s. `target\release\gpu-bench.exe rope_f16`, `... rmsnorm_rows`. Boyutlar bilerek büyük: 8 MB girer, 8 MB çıkar, yani 4 MB'lık L2 tensörü tutamaz.

## 1. Ne yapıyorlar?

**RoPE** (rotary position embedding) bir çifti döndürür. Qwen3'te çift **yarıya bölünmüştür**: `(i, i + d/2)`, yani `i` ile `i + 64`. Hugging Face bunu `rotate_half` diye yazar. Açı konumdan gelir:

```
angle = pos · theta^(-2i/d),   theta = 1e6, d = 128
q'[i]      = q[i]·cos(angle) - q[i + 64]·sin(angle)
q'[i + 64] = q[i]·sin(angle) + q[i + 64]·cos(angle)
```

Döndürme uzunluğu korur, yani bir çiftin normu değişmez. İki vektörün iç çarpımı yalnızca **aradaki konum farkına** bağlı olur. Model "kim nerede" bilgisini böyle taşır.

**Çift düzeni görünmez bir ayrıntıdır ama modeli bozar.** GPT-J'in kullandığı düzen bitişiktir: çift `(2i, 2i+1)`. İlk sürüm bu düzeni yazdı, çünkü şekil ve hız aynı görünüyordu. Ölçüm konuştu: yanlış düzenle ilk geçişin logitleri f32 PyTorch referansıyla 1712 konumun yalnızca **111'inde** aynı argmax'ı verdi (%6,5). Doğru düzenle **1692** konumda (%98,8). Kalan 20 fark, referansın kendi fp16 yuvarlamasıyla aynı mertebede. Yani hata "biraz sayısal gürültü" değil, modelin tamamen başka bir çıktı üretmesiydi.

**q/k RMSNorm** her kafayı ayrı normalleştirir (Whisper'da LayerNorm tüm satırı normalleştiriyordu):

```
x'[c] = x[c] / sqrt(mean(x²) + eps) · w[c],   eps = 1e-6, d = 128
```

`w` (q_norm, k_norm) modelde f32 durur; 28 katmanda 56 tane vardır.

## 2. Neden gerekli, ne sıklıkla çalışır?

Her katmanda sıra şu: q/k/v projeksiyonu → **q_norm, k_norm** → **RoPE** → attention. Yani 28 katman için 56 RMSNorm ve 56 RoPE çağrısı bir geçişte. Maske yinelemeli çözmede 32 geçiş var: **1792 kez**.

Boyut: bir geçişte dizi ~486 konum (8,4 s ses). Sorgu `[486, 16, 128]`, anahtar `[486, 8, 128]` → RoPE için 486·16·128 = 1M değer, RMSNorm için 1,5M değer.

RoPE neden gerekli: onsuz model sözcük sırasını bilemez, "kedi fareyi kovaladı" ile "fare kediyi kovaladı" aynı olurdu. q/k norm neden gerekli: dikkat skorları büyümesin diye; Qwen3 bunu kafa düzeyinde yapar.

## 3. Naive hâller ve ölçümleri

**RoPE, ilk sürüm:** her thread bir (satır, kafa, çift) alır ve açıyı kendisi hesaplar:

```rust
let angle = pos * inv_freq[pair];
let (s, c) = (sinf(angle), cosf(angle));
```

Ölçüm (`[2000, 16, 128]`): **117,9 µs, 138,9 GB/s, sınırın %72'si.**

**RMSNorm, ilk sürüm:** satır başına 256 thread (d = 128 olduğu hâlde), ve yazma geçişinde `x` **ikinci kez** okunur:

```rust
sum += v * v;                       // 1. okuma
let scale = 1.0 / (block_sum(sum) / d as f32 + eps).sqrt();
let v = *x.add(base + c) as f32;    // 2. okuma
*x.add(base + c) = (v * scale * w[c]) as f16;
```

Ölçüm (`[32000, 128]`): **372,6 µs, 44 GB/s, sınırın %23'ü.** İki kayıp var: thread'lerin yarısı boş (128 değerlik satırda 256 thread) ve yarısı ikinci okumayla geçiyor.

## 4. Hızlandırma adımları

### 4.1. RoPE: açıyı satır başına bir kez (117,9 → 98,9 µs)

Bir satırın 16 kafası **aynı açıları** kullanır. İlk sürüm aynı `sinf/cosf` çiftini 16 kez hesaplıyor. Yeni sürümde blok bir satırdır, 64 thread (d/2) açıyı bir kez hesaplar, `sin` ve `cos`'u shared memory'ye yazar, sonra her thread 16 kafayı dolaşır:

```rust
ANG[2*pair] = sinf(angle); ANG[2*pair+1] = cosf(angle);
thread::sync_threads();
let (s, c) = (ANG[2*pair], ANG[2*pair+1]);
while h < heads { /* çifti döndür */ h += 1; }
```

Ölçüm: **98,9 µs, 165,6 GB/s, %86.** Kazanç 1,19 kat.

Bu ölçüm **token-major** girdi içindir: satır bir konumdur ve o konumun 16 kafası satırın içindedir, yani 16 kafa aynı `sin/cos` çiftini paylaşır. Girdi kafa-major ise (kernela `heads = 1` görünür) paylaşım olmaz ve kazanç kaybolur.

Neden bu kadar az? Çünkü kernel **bellek sınırlı**. 2M çiftde tetik hesabı 32k'ya indi, ama süre yalnızca %16 düştü: `sinf`/`cosf` PTX'te birkaç komut (`sin.approx.f32` benzeri), ve bellek gecikmesi zaten baskın. Bu, "hesabı azaltmak her zaman hız demek değildir" kuralının bir örneği daha. Yine de bu değişiklik kalıyor: küçük tensörlerde (L2'de kalan) fark büyür.

Bitişik düzenin paylaşımlı sürümü aynı kodla 109,0 µs (%78) ölçüldü: düzen yalnızca indekstir, hızı değiştirmez. Fark **doğrulukta**.

### 4.2. RMSNorm: 128 thread ve register (372,6 → 154,2 µs)

İki değişiklik birlikte:

1. **Blok 128 thread** (`block_sum_4`, 4 warp). 128 değerlik satırda artık boş thread yok.
2. **Değer register'da:** thread kendi değerini bir kez okur, yazarken onu kullanır. `x` bir kez okunur, bir kez yazılır.

```rust
let v = *x.add(base + c) as f32;                  // tek okuma
let scale = 1.0 / (block_sum_4(v * v) / d as f32 + eps).sqrt();
*x.add(base + c) = (v * scale * *w.add(c)) as f16; // register'dan
```

Ölçüm: **154,2 µs, 106 GB/s, %55.** Kazanç **2,4 kat**.

Neden %100 değil? Trafik 16 MB, 154 µs'de 106 GB/s. Kalan kayıp işlem boyutunda: her blok yalnızca 128 f16 = 256 bayt okuyor. Bir warp 32 thread ile 2 bayt okur: 64 bayt, yani bir 32 baytlık sektörün sadece bir parçası. Blok başına iş küçük olduğu için bellek denetleyicisi tam doldurulamıyor ve 32 000 blok sıraya giriyor. Çözüm alıştırma 1'de: blok başına 4 satır.

### 4.3. Warp başına satır, blok başına dört satır (154,2 → 103,2 µs)

Alıştırma 1'in çözümü: bir blok 128 thread ile tek satır yerine dört warp ile dört satır işler. Her warp bir satırın 128 değerini alır; thread başına dört değer register'da kalır. Toplam `block_sum_4` shared dizisi ve iki `bar.sync` kalkar, yerini beş `shfl.sync.bfly` alır.

`rmsnorm_rows_f16_warp` aynı 8 MB tensörde **103,2 µs, 158,8 GB/s, %83** ölçüldü. Blok başına okuma dört katına çıkınca bellek denetleyicisi daha iyi doluyor. PTX'te `bar.sync` ve `ld.shared` yok, beş warp shuffle var.

Paket 1 `sqrt.rn` + `rcp.rn` çiftini tek `rsqrt.approx.f32` ile değiştirdi. Aynı kernel **94,5 µs, 173,4 GB/s, %90** ölçüldü. Yaklaşık hata f16 çıktıda görünmüyor ve tap testleri geçiyor. Kalan %10, 128 baytlık satırın sektör veriminden ve okuma/yazma karışımından geliyor.

Üretim yolu q/k norm çağrılarını bu kernel'e geçirdi. İzole kernel kazancı açık olsa da uçtan uca decode süresi ölçüm gürültüsü içinde kaldı: gerçek geçiş 87,7 ms'den 86,5 ms'ye, s1 product 2,13 s'den 2,16 s'ye (sıcaklık 66,8 C'den 68,8 C'ye çıktı) değişti. Tap hataları eşik içinde kaldı.

## 4b. Üretimdeki sürüm: konum tablosu yok

Üretimde konum her zaman `0..t-1`, yani satırın kendi sırası. Tabloyu her geçişte hazırlayıp yüklemek iki kopya ve iki senkronizasyon demekti. `rope_half_f16_rows` konumu `row % t` ile kendisi hesaplar (`row` = düz kafa satırı, `t` = dizideki konum sayısı): tablo ne yüklenir ne okunur.

Ölçüm (aynı 8 MB tensör, 2000 konum): tablolu `rope_half_f16` 117,6 µs, tablosuz `rope_half_f16_rows` **111,2 µs**. Fark hem tablo okumasının yokluğu hem de tamsayı modülosunun bellek gölgesinde kalması. Geçiş başına kazanç küçük, asıl kazanç kopyaların ve senkronizasyonların kalkması: paketlenmiş 972 satırlık geçiş 105,5 ms'den 100,0 ms'ye indi ve arena 16,8 MB küçüldü (ölü `kx`/`vx` türevleri de gitti).

Tuzak: satır sayısı `t`'nin tam katı olmalı, yoksa konum dizinin ortasında sıfıra döner. `rope_rows` bunu panikle reddeder.

## 5. Kartın sınırı ve ne kadar yaklaştık

| Kernel | Önce | Sonra | Sınırın yüzdesi |
|---|---|---|---|
| `rope_f16` | 117,9 µs (naive) | **98,9 µs** | %72 → **%86** |
| `rmsnorm_rows_f16` | 372,6 µs (naive) | **154,2 µs** | %23 → **%55** |
| `rmsnorm_rows_f16_warp` | 154,2 µs (blok/satır) | **94,5 µs** (rsqrt) | %55 → **%90** |

RoPE için sınır bellek: 1M f16 oku + yaz = 4 MB... hayır, 2000 satır × 16 kafa × 128 × 2 bayt = 8 MB, oku + yaz = **16,4 MB**. 98,9 µs'de 165,6 GB/s, 192 GB/s'nin %86'sı. Bu, kartta bir bellek kernel'i için iyi bir sonuç.

RMSNorm'un %55'i, blok başına işin küçüklüğünden. Geçiş başına maliyet: RoPE 56 × ~0,05 ms ≈ 2,8 ms değil, ölçek küçük olduğu için gerçek toplam geçiş başına ~5 ms; 32 geçişte ~160 ms. Bu yüzden alıştırma 1 önemli: %55'i %80'e çıkarmak geçiş başına ~2 ms kazandırır.

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `rope_half_f16_shared`: `sin.approx.f32` / `cos.approx.f32` (ya da libm'in polinomu), `st.shared.f32` × 2, `bar.sync`, sonra kafa döngüsünde `ld.global.b16` × 2 + `fma.rn.f32` × 4 + `st.global.b16` × 2.
- `rmsnorm_rows_f16_small`: **tek** `ld.global.b16` (ikinci okuma yok), `shfl.sync.bfly` × 5, `bar.sync` × 2, `ld.shared.f32` × 4, `st.global.b16` × 1. Naive sürümde iki `ld.global.b16` görürsün: dersin ölçtüğü fark tam olarak bu.
- `block_sum_4`: 4 slot (256 thread'lik `block_sum` 8 slot kullanır). Daha az shared trafik.

Nsight Compute:

```powershell
ncu --set full -k regex:rmsnorm_rows -o rms target\release\gpu-bench.exe rmsnorm_rows
```

- **Memory Throughput**: fast sürümde %55, naive'de %23.
- **Memory Workload Analysis → Sectors/Request**: 2 baytlık okumada sektör başına kullanılan bayt düşük çıkar. Blok başına satır sayısını artırınca bu düzelir.
- **Scheduler → Waves Per SM**: 32 000 blok küçük dalgalar hâlinde gelir; dalga başına doluluk düşer.
- `rope` için **Compute (SM) Throughput**'a bak: shared sürümde trig'i 16'da 1'e indirince SM yüzdesi düşer, bellek yüzdesi yükselir.

## 7. Alıştırmalar

1. **Blok başına 4 satır.** RMSNorm'da bir blok 128 thread ile 4 satır işlesin (her satır için ayrı reduksiyon). %55 nereye çıkıyor? Neden 100'e ulaşmıyor?
2. **Çift okuma.** RMSNorm'u `[f16; 2]` ile yaz (thread başına iki değer, d = 128 → 64 thread). Tek okuma korunuyor mu, süre değişiyor mu?
3. **RoPE'u q/k projeksiyonuna kat.** Projeksiyon GEMM'inin çıktısını yazarken döndür (epilogue). Hangi tensör hiç yazılmaz, kaç µs kazanılır?
4. **Tablo.** Bir geçişte tüm konumların `sin/cos` tablosunu bir kez hesapla (486 × 64 × 2 f32 = 249 KB), RoPE onu okusun. Trig tamamen kalkınca süre ne oluyor?
5. **`sinf` yerine polinom.** Kendi sinüs polinomunu yaz (5. derece yeterli) ve `libm::sinf` ile karşılaştır: hız ve hata. Attention skorları ne kadar kayıyor?
6. **Çift düzeni.** Aynı girdiyle `rope_half_f16` ve `rope_f16` kernel'lerini koştur. Farkın hangi indekslerde olduğunu yaz. `rotate_half` ile yarıya bölünmüş düzenin neden bitişik düzenden farklı olduğunu göster. Aynı şeyi bir test iddiasına çevir: yanlış düzen kaç logit argmax'ını bozar?
