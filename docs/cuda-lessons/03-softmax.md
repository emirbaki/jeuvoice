# Ders 3: Softmax (ve "online softmax" neden burada kaybetti)

Softmax, attention'ın kalbi. Bu derste üç kernel yazıyoruz. İlki doğru ve hızlı, ikincisi kitapta "daha iyi" denen yöntem ama burada **daha yavaş** çıkıyor, üçüncüsü register numarasıyla kartın bellek sınırının %94'üne ulaşıyor. Dersin asıl konusu şu: bir algoritma kâğıt üzerinde daha az bellek okusa bile, fazladan bir `exp` ve seri bir bağımlılık zinciri o kazancı yiyebilir.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`block_max`, `softmax_3pass_f16`, `softmax_online_f16`, `softmax_regs_f16`) ve `crates/jeuvoice-gpu/src/ops/softmax.rs`.

Ölçümler: RTX 3050 Ti Laptop, bellek sınırı 192 GB/s, `[30000, 1500]` f16. `target\release\gpu-bench.exe softmax`.

## 1. Ne yapıyor?

Bir satır `x[0..n]` için:

```
m   = max_i (s · x[i])
y[i] = exp(s · x[i] - m) / Σ_j exp(s · x[j] - m)
```

`s` ölçek faktörü: attention'da `s = 1/√64 = 0.125`. Maksimumu çıkarmak isteğe bağlı değildir. `x[i] = 100` olsa `exp(100) ≈ 2.7e43` olur ve f32'nin sınırını (3.4e38) aşar. `m`'yi çıkarınca üst `exp(0) = 1` olur, alt sıfıra yaklaşır. Sonuç hiç taşmaz.

Decoder'da bir de **nedensel maske** var: `j > row + offset` olan sütunlar `-inf` sayılır ve olasılıkları 0 olur. Kernel bunu `causal >= 0` parametresiyle yapar; encoder `-1` (maske yok) geçer.

## 2. Neden gerekli, ne sıklıkla çalışır?

Attention'ın üç adımı: `scores = Q·Kᵀ`, `probs = softmax(scores · s)`, `out = probs·V`. Softmax ortadaki adımdır. Onsuz attention doğrusal bir işlem olurdu ve model "hangi token'a ne kadar bakacağını" öğrenemezdi.

Encoder'da her katmanın 20 kafası vardır ve her kafa 1500 × 1500 skor üretir. Yani bir katmanda softmax `20 · 1500 = 30000` satır işler. 32 katman için **960 000 satır** bir 30 saniyelik pencerede.

**Maliyet:** Bu dersin ölçüm boyutu tam olarak bir katmanın skorlarıdır: `[30000, 1500]`, 45 milyon eleman, 90 MB. Ürün kernel'i bunu **993 µs**'de bitiriyor. 32 katman için yaklaşık **32 ms**. Karşılaştır: bir GEMM `1500×1280×1280` 364 µs. Yani softmax, bir katmanın GEMM'lerinden daha pahalı. Flash attention'ın (ders 5) var olma sebebi tam olarak budur.

## 3. Naive hâl: üç geçiş, satırı üç kez oku

```rust
// Geçiş 1: maksimum
while c < n { m = m.max(*x.add(base + c) as f32 * scale); c += step; }
let m = block_max(m);          // blok içi maksimum
// Geçiş 2: üslerin toplamı
while c < n { s += expf(*x.add(base + c) as f32 * scale - m); c += step; }
let inv = 1.0 / block_sum(s);
// Geçiş 3: yaz (exp'i yeniden hesaplar)
while c < n { *x.add(base + c) = (expf(*x.add(base + c) as f32 * scale - m) * inv) as f16; c += step; }
```

Ölçüm: **1,08 ms, 167 GB/s, sınırın %87'si.**

İlk bakışta şaşırtıcı: satırı **üç** kez okuyor ama yine de %87. Sebep L1 cache. Bir satır 1500 × 2 = 3000 bayt. Bir SM'de aynı anda duran blokların satırları L1'e (128 KB) rahat sığıyor. Yani ikinci ve üçüncü okuma DRAM'e inmiyor. DRAM trafiği zaten ideal: 90 MB oku + 90 MB yaz.

Ama bedava da değil: eleman başına **iki `exp`** (geçiş 2 ve geçiş 3) ve iki toplama turu (4 bariyer).

## 4. Hızlandırma adımları (ve bir yanlış tahmin)

### 4.1. Online softmax: beklenen kazanç, ölçülen kayıp

Online softmax tek geçişte hem maksimumu hem toplamı tutar. Yeni bir maksimum geldiğinde eski toplam yeniden ölçeklenir:

```rust
let m_new = m.max(v);
d = d * expf(m - m_new) + expf(v - m_new);
m = m_new;
```

Sonra blok maksimumu alınır, her thread'in toplamı o maksimuma taşınır: `s = block_sum(d * expf(m - m_block))`. Yazma geçişi ikinci okumadır. Kitap "iki okuma, üç değil, daha hızlı" der.

Ölçüm: **1,32 ms, 137 GB/s, %71.** Üç geçişli sürümden **daha yavaş**.

Neden?

1. **Eleman başına üç `exp`.** Online sürümde her eleman için yeniden ölçekleme `exp`'i ve yeni elemanın `exp`'i hesaplanır; yazma geçişinde bir tane daha. Üç geçişlide iki tane.
2. **Seri bağımlılık.** `d` bir sonraki elemanın `d`'sine bağlıdır: `d = d·exp(...) + exp(...)`. Thread'in 6 elemanı artık paralel değil, sırayla işlenir. GPU gecikmeyi ancak bağımsız işlerle saklar; burada bağımsız iş kalmıyor.
3. **Kazanç zaten yoktu.** İkinci okuma L1'den geliyordu. Yani "bir okuma tasarrufu" kâğıt üzerinde kaldı.

**Ders:** Daha az bellek trafiği her zaman daha hızlı değildir. Ölç, sonra karar ver.

### 4.2. Satırı register'da tut: 1,32 → 0,99 ms

Asıl kazanç, okuma sayısını azaltmak değil, **`exp` sayısını** azaltmak. Satırı ders 2'deki gibi register'da tutarsak hem ikinci okuma hem ikinci `exp` kalkar:

```rust
// 1. Satırı bir kez oku (maskeli slotlar -inf kalır)
while j < MAX { if in_range(j) { v[j] = *x.add(base + c) as f32 * scale; } j += 1; }
// 2. Thread içi maksimum, sonra blok maksimumu
let m = block_max(local_max);
// 3. Register'da üs al VE toplama ekle (tek exp!)
while j < MAX { if in_range(j) { v[j] = expf(v[j] - m); s += v[j]; } j += 1; }
let inv = 1.0 / block_sum(s);
// 4. Register'dan yaz
while j < MAX { if c < n { *x.add(base + c) = (v[j] * inv) as f16; } j += 1; }
```

Ölçüm: **993 µs, 181 GB/s, sınırın %94'ü.**

- Okuma 1 kez (DRAM), yazma 1 kez. Eleman başına **1 `exp`**.
- `-inf` hilesi maske için bedava: `max(-inf, v) = v` ve `exp(-inf - m) = 0`. Maske kontrolü sadece yükleme ve yazma döngülerinde kalır.
- 8 slotluk dizi (`[f32; 8]`) register'da kalır. PTX'te `ld.local` yok. 1500 sütun / 256 thread = 5,86, yani 6 slot doluyor, 2 slot boş gidiyor. Bu, `n ≤ 2048` sınırını doğurur; daha geniş satırda üç geçişli kernel'e düşer (`ops::softmax_rows_with` bunu seçer).

## 5. Kartın sınırı ve ne kadar yaklaştık

`[30000, 1500]` f16: 90 MB oku + 90 MB yaz = 180 MB. 993 µs'de 181 GB/s = 192 GB/s'nin **%94'ü**.

| Kernel | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|
| `softmax_3pass_f16` | 1,08 ms | 167 GB/s | %87 |
| `softmax_online_f16` | 1,32 ms | 137 GB/s | %71 |
| **`softmax_regs_f16`** | **993 µs** | 181 GB/s | **%94** |

%94, bu kartta bir bellek kernel'i için pratik tavandır (ders 1'deki `bias_gelu` %90, ders 2'deki LayerNorm %84). Kalan %6: iki blok toplamı (4 bariyer), slot maskeleri ve `exp`'in kendisi.

**Ölçüm notu:** GPU boşken ilk koşu düşük saat hızında başlar ve %73 gibi okunur. Tablodaki değerler ikinci ve üçüncü koşudan. Bir kernel'i ölçerken tabloyu iki kez koş.

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `softmax_regs_f16`: 8 × `ld.global.b16` (offsetler +0, +512, ...), 8 × `mul.f32`, 8 × `max.f32`, `shfl.sync.bfly` × 5 (warp maksimumu), `bar.sync`, sonra 8 × `ex2.approx.ftz.f32`.
- **`exp` donanımda!** `libm::expf` PTX'te üç komuta iniyor: iki `fma.rn.f32` (x'i `log2(e)` ile çarp, iki parçalı) ve bir `ex2.approx.ftz.f32`. Yani "pahalı polinom" korkusu yersiz. Yine de eleman başına bir tane var ve 45 milyon elemanda bu ölçülebilir.
- `selp.b16` komutları maske için: `j > row + causal` ise 0 yaz.
- `ld.local` **olmamalı**. Varsa `[f32; 8]` dizisi local memory'ye taşmıştır.

Nsight Compute:

```powershell
ncu --set full -k regex:softmax -o sm target\release\gpu-bench.exe softmax
```

- **Memory Throughput**: `regs` için ~%94, `online` için ~%71.
- **Warp State Statistics → Stall Long Scoreboard / Stall Barrier**: `3pass` ve `online` sürümlerinde yüksek. `regs` sürümünde bariyerler kalır ama bellek beklemesi azalır.
- **Instruction Statistics → Executed Instructions**: `online` sürümünde `3pass`'ten fazla komut var. Fazladan `exp` ve yeniden ölçekleme burada görünür.
- **Speed Of Light → Compute (SM) Throughput**: `online` sürümünde bellek %71 iken SM %60'ları görür. İki kısıt birbirine yaklaşınca kernel yavaşlar.

## 7. Alıştırmalar

1. **Online sürümü düzelt.** Yeniden ölçeklemeyi sadece gerçekten yeni bir maksimum geldiğinde yap: `if v > m { d = d*expf(m - v) + 1.0; m = v; } else { d += expf(v - m); }`. Rastgele veride yeni maksimum seyrek gelir. Şimdi `3pass`'i geçiyor mu? `regs`'i geçebilir mi?
2. **Warp başına satır.** Bir blok yerine bir warp bir satırı işlesin: bariyer yok, sadece `shfl.sync`. `[30000, 1500]` üzerinde ölç. Hangi satır uzunluğunda daha iyi?
3. **Maske maliyeti.** `causal = None` ve `causal = Some(0)` ölçümlerini karşılaştır. Maske kontrolü ne kadar sürüyor?
4. **f16 toplam.** Toplamı f32 yerine f16'da yap. `softmax_large_values` testi neden patlar? Hangi satırda?
5. **Çift okuma.** `softmax_regs_f16`'yı `[f16; 2]` ile yaz (ders 2'deki gibi). Kaç slot kalıyor, hız değişiyor mu?
6. **Fusion.** Softmax'ı `Q·Kᵀ` GEMM'inin içine katarsan hangi tensör hiç yazılmaz? 90 MB yazma ve 90 MB okuma ortadan kalkar. Flash attention'ın kazancını bu sayılarla hesapla (ders 5).
