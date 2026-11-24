# Ders 2: LayerNorm

LayerNorm, RMSNorm'un (ders 0) büyük kardeşidir. Aynı satırı normalleştirir ama iki fark var: satırın ortalamasını da çıkarır, ve sonucu `gamma` ile çarpıp `beta` ile kaydırır. Bu iki fark kernel'de iki zorluk getirir: **ikinci bir toplam** ve **sayısal duyarlılık**. Bu dersin sonunda kernel 63 µs'den 47 µs'ye iniyor ve kartın bellek sınırının %84'üne ulaşıyor.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`layernorm_f16`, `layernorm_regs_f16`, `layernorm_regs_f16x2`) ve `crates/jeuvoice-gpu/src/ops/layernorm.rs`.

Ölçümler: RTX 3050 Ti Laptop, bellek sınırı 192 GB/s, `[1500, 1280]` f16. `target\release\gpu-bench.exe layernorm`.

## 1. Ne yapıyor?

Bir satır `x[0..d]` için:

```
mean = (1/d) · Σ x[i]
var  = (1/d) · Σ (x[i] - mean)²
y[i] = (x[i] - mean) / √(var + eps) · gamma[i] + beta[i]
```

Whisper'da `d = 1280`, `eps = 1e-5`'tir. `gamma` ve `beta` modelden gelen öğrenilmiş vektörlerdir (`encoder.blocks.N.attn_ln.weight` gibi).

## 2. Neden gerekli, ne sıklıkla çalışır?

LayerNorm her bloğun içinde **iki** kez çalışır:

| Yer | Girdi |
|---|---|
| `ln1` | attention'dan önce |
| `ln2` | MLP'den önce |

Bu, 32 katman için 64 çağrı eder. Encoder'ın sonunda bir de `ln_post` var: toplam **65 LayerNorm** bir 30 saniyelik pencerede. Her biri `[1500, 1280]` f16 tensörü işler.

Neden gerekli? Her bloğun çıkışı bir öncekinin üstüne ekleniyor (residual). Toplamlar büyüdükçe aktivasyonların ölçeği katman katman büyür. LayerNorm her bloğun girdisini aynı ölçeğe geri getirir. Bu olmasa derin ağ eğitilemezdi, ve fp16 aralığı (65504) 32 katman sonra taşardı.

**Maliyet:** 65 × 47,4 µs ≈ **3,1 ms** bir pencere için. Encoder'ın toplam süresi birkaç ms mertebesinde olduğu için bu, toplam sürenin önemli bir parçasıdır.

## 3. Naive hâl: satırı üç kez oku

İlk kernel üç geçiş yapar. Her geçişte satırı baştan okur:

```rust
// Geçiş 1: ortalama
let mut s = 0.0;
while c < d { s += *x.add(base + c); c += step; }
let mean = block_sum(s) / d as f32;

// Geçiş 2: ortalamadan sapmanın karesi
while c < d { s += (*x.add(base + c) - mean)²; c += step; }
let inv = 1.0 / (block_sum(s) / d as f32 + eps).sqrt();

// Geçiş 3: yaz
while c < d { *y.add(base + c) = (*x.add(base + c) - mean) * inv * gamma[c] + beta[c]; c += step; }
```

Ölçüm: **63,7 µs, 120 GB/s, sınırın %63'ü.**

İki gözlem:

1. Kernel `x`'i üç kez okuyor: 3 × 3,84 MB = 11,5 MB okuma + 3,84 MB yazma = **15,4 MB**. Gerekli olan sadece 7,68 MB. Ama ölçülen hız 120 GB/s; 15,4 MB'yi 63,7 µs'de taşımak 242 GB/s eder. Bu, kartın 192 GB/s bellek sınırının **üstünde**. Demek ki ikinci ve üçüncü okumalar DRAM'den değil, **L1 cache'ten** geliyor. Bir satır 1280 × 2 = 2560 bayt, ve bir SM'de aynı anda duran blokların satırları L1'e (128 KB) rahatça sığıyor.
2. O hâlde kayıp bellekten değil, **komuttan ve gecikmeden** geliyor. Fazladan iki okuma turu, fazladan iki `block_sum` (her biri 2 bariyer), ve her turda döngü kontrolü.

## 4. Hızlandırma adımları

### 4.1. Satırı register'da tut (63,7 → 51,1 µs)

Satırı bir kez okuyup her thread'in payını **register'larda** saklarsak ikinci ve üçüncü geçiş hiç bellek okumaz:

```rust
let mut v = [0.0f32; MAX_COLS_PER_THREAD];   // 8 slot, 2048 sütuna kadar
while j < MAX_COLS_PER_THREAD {
    let c = tid + j * step;
    if c < d { v[j] = *x.add(base + c) as f32; }
    j += 1;
}
// ortalama v'den, varyans yine (v - mean)²'den, yazma v'den
```

Ölçüm: **51,1 µs, 150 GB/s, %78.**

- İki geçiş (mean, sonra varyans) **duruyor**. Sadece `x`'in nerede durduğu değişti. Bu önemli, çünkü tek geçişli `E[x²] - E[x]²` formu sayısal olarak bozuktur (bölüm 6).
- `[f32; 8]` dizisi **register'da** kalıyor. PTX'te tek bir `ld.local` yok, sadece 8 `ld.global.b16`. LLVM döngüyü tamamen açtı ve indeksler sabit oldu. Eğer indeks çalışma anında hesaplansaydı dizi **local memory**'ye (yani DRAM'e) düşerdi ve kazanç kaybolurdu.
- Sınır: `d > 2048` olan satırlar eski üç geçişli kernel'e düşer (`ops::layernorm_with` bunu kendisi seçer). Test `layernorm_kinds_agree` 3000 sütunlu satırla bu yolu da denetler.

### 4.2. Çift okuma (51,1 → 47,4 µs)

Ders 1'deki `[f16; 2]` fikri burada da işe yarıyor. Her thread iki komşu sütunu tek `[f16; 2]` olarak alır:

```rust
let p = *x.add(base + c);       // iki f16
v[2*j]     = p[0] as f32;
v[2*j + 1] = p[1] as f32;
```

Ölçüm: **47,4 µs, 162 GB/s, %84.**

- Thread başına slot sayısı yarıya iniyor (1280 sütun → 640 çift → 3 slot). Döngü kontrolü ve maskeleme işi de yarıya iniyor.
- PTX'te yine iki ayrı `ld.global.b16` var (`[%rd6]` ve `[%rd6+2]`), tek bir `b32` yok. Ders 1'de olduğu gibi L1 bunları birleştiriyor. Kazanç geniş okumadan değil, **daha az komuttan** geliyor.
- Depolama da iki `st.global.b16`. `ptxas` bunları SASS'ta tek `STG.E` komutuna birleştirebilir; bunu görmek için `cuobjdump -sass` kullan (alıştırma 1).

### 4.3. Bu boyutta daha fazlası yok: sabit maliyet

Kernel'i 4 kat büyütünce (6000 satır) tabloya bak:

| Kernel | Boyut | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|---|
| `layernorm_3pass_f16` | 1500×1280 | 63,7 µs | 120 GB/s | %63 |
| `layernorm_f16_regs` | 1500×1280 | 51,1 µs | 150 GB/s | %78 |
| **`layernorm_f16` (çift)** | 1500×1280 | **47,4 µs** | 162 GB/s | **%84** |
| `layernorm_f32` (çift) | 1500×1280 | 89,6 µs | 171 GB/s | %89 |
| `layernorm_f16_x4` | 6000×1280 | 175,5 µs | 175 GB/s | %91 |

1500 ve 6000 satır arasındaki farktan satır başına maliyet çıkar: (175,5 − 47,4) / 4500 = **28,5 ns/satır**. Yani 1500 satırın gerçek işi 42,7 µs. Geri kalan **4,7 µs sabit maliyettir** (kernel başlatma ve son dalganın boşalması). 47,4 yerine 42,7 µs olsaydı %93 olurdu.

Bu yüzden `[1500, 1280]` boyutunda %84 pratikte tavandır. Fazlası kernel'i hızlandırarak değil, **kernel sayısını azaltarak** gelir: LayerNorm'u takip eden GEMM'in içine katmak (kernel fusion) 4,7 µs'lik başlatmayı ve ara tensörün bir tur okunup yazılmasını siler. Ders 4'te GEMM'e gelince buna dönüyoruz.

Not: `layernorm_f32` aynı eleman sayısında %89'a çıkıyor, çünkü iki kat bayt taşıyor ve sabit maliyet toplam sürenin daha küçük bir parçası oluyor. Komut sayısı aynı, süre sadece bellekten geliyor. f16 sürümü komut sınırına yaklaşmış durumda.

## 5. Kartın sınırı ve ne kadar yaklaştık

Gerekli trafik: `x` okuma 3,84 MB + `y` yazma 3,84 MB = **7,68 MB**. (`gamma` ve `beta` 2 × 2560 bayt; tüm bloklar aynı değeri okuduğu için L2'de kalır.)

47,4 µs'de 162 GB/s = 192 GB/s'nin **%84'ü**. Eleman başına 2 bayt okuyup 2 bayt yazıyoruz ve araya iki blok toplamı ile iki bariyer giriyor. Bu iş yükü için %84 iyi bir sonuç. `layernorm_f16_x4` satırı %91'i görüyor, yani kernel'in kendisi sınırın %91'ine kadar çıkabiliyor.

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `layernorm_regs_f16x2`: 8 × `ld.global.b16` (4 çift), `cvt.f32.f16` × 8, bir sıra `sub.f32`/`mul.f32`/`fma.rn.f32`, yazmada 8 × `st.global.b16`.
- **`ld.local` ve `st.local` araması yap.** Hiçbiri olmalı. Varsa dizi register'dan local memory'ye taşmıştır ve kernel yavaşlar.
- `block_sum` içinde: `shfl.sync.down.b32` × 5 (warp içi toplam), bir `bar.sync`, sonra 8 `ld.shared` ve ikinci `bar.sync`. Bir satır için bu blok **4 kez** çalışır (2 toplam × 2 bariyer).

Sayısal duyarlılık, bu dersin gizli konusu. Tek geçişli form şudur:

```
var = E[x²] - E[x]²
```

`layernorm_large_offset` testi ortalaması 1000, yayılımı 1 olan satırlar kullanır. `E[x²] ≈ 1 000 001`, `E[x]² ≈ 1 000 000`. f32 yaklaşık 7 anlamlı basamak tutar, yani 1 000 001'in son basamağı zaten gürültüdür. Aradaki gerçek varyans 0,33; çıkarma sonrası elimizde anlamlı basamak kalmaz ve sonuç saçmalar. **Bu yüzden iki geçişli formu koruyoruz**: önce ortalama, sonra `(x - mean)²`. Register sürümü bunu bedava yapar, çünkü `x` zaten elimizde.

Nsight Compute:

```powershell
ncu --set full -k regex:layernorm -o ln target\release\gpu-bench.exe layernorm
```

- **Memory Throughput**: çift sürümde ~%84, üç geçişlide ~%63.
- **Warp State Statistics → Stall Barrier**: üç geçişli sürümde yüksek çıkar. Fazladan iki `block_sum` bunun sebebidir.
- **Instruction Statistics**: üç geçişli sürümde yürütülen komut sayısı belirgin fazladır. Kayıp bellek değil, komut ve bariyer gecikmesidir.
- **Launch Statistics → Registers Per Thread**: çift sürümde `v` dizisi tek başına 8 slot × 2 = 16 f32 register tutar. Toplam register sayısını buradan oku ve ders 0'daki `rmsnorm_f16` ile karşılaştır.

## 7. Alıştırmalar

1. **SASS'a bak.** `cuobjdump -sass target\release\gpu-bench.exe` çalıştır ve `layernorm_regs_f16x2` içindeki `STG` komutlarını bul. İki `st.global.b16` tek `STG.E` oldu mu? Olduysa PTX'te görünen "iki ayrı yazma" gerçek donanımda tek işlemdir.
2. **Tek geçiş, Welford.** Welford algoritması ortalama ve varyansı tek turda, sayısal olarak kararlı biçimde hesaplar. Register sürümüne ekle. `layernorm_large_offset` testini geçiyor mu? Kaç komut tasarruf ettin, süre değişti mi?
3. **Blok boyu.** 128 ve 512 thread'li sürümleri dene. `block_sum` 8 warp için sabit; 512 thread için 16 warp'lık bir `SharedArray` gerekir. Hangisi daha hızlı? Neden?
4. **Dörtlü okuma.** `[f16; 4]` (8 bayt) ile thread başına 4 sütun. `d` 4'ün katı olmalı. Ders 1'deki gibi ölç, tahmin etme.
5. **LayerNorm ile RMSNorm'u aynı boyutta karşılaştır.** `ops::rmsnorm` ile `ops::layernorm`'u `[1500, 1280]` üzerinde ölç. LayerNorm'un ek maliyeti ne kadar? (İkinci toplam ve `beta`.) `gamma`'yı 1, `beta`'yı 0 yapınca süre değişiyor mu?
6. **Fusion.** LayerNorm çıktısını yazmak yerine doğrudan bir sonraki GEMM'in girişine besleyen bir kernel düşün. Neden bu, kazançların en büyüğüdür? Hangi tensör okunup yazılmaktan kurtulur?
