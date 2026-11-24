# Ders 1: Eleman eleman kernel'ler (bias + GELU, residual toplama)

Bu derste en basit kernel türünü yazıyoruz: her çıktı elemanı sadece kendi girdisine bakar. Basit görünür ama burada GPU'da hızın neye bağlı olduğunu çok net görüyoruz. Ve bir sürpriz var: "daha geniş okuma her zaman daha hızlıdır" fikri bu kernel'de doğru çıkmadı.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`bias_act_f16`, `bias_act_f16x2`, `bias_act_f16x2_b32`, `add_f16`, `add_f16x2`) ve `crates/jeuvoice-gpu/src/ops/elementwise.rs`.

Ölçümler: RTX 3050 Ti Laptop, bellek sınırı 192 GB/s. `target\release\gpu-bench.exe bias_gelu`.

## 1. Ne yapıyorlar?

Whisper encoder'ında üç eleman eleman işlem var:

| İşlem | Formül | Nerede |
|---|---|---|
| bias + GELU | `x = gelu(x + b)` | conv1, conv2 ve her bloğun MLP'sinde ilk katmandan sonra |
| bias ekleme | `x = x + b` | attention çıkışlarından sonra |
| residual toplama | `acc = acc + x` | her bloğun iki "artık bağlantısı" ve positional embedding |

GELU, Whisper'da tam (exact) formdadır:

```
gelu(v) = 0.5 · v · (1 + erf(v / √2))
```

`erf` (hata fonksiyonu) GPU'da bir polinom yaklaşımıyla hesaplanır. Biz `libm::erff` çağırıyoruz ve cuda-oxide bunu NVIDIA'nın libdevice kütüphanesine bağlar. PTX'te bu, bir dizi `fma.rn.f32` (çarp-topla) komutudur.

## 2. Neden gerekli, ne sıklıkla çalışır?

- **Bias**: her doğrusal katmanın (`y = x·Wᵀ + b`) ikinci yarısı. cuBLAS sadece `x·Wᵀ` kısmını yapar. Bias'ı ayrı bir kernel ekler.
- **GELU**: doğrusal olmayan aktivasyon. Bu olmasaydı 32 katman tek bir büyük matris çarpımına eşit olurdu ve model hiçbir şey öğrenemezdi.
- **Residual**: her bloğun girdisi çıktısına eklenir. Bu, derin ağlarda sinyalin ve gradyanın kaybolmamasını sağlar.

Bir encoder geçişinde 32 bloğun her birinde 1 bias+GELU (`[1500, 5120]`), 4 bias ekleme ve 2 residual toplama çalışır. Bir 30 saniyelik pencere için yaklaşık 200 kernel çağrısı eder.

## 3. Temel fikir: eleman eleman kernel bellek sınırlıdır

Her eleman için bellekten 2 bayt okuyup 2 bayt yazıyoruz. Arada birkaç düzine aritmetik işlem var. Kartımız saniyede trilyonlarca işlem yapabilir ama sadece 192 GB veri taşıyabilir. Yani sınır yine bellektir (ders 0'daki RMSNorm gibi). Hedef: taşınan baytı en az tutmak ve bellek yolunu dolu tutmak.

## 4. Üç sürüm ve ölçümleri

`[1500, 5120]` f16 tensör üzerinde, yerinde (in place) `x = gelu(x + b)`:

| Kernel | Thread başına | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|---|
| `bias_act_f16` | 1 eleman | 285.3 µs | 107.7 GB/s | %56 |
| `bias_act_f16x2` | 2 eleman (`[f16; 2]`) | **178.1 µs** | 172.5 GB/s | **%90** |
| `bias_act_f16x2_b32` | 2 eleman (`u32` okuma) | 190.8 µs | 161.0 GB/s | %84 |

### 4.1. Bir eleman, bir thread

```rust
let i = thread::index_1d().get();
if i >= n { return; }
let v = *x.add(i) as f32 + *bias.add(i % d) as f32;
*x.add(i) = gelu(v) as f16;
```

Doğru ve okunaklı. Ama sınırın sadece %56'sı. Neden?

- Bir warp'ın 32 thread'i aynı anda 32 × 2 bayt = **64 bayt** okur. Bellek denetleyicisi veriyi 32 baytlık parçalarla (sector) taşır ve L1 128 baytlık satırlarla çalışır. 64 baytlık bir okuma satırın sadece yarısıdır.
- Her eleman için thread başına sabit bir iş var: indeksi hesapla, `i % d` hesapla (tam sayı bölme GPU'da yavaştır), sınırı kontrol et. Bu iş, işin yararlı kısmına göre büyüktür.

### 4.2. İki eleman, bir thread (`[f16; 2]`)

```rust
let p = *x.add(i);            // x: *mut [f16; 2]
let b = *bias.add(i % d2);
let v0 = gelu(p[0] as f32 + b[0] as f32);
let v1 = gelu(p[1] as f32 + b[1] as f32);
*x.add(i) = [v0 as f16, v1 as f16];
```

Şimdi bir warp 128 bayt okur: tam bir L1 satırı. Thread sayısı yarıya iner ve indeks, `%` ve sınır kontrolü iki eleman arasında paylaşılır. Sonuç: %90.

**Şart:** bir satırın uzunluğu `d` çift olmalıdır. Yoksa bir çift iki satıra taşar ve bias'ın yanlış elemanını alır. `d` tekse kod otomatik olarak tek elemanlı kernel'e döner (test: `3x5` şekli).

### 4.3. Sürpriz: tek 4 baytlık okuma daha yavaş

PTX'e bakınca `[f16; 2]` okumasının **iki** ayrı 2 baytlık okuma olduğunu gördük:

```
ld.global.b16 	%rs1, [%rd4+2];
ld.global.b16 	%rs2, [%rd4];
```

"Tek bir 4 baytlık okuma daha hızlı olmalı" diye düşündük ve `bias_act_f16x2_b32`'yi yazdık. Bu sürüm çifti bir `u32` olarak okur ve bitlerini ayırır:

```rust
let (p0, p1) = unpack_f16x2(*x.add(i));   // ld.global.b32
// ...
*x.add(i) = pack_f16x2(v0, v1);            // st.global.b32
```

PTX'te gerçekten `ld.global.b32` var. Ama ölçüm **daha yavaş**: 190.8 µs, %84. Neden?

1. **Bellekten taşınan bayt aynı.** Komşu thread'lerin 2 baytlık okumalarını L1 cache zaten birleştiriyor. DRAM tarafında iki sürüm aynı 128 baytlık satırları okur.
2. **Ek komutlar.** Paketi açmak ve kapatmak için `cvt.u32.u16`, `or.b32`, kaydırma ve `mov.b32 {tmp, ...}` gibi komutlar gerekir. Bu kernel'de her eleman zaten `erf` için yaklaşık 10 `fma` çalıştırıyor. Kernel bellek sınırına çok yakın ama komut yükü de önemli. Fazladan komutlar zamanı uzatır.

**Ders:** Bir optimizasyonu PTX'e bakarak değil, **ölçerek** doğrula. "Daha geniş okuma" genelde iyidir. Ama burada cache işi zaten yapıyordu, ve bizim değişikliğimiz sadece komut ekledi. Ürün `[f16; 2]` sürümünü kullanıyor. `u32` sürümü ders için duruyor.

### 4.4. Residual toplama

`add_f16x2` hesap yapmaz, sadece okur ve yazar (iki okuma, bir yazma). Burada `u32` paketleme kullanıyoruz. Bu kernel'de `erf` yok, bu yüzden paket açma komutları önemli değil. Alıştırma 2'de iki biçimi karşılaştır.

## 5. Kartın sınırı ve ne kadar yaklaştık

`[1500, 5120]` f16: okuma 15,36 MB + yazma 15,36 MB = 30,7 MB. 178,1 µs'de 172,5 GB/s, sınırın %90'ı. Ders 0'daki RMSNorm ile aynı seviye. Eleman eleman bir kernel için bu, pratikte üst sınırdır.

**Daha da hızlısı mümkün mü?** Evet, ama başka bir yoldan: kernel sayısını azaltarak. bias + GELU'yu ayrı bir kernel yerine GEMM'in içinde (cuBLASLt "epilogue" ya da kendi GEMM'imizde) yapsaydık, `[1500, 5120]` tensörü bellekten bir kez daha okuyup yazmazdık. Buna **kernel fusion** (birleştirme) denir. Ders 4'te GEMM'i işlerken buna döneceğiz.

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `bias_act_f16`: `ld.global.b16` × 2 (x ve bias), bir sıra `fma.rn.f32` (erf polinomu), `cvt.rn.f16.f32` (f32'den f16'ya), `st.global.b16`.
- `bias_act_f16x2`: `ld.global.b16` × 4, iki erf polinomu, `st.global.b16` × 2.
- `bias_act_f16x2_b32`: `ld.global.b32` × 2, `mov.b32 {tmp, ...}` ve `cvt` ile paket açma, `or.b32` ile paketleme, `st.global.b32`.

Nsight Compute:

```powershell
ncu --set full -k regex:bias_act -o gelu target\release\gpu-bench.exe bias_gelu
```

- **Memory Throughput**: `x2` için ~%90 olmalı.
- **Compute (SM) Throughput** ve **Issue Slots Busy**: `x2_b32` sürümünde komut yükünün artışını burada görürsün.
- **Memory Workload Analysis → Sectors/Request**: 2 baytlık ve 4 baytlık okumalarda DRAM'e giden sektör sayısının aynı olduğunu görürsün.

## 7. Alıştırmalar

1. **Dört eleman.** `[f16; 4]` (8 bayt) ile thread başına dört eleman yap. `d` 4'ün katı olmalı. Daha hızlı mı? Neden ya da neden değil?
2. **Residual için iki biçim.** `add_f16x2`'yi bir de `[f16; 2]` ile yaz ve ikisini `[1500, 1280]` üzerinde ölç. `erf` olmayınca paket açmanın maliyeti hâlâ görünür mü?
3. **Yaklaşık GELU.** whisper.cpp GELU'yu `tanh` yaklaşımıyla hesaplar: `0.5·v·(1 + tanh(√(2/π)·(v + 0.044715·v³)))`. Bunu yaz, hızını ölç ve testteki tam GELU'ya göre hatasını bul. Hangi değerlerde hata en büyük?
4. **`%` olmadan.** `i % d` yerine 2 boyutlu bir grid kullan: `blockIdx.y` satır, `blockIdx.x × blockDim.x + threadIdx.x` sütun. Bölme kalkınca süre değişiyor mu?
