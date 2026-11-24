# Ders 0: GPU temelleri ve araç zinciri (RMSNorm)

Bu derste ilk gerçek kernel'imizi yazıyoruz: **RMSNorm**. Aynı işi üç kernel ile yapıyoruz ve her birini ölçüyoruz. Yol boyunca GPU'nun temel kavramlarını görüyoruz: thread, warp, block, grid, bellek türleri ve PTX.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (kernel'ler) ve `crates/jeuvoice-gpu/src/ops/rmsnorm.rs` (kernel'i başlatan host kodu).

Ölçümler: RTX 3050 Ti Laptop, 20 SM, bellek sınırı 192 GB/s. Tablo `target\release\gpu-bench.exe rmsnorm` komutundan gelir.

## 1. RMSNorm ne yapar?

Bir dil modelinde her token bir vektördür. Qwen3-0.6B'de (OmniVoice'un omurgası) bu vektörün boyu `d = 1024`'tür. RMSNorm her vektörü ölçekler:

```
rms   = sqrt( (x[0]² + x[1]² + ... + x[d-1]²) / d  +  eps )
y[i]  = x[i] / rms * w[i]
```

- `eps` (1e-6) çok küçük bir sayıdır. Vektör tamamen sıfır olursa sıfıra bölmeyi engeller. Test `rmsnorm_zero_row` bunu kontrol eder.
- `w` öğrenilmiş bir ağırlık vektörüdür (boyu `d`).

Girdi bir matristir: `[rows, d]`. Her satır bir token'dır ve her satır diğerlerinden bağımsızdır. Bu bağımsızlık önemlidir: satırları paralel işleyebiliriz.

**Whisper neden LayerNorm kullanır?** LayerNorm önce ortalamayı çıkarır, sonra standart sapmaya böler, sonra bir bias ekler. RMSNorm ortalamayı çıkarmaz ve bias eklemez. Daha az iş yapar ve dil modellerinde aynı kalitede çalışır. Whisper eski bir mimaridir ve LayerNorm kullanır. Ders 2'de LayerNorm'u bu dersin üstüne kuracağız.

## 2. Neden gerekli ve ne sıklıkla çalışır?

Derin bir ağda her katmandan sonra değerlerin büyüklüğü değişir. Normalizasyon olmadan değerler katmanlar boyunca büyür ya da küçülür ve eğitim bozulur. RMSNorm her katmanın girdisini aynı ölçeğe getirir.

Qwen3'te her katmanda RMSNorm 2 kez çalışır (attention'dan önce ve MLP'den önce). Ayrıca Q ve K için bir RMSNorm daha vardır. 28 katmanla bir ileri geçişte yaklaşık 60 RMSNorm çalışır. OmniVoice bir cümle için bu ileri geçişi `num_step` kez (örneğin 32 kez) yapar. Yani tek bir kısa kernel binlerce kez çalışır. Yavaş bir RMSNorm bütün modeli yavaşlatır.

## 3. GPU temelleri, bu kernel üzerinden

### Thread, warp, block, grid

- **Thread:** En küçük iş birimi. Her thread aynı kernel kodunu çalıştırır ama kendi numarasıyla. Numarayı `thread::threadIdx_x()` (block içindeki sıra) ve `thread::blockIdx_x()` (block'un sırası) verir.
- **Warp:** 32 thread'lik grup. Donanım bir warp'ın 32 thread'ini **aynı anda, aynı komutla** çalıştırır. GPU'nun hızının kaynağı budur. Bir warp içindeki thread'ler farklı yollara saparsa (`if` dalları), donanım iki yolu sırayla çalıştırır ve hız düşer.
- **Block:** Birkaç warp'tan oluşan grup (bizde 256 thread = 8 warp). Bir block'un bütün thread'leri aynı SM'de (streaming multiprocessor) çalışır. Bu yüzden ortak bir hızlı belleği (shared memory) paylaşabilir ve birbirini bekleyebilirler (barrier).
- **Grid:** Bir kernel çağrısındaki bütün block'lar. Bizim kartta 20 SM var. Grid'in block'ları bu 20 SM'e dağıtılır.

Kernel'i başlatırken grid ve block boyunu biz seçeriz (`LaunchConfig`):

```rust
let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (ROW_BLOCK, 1, 1), shared_mem_bytes: 0 };
```

Burada her satıra bir block veriyoruz: 4096 satır için 4096 block, her biri 256 thread.

### Bellek türleri

| Bellek | Nerede | Boyut (bu kart) | Hız | Kim görür |
|---|---|---|---|---|
| Register | Her thread'in kendisi | thread başına yüzlerce | en hızlı | sadece o thread |
| Shared memory | SM'nin içinde | block başına ~48 KB | çok hızlı | block'un bütün thread'leri |
| L1 / L2 cache | SM'de / çipte | L2 birkaç MB | hızlı | otomatik |
| Global memory (VRAM) | Kartın belleği | 4 GB | yavaş (192 GB/s) | herkes |

RMSNorm gibi bir kernel'de asıl sınır global bellektir. Her değeri bir kez okuyup bir kez yazmamız gerekir. Hesap (birkaç çarpma ve toplama) bellekten çok daha hızlıdır. Bu tür kernel'lere **memory-bound** (bellek sınırlı) denir.

### PTX ve cuda-oxide

GPU, Rust ya da C++ çalıştırmaz. Derleyici kernel'i önce **PTX**'e çevirir. PTX, NVIDIA GPU'ları için bir ara dildir (bir tür GPU assembly'si). Program ilk açıldığında NVIDIA sürücüsü PTX'i kartın kendi makine koduna (SASS) çevirir.

cuda-oxide bu yolu izler: Rust → Rust MIR → Pliron IR → LLVM IR → PTX. `#[cuda_module]` ile işaretli modüldeki `#[kernel]` fonksiyonları PTX'e çevrilir ve PTX programın içine gömülür. Bizim PTX `sm_80` hedefiyle üretilir. Bizim kart `sm_86`'dır ve sürücü PTX'i onun için derler.

Bunu ölçtük: `gpu-bench.exe`, PATH'te CUDA toolkit olmadan da çalıştı. Kullanıcının bilgisayarında sadece NVIDIA sürücüsü yeterlidir.

PTX'i görmek için:

```powershell
scripts\cargo.cmd oxide build -- -p jeuvoice-gpu
```

Sonra `target\oxide-ptx\jeuvoice_gpu.ptx` dosyasını aç ve `.entry rmsnorm_f32` satırını ara.

## 4. Üç kernel ve ölçümleri

| Kernel | Süre (4096×1024 f32) | Hız | Bellek sınırının yüzdesi |
|---|---|---|---|
| `rmsnorm_naive_f32` | 932.6 µs | 36.0 GB/s | %19 |
| `rmsnorm_f32` | 192.2 µs | 174.6 GB/s | **%91** |
| `rmsnorm_f16` | 98.5 µs | 170.3 GB/s | %89 |

### 4.1. Naive kernel: bir thread, bir satır

En doğrudan yol: her thread bir satırın tamamını işler.

```rust
let row = thread::index_1d().get();
let base = row * d;
let mut sum = 0.0f32;
let mut c = 0;
while c < d {
    let v = unsafe { *x.add(base + c) };
    sum += v * v;
    c += 1;
}
```

Doğru çalışır ama yavaştır (%19). İki neden var:

1. **Az thread.** 4096 satır için sadece 4096 thread çalışır (16 block). Kartın 20 SM'i var ve her SM aynı anda 1536 thread çalıştırabilir. Kartın büyük kısmı boş kalır.
2. **Birleşmeyen okumalar (uncoalesced).** Bir warp'ın 32 thread'i aynı anda okur. Thread 0 satır 0'ın sütun 0'ını, thread 1 satır 1'in sütun 0'ını okur. Bu iki adres arasında `d = 1024` float (4 KB) vardır. Donanım 32 okumayı tek bir bellek işlemine birleştiremez ve 32 ayrı işlem yapar.

### 4.2. Hızlı kernel: bir block, bir satır

Şimdi her satıra 256 thread'lik bir block veriyoruz. Thread `t`, sütun `t`, `t+256`, `t+512` ve `t+768`'i okur:

```rust
let base = thread::blockIdx_x() as usize * d;
let tid = thread::threadIdx_x() as usize;
let mut sum = 0.0f32;
let mut c = tid;
while c < d {
    let v = unsafe { *x.add(base + c) };
    sum += v * v;
    c += ROW_BLOCK as usize;
}
```

Bu döngüye **strided loop** denir. İki sorunu da çözer:

1. **Çok thread:** 4096 block × 256 thread = 1 milyon thread. Bütün SM'ler dolar.
2. **Birleşen okumalar (coalesced):** Bir warp'ın 32 thread'i 32 komşu float'u okur (128 byte). Donanım bunu tek bir bellek işlemiyle yapar.

Ama şimdi yeni bir sorun var: her thread satırın sadece bir kısmının toplamını bilir. 256 kısmi toplamı birleştirmemiz gerekir. Buna **reduction** (indirgeme) denir. `block_sum` fonksiyonu bunu üç adımda yapar.

**Adım 1: Warp içinde toplama (shuffle).** Bir warp'ın 32 thread'i register'larındaki değerleri doğrudan birbirine verebilir. Bunun için belleğe yazmaları gerekmez. `warp::reduce_sum_f32` bunu 5 adımda yapar: her adımda thread'ler 16, 8, 4, 2 ve 1 sıra uzaktaki komşunun değerini alıp ekler. Sonunda 32 thread'in hepsi toplamı bilir. PTX'te bu 5 satırdır:

```
shfl.sync.bfly.b32 	%r7, %r45, 16, 31, -1;
shfl.sync.bfly.b32 	%r9, %r8, 8, 31, -1;
shfl.sync.bfly.b32 	%r11, %r10, 4, 31, -1;
shfl.sync.bfly.b32 	%r13, %r12, 2, 31, -1;
shfl.sync.bfly.b32 	%r15, %r14, 1, 31, -1;
```

`bfly` "butterfly" demektir: thread `i`, thread `i XOR 16` ile değer değiştirir, sonra `i XOR 8` ile, ve böyle devam eder.

**Adım 2: Shared memory ve barrier.** Artık 8 warp'ın her biri kendi toplamını biliyor. Bunları birleştirmek için warp'lar arası bir yer gerekir: shared memory. Her warp'ın 0. thread'i (lane 0) toplamı shared memory'deki kendi yuvasına yazar:

```rust
static mut WARP_SUMS: SharedArray<f32, ROW_WARPS> = SharedArray::UNINIT;
if warp::lane_id() == 0 {
    unsafe { WARP_SUMS[tid / 32] = warp_sum };
}
thread::sync_threads();
```

`sync_threads()` bir **barrier**'dır (PTX'te `bar.sync 0`). Block'un bütün thread'leri bu satıra gelene kadar hiçbiri devam etmez. Barrier olmasaydı, bir warp diğer warp'lar daha yazmadan okumaya başlayabilirdi. Buna **data race** denir ve sonuç rastgele bozulur.

**Neden `unsafe`?** Rust, shared memory'ye yazan iki thread'in aynı yuvaya yazmadığını kendisi kanıtlayamaz. Biz kanıtlıyoruz: sadece warp `k`'nin lane 0'ı yuva `k`'ye yazar. Bu açıklamayı her `unsafe` bloğun üstüne `SAFETY` yorumu olarak yazıyoruz. cuda-oxide bu kanıtı Lean 4 ile derleme sırasında da yapabilir (`#[verify]`). Bu, alıştırma 3'tür.

**Adım 3: Yayın okuma (broadcast).** Her thread 8 yuvayı okuyup toplar. Bir warp'ın 32 thread'i aynı adresi okuduğu için donanım değeri bir kez okur ve hepsine dağıtır. PTX'te derleyici bu 8 adımlık döngüyü tamamen açmıştır: 8 tane `ld.shared.b32` satırı görürsün.

Sonra her thread ölçeği hesaplar ve kendi sütunlarını yazar. Sonuç: 192.2 µs, sınırın %91'i. Naive kernel'den 4,9 kat hızlı.

**Bir not:** İkinci döngü `x`'i bellekten bir kez daha okur. Yine de %91'e ulaşıyoruz. Neden? Bir satır sadece 4 KB'tır. Block onu ilk döngüde okuduktan hemen sonra ikinci kez okur ve veri hâlâ L1/L2 cache'tedir. Bu yüzden ölçümde baytları "bir okuma + bir yazma" olarak sayıyoruz.

### 4.3. fp16 kernel: yarı bayt, yarı süre

Modelin ağırlıkları ve ara değerleri fp16'dır (2 bayt). `rmsnorm_f16` fp16 okur, bütün hesabı f32'de yapar ve fp16 yazar:

```rust
let v = unsafe { *x.add(base + c) } as f32;
```

Toplamı f32'de tutmak önemlidir. fp16'nın sadece ~3 basamak hassasiyeti vardır. 1024 karenin toplamı fp16'da hızla hassasiyet kaybeder.

Kernel memory-bound olduğu için süre taşınan bayta bağlıdır. Yarı bayt, yaklaşık yarı süre: 98.5 µs. Bu, GPU'da hızın çoğu zaman hesaptan değil, bellekten geldiğini gösteren en net örnektir.

## 5. Kartın sınırı ve ne kadar yaklaştık

Bellek sınırını kartın kendisinden hesaplıyoruz (`Gpu::open`):

```
bant genişliği = 2 × bellek saati × bus genişliği / 8
               = 2 × 6001 MHz × 128 bit / 8 ≈ 192 GB/s
```

`2 ×` çünkü GDDR bellek saatin iki kenarında da veri taşır.

RMSNorm 4096×1024 f32 için taşınan bayt: okuma 16 MB + yazma 16 MB = 32 MB. 192.2 µs'de bu 174.6 GB/s eder, yani sınırın %91'i. Pratikte %90'ın üstü çok iyidir: geri kalan kısım kernel başlatma, block'ların kuyruğu ve DRAM yenileme gibi kaçınılmaz giderlerdir.

Karşılaştırma için: cuBLAS GEMM aynı kartta fp16 tensor core sınırının (15,2 TFLOP/s) %88–95'ine ulaşıyor. GEMM **compute-bound**'dur: her baytı birçok kez kullanır ve sınırı hesap hızıdır. Ders 4'te bunu göreceğiz.

## 6. PTX ve Nsight Compute'ta neye bakmalı

### PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`)

`rmsnorm_f32` içinde şunlara bak:

- `ld.global.b32`: global bellekten okuma. Döngüde birer tane olmalı.
- 5 tane `shfl.sync.bfly.b32`: warp toplaması.
- `st.shared.b32` ve sonra `bar.sync 0`: shared memory'ye yazma ve barrier.
- 8 tane `ld.shared.b32`: açılmış (unrolled) döngü.
- `div.rn.f32`, `sqrt.rn.f32`, `rcp.rn.f32`: bölme, karekök ve `1 / x`. Bunlar tam hassasiyetli ve yavaş komutlardır, ama satır başına bir kez çalıştıkları için önemli değildir.

### Nsight Compute

Nsight Compute bir kernel'i donanım sayaçlarıyla ölçer. Kurulu: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1`.

```powershell
ncu --set full -k regex:rmsnorm -o rmsnorm target\release\gpu-bench.exe rmsnorm
```

Sonra `rmsnorm.ncu-rep` dosyasını Nsight Compute arayüzünde aç. Bakılacaklar:

- **Memory Throughput (%)**: DRAM sınırının yüzdesi. `rmsnorm_f32` için %90 civarı olmalı.
- **Memory Workload Analysis → L1/L2 hit rate**: ikinci okumanın cache'ten geldiğini gösterir.
- **Achieved Occupancy**: SM başına aktif warp oranı. Naive kernel'de düşük, hızlı kernel'de yüksek olmalı.
- **Source** sekmesi: her PTX/SASS satırının ne kadar zaman aldığı.

Not: Windows'ta sayaçlara erişim için `ncu`'yu bir kez yönetici olarak çalıştırman ya da NVIDIA Control Panel'de "Developer → Manage GPU Performance Counters → Allow access to all users" seçeneğini açman gerekebilir.

## 7. Alıştırmalar

1. **Block boyu.** `ROW_BLOCK`'u 128 yap (ve `ROW_WARPS` buna göre 4 olur). Testleri ve ölçümü çalıştır. Daha mı hızlı, daha mı yavaş? Neden? İpucu: `d = 1024` için her thread kaç değer işler, ve SM başına kaç block aynı anda çalışabilir?
2. **Vektör okuma.** `rmsnorm_f16`'da her thread bir seferde bir fp16 (2 bayt) okur. İki fp16'yı tek bir 4 baytlık okumayla oku (`f16x2`). Ölçümü karşılaştır. İpucu: `cuda_device::f16x2` modülüne bak.
3. **`unsafe` yerine kanıt.** `block_sum`'daki shared memory erişimini `#[verify(no_oob_idx(WARP_SUMS), no_data_race(WARP_SUMS), uniform_barriers)]` ile kanıtla ve `unsafe` bloklarını kaldır. Örnek: cuda-rust deposunda `cuda-oxide/examples/safe_warp_reduce`.
4. **(Ek) Hızlı karekök.** `1.0 / x.sqrt()` yerine yaklaşık ters karekök kullan (`cuda_device` içinde `rsqrt_approx_f32`). PTX'te ne değişir? Testler hâlâ geçer mi? Neden bu kernel'de hız farkı neredeyse sıfırdır?

## Bu fazda öğrendiklerimiz (araç zinciri notları)

- **Kernel argümanları:** cuda-oxide'ın senkron başlatma kodu sadece bütün bir `DeviceBuffer` ya da ham pointer (`*const T`, `*mut T`) kabul eder. Bizim tensörlerimiz tek bir büyük bellek bloğunun (arena) parçalarıdır. Bu yüzden arena kernel'leri ham pointer ve uzunluk alır.
- **MinGW çakışması:** PATH'te `C:\mingw64` varsa, libclang yanlış C başlık dosyalarını okur ve derleme bozulur. `scripts\cargo.cmd oxide` MinGW'yi PATH'ten çıkarır.
- **PTX dosya yarışı:** cuda-oxide bir crate'in PTX'ini crate adıyla tek bir dosyaya yazar. Cargo kütüphaneyi ve kütüphane testini aynı anda derlediğinde iki derleme aynı dosyaya yazıp okudu. Bu yüzden `jeuvoice-gpu`'da `[lib] test = false` var ve bütün testler `tests/` içinde.
- **fp16 tipi:** Kernel'lerde Rust'ın yerleşik `f16` tipini kullanıyoruz (`#![feature(f16)]`). Host tarafında aynı 2 baytlık düzeni `half::f16` taşır.
