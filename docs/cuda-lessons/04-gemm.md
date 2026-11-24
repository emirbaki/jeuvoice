# Ders 4: GEMM (matris çarpımı) — cuBLAS neden bizden 9 kat hızlı?

Whisper encoder'ının işinin neredeyse tamamı matris çarpımıdır. Bu derste aynı çarpımı üç kez yazıyoruz: önce saf hâliyle, sonra shared memory ile, sonra register bloklarıyla. Her adımda ölçüyoruz. Sonunda cuBLAS'ın neden bu kadar önde olduğunu görüyoruz: **tensor core**lar.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`gemm_naive_f16`, `gemm_tiled_f16`, `gemm_regtile_f16`) ve `crates/jeuvoice-gpu/src/ops/gemm_lesson.rs`.

Ölçümler: RTX 3050 Ti Laptop, fp16 tensor sınırı 15,2 TFLOP/s, `1500 x 1280 x 1280`. `target\release\gpu-bench.exe gemm`.

## 1. Ne yapıyor?

```
C[m, n] = A[m, k] · B[k, n]
```

`C[i][j] = Σ_p A[i][p] · B[p][j]`. A ve B f16 gelir, toplama f32'de yapılır, sonuç f16 yazılır.

Whisper'da dört yerde çalışır (her blokta):

| Katman | Şekil |
|---|---|
| QKV projeksiyonu | `[1500, 1280] × [1280, 3840]` |
| attention çıkışı | `[1500, 1280] × [1280, 1280]` |
| MLP 1. katman | `[1500, 1280] × [1280, 5120]` |
| MLP 2. katman | `[1500, 5120] × [5120, 1280]` |

Bunlara dikkat skorları (`Q·Kᵀ`) ve `probs·V` de eklenir.

## 2. Neden gerekli, ne sıklıkla çalışır?

GEMM modelin **hesap yükünün** neredeyse tamamıdır. Bir GEMM `2·m·n·k` işlem yapar. `1500 × 1280 × 1280` için 4,9 GFLOP. Bir blokta bu boyutta iki tane, daha büyükleri de var; kabaca **blok başına 20 GFLOP**. 32 blok için **640 GFLOP** bir 30 saniyelik pencerede.

Ölçülen hızla (cuBLAS, 13,5 TFLOP/s) bu 47 ms eder. Yani encoder'ın toplam süresini GEMM belirler. Bunu %89'dan %40'a düşürmek tüm uygulamayı iki katına çıkarır. Bu yüzden **ürün kodu cuBLAS kullanır** ve bu dersteki kernel'ler öğrenme içindir.

## 3. Naive hâl: her çıktı için bir thread

```rust
let mut sum = 0.0f32;
while p < k { sum += a[row*k + p] as f32 * b[p*n + col] as f32; p += 1; }
c[row*n + col] = sum as f16;
```

Ölçüm: **9,3 ms, 0,5 TFLOP/s, tensor sınırının %3'ü.**

Neden bu kadar yavaş? Her thread `k = 1280` elemanı A'dan ve B'den okur. Toplam trafik: `m·n·k·2·2` bayt = **9,8 GB**. Gerekli olan sadece `A + B + C ≈ 11 MB`. Aritmetik yoğunluk (arithmetic intensity) = 1 işlem / 1 bayt. Bu kart 192 GB/s taşır, yani en iyi hâlde 9,8 GB / 192 GB/s = 51 ms. Ölçülen 9,3 ms daha iyi çıktı, çünkü L2 cache (4 MB) aynı satırları tekrar tekrar tutuyor. Yine de sınır bellek.

**Ders:** GEMM'de kazanç, aynı veriyi **birden fazla çıktı için** kullanmaktır. Buna veri yeniden kullanımı (data reuse) denir.

## 4. Hızlandırma adımları

### 4.1. Shared memory tile'ları (9,3 → 6,7 ms)

Blok, A ve B'nin 16×16'lık parçalarını shared memory'ye bir kez yükler ve 256 çıktı için kullanır:

```rust
while p0 < k {
    TA[ty][tx] = a[row][p0 + tx];       // koşullu, kenar dışı 0
    TB[ty][tx] = b[p0 + ty][col];
    thread::sync_threads();
    while pp < 16 { sum += TA[ty][pp] * TB[pp][tx]; pp += 1; }
    thread::sync_threads();
    p0 += 16;
}
```

Global trafik 16 kat azalır: her k-step'te 2 × 16 × 16 eleman okunur, 256 çıktı üretilir.

Ölçüm: **6,7 ms, 0,7 TFLOP/s, %5.** Kazanç var ama az (1,4 kat). Neden?

1. **Bank çakışması.** Bir satır 16 f16 = 32 bayt. Warp'taki thread'ler aynı banklara düşer. Shared memory 32 banktan oluşur ve aynı banka gelen istekler sıraya girer.
2. **Thread başına iş az.** Her thread 16 FMA için 32 shared okuma yapar. `ld.shared` komutları FMA'lardan fazla yer kaplar. GPU'da komut yuvası (issue slot) sınırlıdır.

### 4.2. Register bloğu: 4×4 çıktı (6,7 → 2,5 ms)

Şimdi her thread **16 çıktı** hesaplar: 4 satır × 4 sütun. 64×64 tile shared memory'de durur, her thread kendi 4×4'ünü register'da tutar:

```rust
while pp < 64 {
    let bv = [RB[pp][tx*4], RB[pp][tx*4+1], RB[pp][tx*4+2], RB[pp][tx*4+3]];  // 4 okuma
    let mut r = 0;
    while r < 4 {
        let av = RA[ty*4 + r][pp];           // 1 okuma
        acc[r][0] += av * bv[0];  ...  acc[r][3] += av * bv[3];   // 4 FMA
        r += 1;
    }
    pp += 1;
}
```

8 shared okuma → 16 FMA. Okuma başına 2 işlem (önce 0,5'ti).

Ölçüm: **2,5 ms, 2,0 TFLOP/s, %13.**

### 4.3. cuBLAS: tensor core (2,5 ms → 0,29-0,36 ms)

Aynı şekil, cuBLAS ile:

| Kernel | Süre | Hız | Tensor sınırının yüzdesi |
|---|---|---|---|
| `gemm_naive_f16` | 9,3-10,9 ms | 0,5 TF/s | %3 |
| `gemm_tiled_f16` | 6,7-7,2 ms | 0,7 TF/s | %5 |
| `gemm_regtile_f16` | 2,5 ms | 2,0 TF/s | %13 |
| **cuBLAS `gemm_f16`** | **0,29-0,36 ms** | 13,5-17 TF/s | %89-113 |

cuBLAS bizim en iyimizden **9 kat** hızlı. Aradaki fark algoritmada değil, **donanım biriminde**: matris çarpımını normal FMA birimleriyle değil, tensor core'larla yapar.

Tensor core'lar tek komutta küçük bir matris parçasını çarpar:

```
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32   // sm_75 (Turing) ve sonrası
```

Bir warp, 16×8 × 8×8'lik bir çarpımı **tek komutta** yapar: 16·8·8 = 1024 çarpma-toplama. Aynı işi normal FMA birimleriyle yapmak 32 thread × 32 FMA = 32 komut sürer. Yani tensor core kaba bir hesapla 32 kat, gerçekte saat başına ~8-16 kat daha hızlıdır. Kartın "15,2 TFLOP/s fp16 tensor" değeri buradan gelir; normal fp32 FMA birimlerinin teorik sınırı 20 SM × 128 çekirdek × 1,76 GHz × 2 ≈ **9 TFLOP/s**'tir.

**Bu plan için önemli not:** `mma.sync` kernel'i bu depoya eklenmedi. Sebep bir toolchain hatası: `ptx_asm!` makrosu test profilinde çalışmıyor ("ptx_asm template must be a byte string literal, got MIR operand Move(_3)"), ve cuda-oxide'in hazır `mma` sarmalayıcıları `m16n8k16` için `sm_80` ister; bizim PTX hedefimiz `sm_75`. Bu engel `docs/development-log.md` içinde ertelenmiş iş olarak duruyor, ve alıştırma 5 olarak aşağıda. Ürün yolu zaten cuBLAS olduğu için encoder bundan etkilenmez.

## 5. Kartın sınırı ve ne kadar yaklaştık

İki farklı tavan var:

| Tavan | Değer | Bizim en iyimiz |
|---|---|---|
| fp16 tensor core (cuBLAS) | 15,2-19 TFLOP/s | 2,0 TF/s (13%) |
| fp32 FMA birimleri (elle yazılmış SIMT) | ~9 TFLOP/s | 2,0 TF/s (23%) |

Yani kendi kernel'imiz SIMT tavanının %23'ünde. İyi bir SGEMM %80-90'a çıkar. Fark, deneyimli bir GEMM'in yaptığı üç şeyden gelir:

1. **Daha büyük register bloğu** (8×8 = 64 çıktı).
2. **Vektörel shared okuma** (`ld.shared.v4.b32`, 128 bit).
3. **Bank çakışmasını kaldırmak** (tile'ı 64+8 sütunla hizalamak, "swizzle").

Bu üçü de bu derste ölçülebilir; alıştırmalarda var.

Not: tabloda cuBLAS'ın yüzdesi bazen %100'ü geçiyor. Sebep, kartın saat hızının (clock boost) sabit olmaması: `gpu.info()`'daki 15,2 TFLOP/s muhafazakâr bir değer. Soğuk GPU'da ilk koşu düşük, ısınmış GPU'da yüksek okunur. Tabloyu iki kez koş.

## 6. PTX ve Nsight'ta neye bakmalı

Kendi kernel'lerimiz, PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `gemm_regtile_f16`: iç döngüde 16 × `fma.rn.f32` ve 5 × `ld.shared.u16`. Oran 3,2 işlem/okuma. Oranı yükseltmek (8×8) doğrudan hız demektir.
- `gemm_naive_f16`: iç döngüde 2 × `ld.global.b16` + 1 `fma`. `ld.global` gecikmesi (~500 devir) ancak çok sayıda warp ile saklanır.

Nsight Compute:

```powershell
ncu --set full -k regex:gemm_regtile -o gemm target\release\gpu-bench.exe gemm
```

- **Speed Of Light → Compute (SM)**: %13 civarı. **Memory**: daha düşük. Yani kernel hesaba değil, komut yayınına ve shared gecikmesine takılıyor.
- **Memory Workload Analysis → Shared Memory → Bank Conflicts**: `tiled` sürümünde yüksek çıkar. Padding ile düşür.
- **Occupancy**: `regtile` 16 f32 accumulator + 8 yükleme register'ı tutar; 256 thread'lik blokla 20 SM'de kaç blok sığdığına bak.
- cuBLAS kernel'ini SASS'ta görmek için: `cuobjdump -sass` ya da Nsight'ta `HMMA` komutlarını ara. Fark orada tek satırda görünür.

## 7. Alıştırmalar

1. **8×8 register bloğu.** 64 accumulator, 8+8 yükleme. Aynı 64×64 tile ile 8×8 blok, thread başına 16 FMA / 2 okuma verir. Ölç: hangi noktada register baskısı hızı düşürür?
2. **Vektörel shared okuma.** `RA`'yı `[f16; 8]` olarak oku (16 bayt). PTX'te `ld.shared.v4.b32` görüyor musun? Süre değişiyor mu?
3. **Bank çakışması.** Tile'ı `64 + 8` sütunla ayır (satır adımı 72 f16). `ncu`'da bank conflict sayısı ne oldu, süre ne oldu?
4. **Kenar maliyeti.** `1500 x 1280 x 1280` (kenarlar var) ile `1536 x 1280 x 1280` (tam tile) ölç. Kenar maskeleri ne kadar sürüyor?
5. **Tensor core.** `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` kernel'ini yaz. Cihaz fonksiyonu içinde `cuda_device::ptx::__ptx_asm_out_7` doğrudan çağrısıyla test profilinde de derlenebildiğini doğruladık (bkz. `docs/development-log.md`). Fragment yerleşimi: bu dersin 6. bölümü ve PTX dokümanı. cuBLAS'ın %89'una ne kadar yaklaşıyor?
6. **Neden 4 GEMM?** QKV projeksiyonunu tek `[1280, 3840]` matrisi yerine üç ayrı GEMM olarak yapmak ne kaybettirir? Tek GEMM'i ölç, üçünü ölç.
