# Ders 4: GEMM (matris çarpımı) — cuBLAS neden hızlı, ve tensor core ile ona nasıl yaklaşırız?

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

### 4.4. Kendi tensor core kernel'imiz (2,5 ms → 0,38 ms)

İlk denemede bu kernel yazılamadı: `ptx_asm!` makrosu test profilinde derlenmiyordu. Sonra cuda-oxide'in **üretilmiş** sarmalayıcılarında aradığımız şeyi bulduk: `cuda_device::wmma::mma_m16n8k8_f32_f16`. Belgesi "PTX 6.5 ve sm_75+" der. `ldmatrix_x1/x2/x4` (ve `_trans`) da orada. Yani `ptx_asm!` gerekmedi. **Ders:** "yok" demeden önce kütüphanenin üretilmiş kodunu da ara.

#### Adım 0: Fragment düzenini doğrula

Tensor core komutunda **thread'lerin hangi değeri tuttuğu** sabittir. Buna "fragment düzeni" denir. `m16n8k8` için, warp içindeki şerit (lane) `l` için `g = l / 4` (grup, 0..7) ve `t = l % 4` (grup içindeki sıra, 0..3):

| Fragment | Register | İçerik |
|---|---|---|
| A (16 × 8, satır düzeninde) | `a[0]`, `a[1]` | `a[0]` = A[g][2t, 2t+1], `a[1]` = A[g+8][2t, 2t+1] |
| B (8 × 8, sütun düzeninde) | `b` | B[2t, 2t+1][g] (g sütununun iki k satırı) |
| C, D (16 × 8, f32) | `c[0..4]` | `c[0], c[1]` = C[g][2t, 2t+1], `c[2], c[3]` = C[g+8][2t, 2t+1] |

İki f16 değer bir `u32` içinde durur: küçük sütun alt 16 bitte.

Önce tek bir warp ile tek bir `16×8×8` çarpım yapan bir **sonda** (probe) kernel'i yazdık (`mma_probe_f16`). Her thread değerlerini tabloya göre global bellekten okur. Test (`tests/tensor_cores.rs`) iki şey kontrol eder: rastgele matrislerde CPU ile aynı sonuç, ve A'nın her satırında tek bir 1 olduğunda D'nin tam olarak B'nin bir satırı olması. Düzen yanlış olsaydı ikinci test hangi satır ve sütunun karıştığını söylerdi. İlk koşuda geçti. PTX'te:

```
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%r10, %r11, %r12, %r13}, {%r1, %r2}, {%r15}, ...
```

#### Adım 1: `gemm_tc_f16` (v1)

| Seçim | Değer | Neden |
|---|---|---|
| Block tile | 64 × 64, k adımı 32 | `regtile` ile aynı tile, karşılaştırma adil olsun |
| Warp'lar | 4 warp (2 × 2), her biri 32 × 32 | 2 satır döşemesi (16) × 4 sütun döşemesi (8) = 8 accumulator, 32 f32 register |
| Shared satır adımı | 32 + 8 ve 64 + 8 half | 80 ve 144 bayt: `ldmatrix` 16 baytlık hizalı satır ister. Ek 8 half satırları farklı bank'lara kaydırır |
| A fragment'i | `ldmatrix_x2` | şerit 0–15 satır adreslerini verir (0–7 ilk 8×8 matris, 8–15 ikinci). sm_75'te 32 şeridin hepsi geçerli adres vermeli, bu yüzden 16–31 aynısını tekrarlar |
| B fragment'i | `ldmatrix_x4_trans` | B shared'da satır düzeninde (`[k][n]`), fragment sütun ister. `.trans` bunu çevirir. x4, bir warp'ın 4 sütun döşemesini tek komutta yükler |

Kenarlar: tile kopyalanırken matrisin dışı sıfır yazılır. Sıfırlar toplama bir şey eklemez. Yazarken her çıktı satır ve sütun kontrolüyle yazılır (`67 × 129 × 96` ve `17 × 33 × 17` testleri).

#### Adım 2 ve 3: kopyalamayı genişlet, sonra önceden yükle

| Kernel | Değişiklik | Süre (1500 × 1280 × 1280) | Hız |
|---|---|---|---|
| `gemm_regtile_f16` | ders 4'ün en iyi SIMT kernel'i | 2533 µs | 1,9 TF/s |
| `gemm_tc_f16` (v1) | tensor core, kopyalama 2 baytlık okumalarla | 658–662 µs | 7,4 TF/s |
| `gemm_tc_vec_f16` (v2) | kopyalama 16 baytlık okumalarla | 384–394 µs | 12,5–12,8 TF/s |
| `gemm_tc_pre_f16` (v3) | v2 + bir sonraki tile'ı register'a önceden yükle | **379–381 µs** | 12,9–13,0 TF/s |
| cuBLAS | | 287–364 µs | 13,5–17,1 TF/s |

Tensor core tek başına 3,8 kat kazandırdı. Ama v1'de vaktin yarısı **tile kopyalamasında** gidiyordu: her thread 32 tane 2 baytlık okuma yapıyordu ve tensor core'lar bunu bekliyordu.

**v2'deki tuzak:** İlk v2 denemesinde 8 half'ı `[u32; 4]` olarak okuduk. PTX'te tek bir 16 baytlık okuma beklerken **dört tane** `ld.global.b32` gördük. Sebep: okumanın genişliğini **tipin hizalaması** belirler. `[u32; 4]` 4 bayt hizalıdır, derleyici 16 baytlık hizayı varsayamaz. Çözüm, 16 bayt hizalı bir tip:

```rust
#[repr(C, align(16))]
pub struct Q16(pub [u32; 4]);
```

Bununla PTX'te 12 tane `ld.global.v4` ve 4 tane `st.shared.v4` var, hiç `ld.global.b32` yok. (cuda-oxide'in `vector.rs` belgesi de bunu anlatır.)

**v3 neden az kazandırdı (%1)?** Önceden yükleme, bir sonraki tile'ın okumalarını tensor core'lar çalışırken başlatır. Ama her warp tile başına sadece 4 × 8 = 32 `mma` yapar. Bu, ~500 devirlik bir global okumayı saklamaya yetmez. cuBLAS bu kartta (sm_86) daha büyük tile'lar ve `cp.async` kullanır: global bellekten shared'a doğrudan, register'a uğramadan, birden çok tile'ı aynı anda. `cp.async` sm_80'dir. Bizim PTX hedefimiz sm_75 olduğu için kullanamayız.

## 5. Kartın sınırı ve ne kadar yaklaştık

İki farklı tavan var:

| Tavan | Değer | Bizim en iyimiz |
|---|---|---|
| fp16 tensor core (cuBLAS) | 15,2-19 TFLOP/s | 13,0 TF/s (`gemm_tc_pre_f16`, %85) |
| fp32 FMA birimleri (elle yazılmış SIMT) | ~9 TFLOP/s | 2,0 TF/s (23%) |

Tensor core kernel'imiz cuBLAS'ın süresinin 1,32 katında (379 µs ve 288 µs). Eski SIMT kernel'imiz ise SIMT tavanının %23'ünde. İyi bir SGEMM %80-90'a çıkar. Fark, deneyimli bir GEMM'in yaptığı üç şeyden gelir:

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
- `gemm_tc2_f16`: PTX'te `mma.sync.aligned.m16n8k8`, `ldmatrix.sync.aligned.m8n8.x2` ve `x4.trans`, `ld.global.v4` (16 bayt) ve `st.shared.v4`. Nsight'ta **Tensor Pipe Utilization** ve **Warp State → Stall Long Scoreboard** (global okuma beklemesi) değerlerine bak. v1 ile v3 arasındaki farkın çoğu bu bekleme satırında görünür.

## 7. Alıştırmalar

1. **8×8 register bloğu.** 64 accumulator, 8+8 yükleme. Aynı 64×64 tile ile 8×8 blok, thread başına 16 FMA / 2 okuma verir. Ölç: hangi noktada register baskısı hızı düşürür?
2. **Vektörel shared okuma.** `RA`'yı `[f16; 8]` olarak oku (16 bayt). PTX'te `ld.shared.v4.b32` görüyor musun? Süre değişiyor mu?
3. **Bank çakışması.** Tile'ı `64 + 8` sütunla ayır (satır adımı 72 f16). `ncu`'da bank conflict sayısı ne oldu, süre ne oldu?
4. **Kenar maliyeti.** `1500 x 1280 x 1280` (kenarlar var) ile `1536 x 1280 x 1280` (tam tile) ölç. Kenar maskeleri ne kadar sürüyor?
5. **Daha büyük tensor core tile'ı.** `gemm_tc2_f16`'yı 128 × 128 tile ve 8 warp (her biri 64 × 32) ile yaz. Her warp tile başına iki kat `mma` yapar, ve önceden yükleme daha çok işe yarar. cuBLAS'a ne kadar yaklaşıyor? Register sayısı (`ncu`'da "Registers Per Thread") ne oldu?
6. **Neden 4 GEMM?** QKV projeksiyonunu tek `[1280, 3840]` matrisi yerine üç ayrı GEMM olarak yapmak ne kaybettirir? Tek GEMM'i ölç, üçünü ölç.
7. **A için `ldmatrix_x4`.** A fragment'lerini iki k adımı için tek `ldmatrix_x4` ile yükle (16 × 16 bir parça). Komut sayısı ve süre ne değişti?
