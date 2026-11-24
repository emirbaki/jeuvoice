# Ders 6: conv1d (Whisper encoder'ının girişi)

Bu derste 1 boyutlu konvolüsyonu iki yoldan yazıyoruz: doğrudan (direct) ve **im2col + GEMM**. Sonuç çok net: konvolüsyonu bir matris çarpımına çevirirsek tensor core'ları kullanabiliriz ve 30 kattan fazla hızlanırız. Bu, derin öğrenmede en sık kullanılan hilelerden biridir.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`conv1d_k3_direct_f16`, `im2col_k3_f16`) ve `crates/jeuvoice-gpu/src/ops/conv1d.rs`.

Ölçümler: RTX 3050 Ti Laptop, fp16 tensor sınırı 15,2 TFLOP/s. `target\release\gpu-bench.exe conv1d`.

## 1. Ne yapar?

Whisper encoder'ı mel spektrogramı (`[128, 3000]`: 128 frekans bandı, 3000 zaman adımı) iki konvolüsyondan geçirir:

| Katman | Girdi | Çıktı | Adım (stride) |
|---|---|---|---|
| conv1 | `[128, 3000]` | `[1280, 3000]` | 1 |
| conv2 | `[1280, 3000]` | `[1280, 1500]` | 2 |

Çekirdek boyu 3, kenarlarda 1 sıfır dolgu (padding):

```
y[co][t] = Σ_ci Σ_k  w[co][ci][k] · x[ci][t·stride + k − 1]
```

`x` dışına taşan indeks (−1 ya da `t_in`) sıfır sayılır. `t_out = (t_in − 1) / stride + 1`. Her iki katmandan sonra bias ve GELU gelir (ders 1'deki `bias_gelu`). Bizim `conv1d` bias eklemez: bias'ı `bias_gelu` ile tek seferde ekleriz.

## 2. Neden gerekli, ne sıklıkla çalışır?

- **conv1**: her zaman adımında 3 komşu mel çerçevesini birlikte görür ve 128 kanalı modelin genişliği olan 1280'e çıkarır.
- **conv2**: stride 2 ile zaman adımını yarıya indirir (3000 → 1500). Encoder'ın geri kalanı 1500 token ile çalışır. Attention maliyeti token sayısının karesiyle büyüdüğü için bu yarıya indirme önemlidir.

Her 30 saniyelik pencere için iki konvolüsyon bir kez çalışır. Ama conv2 büyüktür: `2 × 1280 × 1500 × 1280 × 3 ≈ 14,7 GFLOP`. Bu, encoder'ın bir bloğundaki bütün GEMM'lere yakın bir iş.

## 3. Doğrudan kernel: bir thread, bir çıktı

```rust
let (co, t) = (i / t_out, i % t_out);
let mut s = 0.0f32;
for ci in 0..c_in {
    for k in 0..3 {
        let p = t * stride + k;          // dolgulu indeks
        if p >= 1 && p <= t_in {
            s += w[(co * c_in + ci) * 3 + k] * x[ci * t_in + p - 1];
        }
    }
}
y[i] = s;
```

| Şekil | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|
| conv1 (128 → 1280) | 5,1–6,4 ms | 0,5 TFLOP/s | %3–4 |
| conv2 (1280 → 1280, stride 2) | 30,8–43,6 ms | 0,3–0,5 TFLOP/s | %2–3 |

Doğru ama çok yavaş. Süre de koşudan koşuya değişiyor: kart ısındıkça saat hızı düşer ve bu kernel saat hızına çok bağlıdır. Neden yavaş?

1. **Tensor core yok.** Her çarpma bir skaler `fma.rn.f32`'dir. Kartın skaler f32 hızı tensor core'ların yaklaşık dörtte biridir.
2. **Her veri tekrar tekrar okunur.** conv2'de her çıktı 3840 çarpım yapar ve her çarpım için iki değer yükler. Aynı `x` değerini 1280 farklı çıktı kanalı okur. L1/L2 cache'in bir kısmı yardım eder, ama yükleme komutları yine çalışır.
3. **Ders 4 ile aynı sorun.** Bu döngü aslında bir matris çarpımıdır ve biz onu ders 4'teki "naive GEMM" gibi yazdık (orada da %3 civarıydı).

Doğrudan kernel'i hızlandırmanın yolu, ders 4'teki gibi shared memory ve register döşemesi eklemektir. Ama o zaman ikinci bir GEMM yazmış oluruz. Daha iyi bir yol var.

## 4. im2col: konvolüsyonu GEMM'e çevir

Gözlem: `w` bellekte `[c_out, c_in, 3]` biçimindedir. Bu aynı baytlar `[c_out, c_in·3]` biçiminde bir matristir. Satır `co`, sütun `ci·3 + k`.

Şimdi bir "pencere matrisi" `col` kuralım: `[c_in·3, t_out]`, satır `ci·3 + k`, sütun `t`:

```
col[ci·3 + k][t] = x[ci][t·stride + k − 1]     (dışarıdaysa 0)
```

O zaman konvolüsyon tam olarak bir matris çarpımıdır:

```
y [c_out, t_out] = w [c_out, c_in·3] · col [c_in·3, t_out]
```

`im2col_k3_f16` kernel'i `col`'u doldurur (bir thread, bir eleman: sadece kopyalama). Sonra `blas.gemm_f16(w, N, col, N, y, c_out, t_out, c_in·3)` çağrılır. Bu adım cuBLAS'ın tensor core GEMM'ini kullanır.

| Şekil | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|
| conv1 | **246,7 µs** | 12,0 TFLOP/s | %79 |
| conv2 | **942,4 µs** | 15,6 TFLOP/s | %103 |

conv1'de 21–26 kat, conv2'de 32–46 kat hızlı.

**Bedeli:** fazladan bellek ve fazladan bir kopyalama. `col`, her `x` değerinin üç kopyasını tutar:

- conv1: `384 × 3000 × 2 B = 2,3 MB`
- conv2: `3840 × 1500 × 2 B = 11,5 MB`

Bu kopyalamayı yazıp okumak, conv2'de yaklaşık 23 MB'lık ek bellek trafiğidir. 192 GB/s'de bu 120 µs eder. GEMM'den kazandığımız 30 ms'nin yanında küçüktür. 4 GB'lık kartta 11,5 MB'lık bir arena tensörü de sorun değildir.

**%103 nasıl olur?** `DeviceInfo` içindeki 15,2 TFLOP/s, sürücünün bildirdiği saat hızıyla hesaplanır. Kart kısa yüklerde bu saatin üstüne çıkabilir (boost). Yüzde 100'ün üstü bir hata değildir. Sınır tahminimiz biraz düşüktür.

## 5. Sınır ve ne kadar yaklaştık

GEMM boyutları:

| Katman | m (c_out) | n (t_out) | k (c_in·3) |
|---|---|---|---|
| conv1 | 1280 | 3000 | 384 |
| conv2 | 1280 | 1500 | 3840 |

conv2'nin `k`'si büyüktür. Her çıktı uzun bir toplam yapar, bu yüzden tensor core'lar dolu çalışır (%103). conv1'in `k`'si sadece 384'tür. GEMM'in her çıktısı daha az iş yapar, ve `y`'yi yazmak (7,7 MB) ile im2col kopyası işin büyük kısmını oluşturur. Bu yüzden %79'da kalır.

Encoder'da ürün her iki katmanda `ConvKind::Im2col` kullanır.

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `conv1d_k3_direct_f16`: iç içe iki döngü. İç gövdede iki `ld.global.b16`, iki `cvt.f32.f16` ve bir `fma.rn.f32` var. Bir çarpım için beş komut demektir. Ayrıca sınır kontrolü için `setp` ve dallanma var.
- `im2col_k3_f16`: hiç çarpım yok. Bir `ld.global.b16` (ya da sıfır) ve bir `st.global.b16`. Saf bir kopyalama kernel'i.

Nsight Compute:

```powershell
ncu --set full -k regex:conv1d_k3_direct -o conv target\release\gpu-bench.exe conv1d_direct
```

- **Compute (SM) Throughput** yüksek, **Memory Throughput** düşük ise darboğaz komut sayısıdır, bellek değildir.
- **Instruction Statistics**: `FFMA` ve `LDG` sayılarını karşılaştır. Her FMA için iki yükleme görürsün.
- GEMM tarafında (`regex:gemm` ya da cuBLAS'ın `sm80_xmma` / `ampere_h1688gemm` gibi kernel adları) **Tensor Pipe Utilization** değerine bak.

## 7. Alıştırmalar

1. **Bias'ı GEMM'e kat.** cuBLAS GEMM'i `beta = 1` ile çağır ve `y`'yi önceden bias ile doldur. Ayrı bir `bias_gelu` yerine sadece GELU kalır. Ne kadar kazanırsın?
2. **im2col'suz GEMM.** stride 1 için `col`'un satırları `x`'in kaydırılmış kopyalarıdır. 3 ayrı GEMM ile (her `k` için bir tane, `beta = 1` ile toplayarak) `col`'u hiç yazmadan konvolüsyon yap. `x`'in kaydırılmış başlangıcı ve sınırlar için ne yapman gerekir?
3. **Doğrudan kernel'e shared memory.** Bir block'a 32 çıktı kanalı ve 64 zaman adımı ver. `w` ve `x` döşemelerini shared memory'ye yükle (ders 4'teki `gemm_tiled_f16` gibi). Ne kadar hızlanır? im2col'a yaklaşır mı?
4. **Isınma.** `gpu-bench.exe conv1d_direct`'i art arda 5 kez çalıştır. Süreler neden değişiyor? `nvidia-smi -q -d CLOCK` ile saat hızına bak.
