# Ders 11: SwiGLU

Qwen3'ün MLP'si Whisper'ınkinden farklı: iki paralel projeksiyon ve aralarında bir **kapı** (gate). Bu dersin konusu kapının aktivasyonu, SwiGLU. Ders 1'deki "iki eleman, bir thread" hilesini burada tekrar deniyoruz ve **hiçbir şey kazandırmıyor**. Sebebini ölçerek buluyoruz.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`silu`, `swiglu_f16`, `swiglu_f16x2`), `crates/jeuvoice-gpu/src/ops/swiglu.rs`.

Ölçümler: RTX 3050 Ti Laptop, bellek sınırı 192 GB/s, `[486, 3072]` f16. `target\release\gpu-bench.exe swiglu_f16`.

## 1. Ne yapıyor?

```
out[i] = silu(gate[i]) · up[i]
silu(x) = x / (1 + exp(-x))
```

Qwen3'ün MLP'si şöyle:

```
gate = h · W_gateᵀ        [1024 → 3072]
up   = h · W_upᵀ          [1024 → 3072]
mid  = silu(gate) · up    [3072]        <- bu ders
down = mid · W_downᵀ      [3072 → 1024]
```

`silu` (sigmoid linear unit) kapının ne kadarının geçeceğine karar verir. `up` ise taşınan değerdir. Çarpım, "kapı açıksa değeri geçir" demektir. Qwen3'te `gate` ve `up` ayrı matrislerdir (Llama ailesinin standardı); Whisper'da tek bir MLP ve GELU vardı (ders 1).

## 2. Neden gerekli, ne sıklıkla çalışır?

Aktivasyon olmasaydı iki doğrusal katman tek bir doğrusal katmana eşit olurdu ve model hiçbir doğrusal olmayan şey öğrenemezdi. Kapı, modelin "bu özellik bu bağlamda önemli mi" kararını öğrenmesini sağlar.

Sıklık: her katmanda bir kez, her geçişte. 28 katman × 32 geçiş = **896 kez** bir render'da. Boyut her seferinde `[486, 3072]` f16: 1,5M eleman.

## 3. Naive hâl: bir eleman, bir thread

```rust
let g = *gate.add(i) as f32;
let u = *up.add(i) as f32;
*out.add(i) = (silu(g) * u) as f16;
```

Ölçüm: **53,8 µs, 166 GB/s, sınırın %87'si.**

Dikkat: bu, ders 1'deki `bias_gelu`'nun ilk sürümünden (%56) çok daha iyi. Fark nerede? `bias_gelu` her elemanda `i % d` hesaplıyordu (tam sayı bölme, GPU'da pahalı) ve `erf` polinomu çalıştırıyordu. Burada `%` yok (bias yok) ve `silu` sadece bir bölme ile bir `exp`.

## 4. Hızlandırma denemesi: çift okuma — kazanç yok

Ders 1'de `[f16; 2]` okuma %56'dan %90'a çıkarmıştı. Aynı hileyi burada denedik: thread başına iki komşu eleman, tek `[f16; 2]` yüklemesi.

Ölçüm (üç koşu, izole): **53,3 µs, 168 GB/s, %87.** Fark yok. İki sürüm de aynı sınırda.

Neden? Üç sebep:

1. **Eleman başına 6 bayt hareket ediyor** (iki okuma, bir yazma). x1 sürümünde her thread zaten 6 baytlık iş yapıyor; birleştirmenin paylaştıracağı "sabit yük" (indeks, `%`, sınır kontrolü) neredeyse yok.
2. **Warp zaten 64 baytlık okuma yapıyor** (32 thread × 2 bayt × 2 tensör) ve iki okuma akışı birbirinden bağımsız; bellek gecikmesini saklayacak yeterli iş var.
3. `silu`'nun `exp`'i PTX'te `ex2.approx.ftz.f32` (ders 3), yani iki `fma` ve bir komut. Hesap yükü bellek gölgesinde kayboluyor.

**Sonuç:** ürün çift sürümü kullanıyor, ama gerekçe hız değil: yarı thread ile aynı işi yapıyor ve ileride `[f16; 2]`'yi başka bir kernel ile birleştirmek kolay. Hız farkı ölçüm gürültüsünün içinde.

Bu dersin asıl kazanımı şu: **ders 1'in hilesi evrensel değil.** Hangi optimizasyonun işe yaradığı, kernel'in eleman başına ne kadar "sabit yük" taşıdığına bağlı. Ölçmeden bilemezsin.

## 5. Kartın sınırı ve ne kadar yaklaştık

Traffic: iki okuma + bir yazma = 3 × 486 × 3072 × 2 bayt = **9,0 MB**.

| Sürüm | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|
| `swiglu_f16_x1` | 53,8 µs | 166 GB/s | %87 |
| `swiglu_f16` (çift) | 53,3 µs | 168 GB/s | %87 |

%87, bu kartta üç tensörlü bir bellek kernel'i için tavan (ders 1'deki `bias_gelu` de %90, ders 10'daki RoPE %87). Kalan %13: yazma trafiği (okuma/yazma karışıkken DRAM verimi düşer), L2'ye sığmayan 9 MB ve `exp`.

**Daha hızlısı nasıl olur?** Kernel'i hızlandırarak değil, **kaldırarak**. SwiGLU'yu `up` projeksiyonunun GEMM'ine epilog olarak katarsak `up` ve `mid` tensörleri hiç yazılmaz: 9 MB yerine 6 MB trafik, ve bir kernel çağrısı eksilir. Ders 4'teki "kernel fusion" fikrinin somut hâli (alıştırma 1).

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `swiglu_f16`: `ld.global.b16` × 2, `neg.f32`, `ex2.approx.ftz.f32` (libm'in `expf`'i), `add.f32`, `div`/`rcp`, `mul.f32`, `cvt.rn.f16.f32`, `st.global.b16`.
- `swiglu_f16x2`: `ld.global.b16` × 4 (ikişer ikişer), iki `ex2`, `st.global.b16` × 2.
- **`ld.local` olmamalı**: `[f16; 2]` paketi register'da kalmalı.

Nsight Compute:

```powershell
ncu --set full -k regex:swiglu -o sw target\release\gpu-bench.exe swiglu
```

- **Memory Throughput**: iki sürümde de ~%88. Farkın yokluğunu burada görürsün.
- **Memory Workload Analysis → DRAM**: okuma ~%100, yazma daha düşük; karışık trafikte verim düşer.
- **Compute (SM) Throughput**: `ex2` yüzünden %10-20 civarı; yani hesap bellek gölgesinde.
- **Launch Statistics → Threads**: çift sürüm yarı thread ile aynı süreyi verir; bu, "thread sayısı değil, taşınan bayt önemli" demektir.

## 7. Alıştırmalar

1. **Fusion.** `up` GEMM'inin çıktısını yazarken `silu(gate)` ile çarp (epilog). `mid` tensörü hiç yazılmazsa kaç µs kazanılır? `gate`'i de aynı GEMM'de üretmek mümkün mü, neyi engeller?
2. **f32 çıktı.** `mid`'i f32 yaz (down projeksiyonu f16 giriş istiyor, ama ölçüm için). Trafik nasıl değişir, süre nasıl?
3. **`[f16; 4]`.** Thread başına 4 eleman (8 bayt). %87'nin üstüne çıkıyor mu? Neden?
4. **GELU ile karşılaştır.** Ders 1'deki `bias_act_f16` (GELU, bias'lı) ile aynı boyutta bu kernel'i ölç. Aradaki farkın ne kadarı `%`'den, ne kadarı `erf`'ten?
5. **Sadece `silu`.** Kapı çarpımını kaldır, sadece `silu(gate)` yaz. Trafik 6 MB'a inince süre tam olarak 2/3'e iniyor mu? İnmiyorsa neden?
