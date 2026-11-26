# Ders 11: SwiGLU

Qwen3'ün MLP'si Whisper'ınkinden farklı: iki paralel projeksiyon ve aralarında bir **kapı** (gate). Bu dersin konusu kapının aktivasyonu, SwiGLU. Ders 1'deki "iki eleman, bir thread" hilesini burada tekrar deniyoruz ve **hiçbir şey kazandırmıyor**. Sebebini ölçerek buluyoruz.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`silu`, `swiglu_f16`, `swiglu_f16x2`, `swiglu_strided_f16`), `crates/jeuvoice-gpu/src/ops/swiglu.rs`.

Ölçümler: RTX 3050 Ti Laptop, bellek sınırı 192 GB/s, `[486, 3072]` f16. `target\release\gpu-bench.exe swiglu`.

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

Bu ölçüm tek satır (`[486, 3072]`) içindir. Üretimde bir geçiş iki satırdır (`[972, 3072]`) ve süre satır sayısıyla doğrusal artar, çünkü kernel her öğeyi bir kez işler. Karşılaştırma yaparken aynı satır sayısını kullan.

Dikkat: bu, ders 1'deki `bias_gelu`'nun ilk sürümünden (%56) çok daha iyi. Fark nerede? `bias_gelu` her elemanda `i % d` hesaplıyordu (tam sayı bölme, GPU'da pahalı) ve `erf` polinomu çalıştırıyordu. Burada `%` yok (bias yok) ve `silu` sadece bir bölme ile bir `exp`. Paket 1 o bölmeyi `rcp.approx.ftz.f32` ile değiştirdi; PTX'te `div.rn` kalmadı, ama kernel bellek sınırlı olduğu için süre değişmedi (486 satır 53,2 µs).

## 4. Hızlandırma denemesi: çift okuma — kazanç yok

Ders 1'de `[f16; 2]` okuma %56'dan %90'a çıkarmıştı. Aynı hileyi burada denedik: thread başına iki komşu eleman, tek `[f16; 2]` yüklemesi.

Ölçüm (üç koşu, izole): **53,3 µs, 168 GB/s, %87.** Fark yok. İki sürüm de aynı sınırda.

Neden? Üç sebep:

1. **Eleman başına 6 bayt hareket ediyor** (iki okuma, bir yazma). x1 sürümünde her thread zaten 6 baytlık iş yapıyor; birleştirmenin paylaştıracağı "sabit yük" (indeks, `%`, sınır kontrolü) neredeyse yok.
2. **Warp zaten 64 baytlık okuma yapıyor** (32 thread × 2 bayt × 2 tensör) ve iki okuma akışı birbirinden bağımsız; bellek gecikmesini saklayacak yeterli iş var.
3. `silu`'nun `exp`'i PTX'te `ex2.approx.ftz.f32` (ders 3), yani iki `fma` ve bir komut. Hesap yükü bellek gölgesinde kayboluyor.

**Sonuç:** ürün çift sürümü kullanıyor, ama gerekçe hız değil: yarı thread ile aynı işi yapıyor ve ileride `[f16; 2]`'yi başka bir kernel ile birleştirmek kolay. Hız farkı ölçüm gürültüsünün içinde.

Bu dersin asıl kazanımı şu: **ders 1'in hilesi evrensel değil.** Hangi optimizasyonun işe yaradığı, kernel'in eleman başına ne kadar "sabit yük" taşıdığına bağlı. Ölçmeden bilemezsin.

## 4b. Üretimdeki sürüm: iki yarı tek matristen

P3 (performans incelemesi, D6) `gate` ve `up` projeksiyonlarını tek bir `N = 6144` GEMM'de birleştirdi. Ağırlık yüklemede `[gate; up]` olarak yazılıyor ve çıktı tek bir `[rows, 6144]` tensörü oluyor: `gate` ilk 3072 sütunda, `up` sonraki 3072 sütunda. SwiGLU artık iki ayrı tensör değil, bu geniş satırın iki yarısını okuyor.

Kernel: `swiglu_strided_f16`. Thread başına bir çıktı elemanı:

```rust
let src = (i / row_width) * row_stride + i % row_width;
let g = *gate.add(src) as f32;
let u = *up.add(src) as f32;
*out.add(i) = (silu(g) * u) as f16;
```

`gate` ve `up` işaretçileri zaten kendi sütunlarının başına kaydırılmış; kernel'e kalan tek şey satır adımı.

Ölçüm (izole, kararlı koşu):

| Sürüm | Satır | Süre | Hız | Sınırın yüzdesi |
|---|---|---|---|---|
| `swiglu_f16` (bitişik) | 486 | 53,2 µs | 168,2 GB/s | %88 |
| `swiglu_strided_f16` | 486 (6144 sütunlu satırdan) | **54,7 µs** | 163,9 GB/s | %85 |
| `swiglu_strided_f16` | 214 | 29,1 µs | 135,5 GB/s | %71 |

İlk ölçümde strided sürüm **%11 daha yavaştı** (59,2 µs). Sebep iki taneydi:

1. `i / row_width` ve `i % row_width`: satır genişliği 3072, ikinin kuvveti değil, ve iki tam sayı bölmesi her elemanda çalışıyordu. Ders 1'deki `bias_gelu`'nun %56'sının sebebi de buydu.
2. Satır 12 KB: bir warp bir blokta 64 baytlık bitişik bir satır okuyor (birleşme bozulmuyor), ama diğer blok 6 KB ötede. DRAM iki ayrı akış görüyor, yazma da üçüncü akış. Bitişik sürümde üç tensör de aynı hizada ilerliyor.

Sihirli bölme birinci maddeyi kaldırdı. Host, 3072 için çarpan ve kaydırma değerini bir kez hesaplıyor ve kernel `mul.wide.u32` + `shr.u64` ile satırı buluyor. PTX'te artık `rem.u32` yok. Strided sürüm 54,7 µs'ye indi; bitişik sürümün %3'ü kadar bir fark kaldı ve o fark ikinci maddeden geliyor. Aynı yöntem `to_heads_f16`, `from_heads_f16` ve `rope_half_f16_rows` içindeki tam sayı bölmelerine de uygulandı.

Bu maliyete rağmen birleştirme kârlı, çünkü kazancı GEMM tarafında. `M = 214`: tek `N = 6144` GEMM **242,9 µs**, iki `N = 3072` GEMM **324,4 µs** (−%25). `M = 486`: 477,9 µs'ye karşı 468,4 µs (+9,5 µs, yani hafif zarar). İki geçişin toplamında katman başına net kazanç ~72 µs; 28 katman × 32 adım = adım başına yaklaşık 4 ms.

**Neden çıktıyı `gate` yarısına yazmıyoruz?** O zaman `down_proj` GEMM'i `ld = 6144` ile okumak zorunda kalırdı; `blas.rs` sarmalayıcısında `ld` parametresi yok. Ayrı `[rows, 3072]` çıktı 6 MB fazla yer kaplıyor, karşılığında sarmalayıcı sade kalıyor (alıştırma 6).

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
6. **`ld` ile yerinde yazma.** `swiglu_strided_f16` çıktısını `gate` yarısının üstüne yaz ve `down_proj` GEMM'ine `ld = 6144` ver. `blas.rs`'e bir `ld` parametresi ekle. 6 MB'lık tensör ve bir kopya ortadan kalkarken süre ne oluyor? cuBLAS geniş `ld` ile aynı hızda mı çalışıyor?
