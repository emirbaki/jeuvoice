# Ders 12: Maskeli yinelemeli çözme (OmniVoice)

OmniVoice bir dil modeli değil, bir **doldurma** modelidir: hedef çerçevelerin hepsi bir maskeyle başlar ve 32 adımda, her adımda en güvenli konumlar açılarak doldurulur. Bu ders o döngünün tamamını anlatır: çizelge, guidance, seçim ve ölçülen maliyet.

Kod: `crates/jeuvoice-omnivoice/src/decode.rs`, `crates/jeuvoice-gpu/src/ops/guide.rs`, `crates/jeuvoice-gpu/src/kernels/mod.rs` (`guide_peak_f32`).

## 1. Ne yapıyor?

Sekiz kod defteri (codebook) ve `T` çerçeve vardır; toplam `8·T` token. Hepsi `mask_id = 1024` ile başlar. Bir adımda:

1. Dizi kurulur: `[stil][metin][prompt çerçeveleri][maskeli hedef]` ve aynı anda **koşulsuz** satır (yalnız hedef).
2. İki satır tek geçişte ileriye gider (bidirectional attention).
3. Her `(konum, defter)` çifti için guided log-softmax:

```
y = (1 + g)·c − g·u                 c: koşullu, u: koşulsuz logit, g = 2.5
log_p = y − logsumexp(y)            maske kimliği normalleştiricide KALIR
güven = max_{v ≠ mask} log_p        token = argmax_{v ≠ mask} y
```

4. Skor: `güven − 5·defter` (küçük defterler önce açılır), ürün modunda `+ Gumbel`.
5. Çizelgenin söylediği `k` en yüksek skorlu çift açılır; yazılan token bir daha değişmez.

Çizelge: `t_i = 0.1·i/32 / (1 − 0.9·i/32)`, `k_i = ceil(8T·(t_{i+1} − t_i))`, son adım kalanı alır. s1'de (T = 214, 1712 çift) ilk adım **6** çift açar.

Neden ilk adımlar az? Erken yazılan yanlış bir token sonraki tüm adımların bağlamını bozar; geç adımlar daha çok bilgi görür.

## 2. Neden gerekli, ne sıklıkla çalışır?

Otoregresif bir model her token için bir geçiş ister: 8·214 = 1712 geçiş. Maskeli çözme **32 geçiş** yapar, çünkü attention bidirectional ve bir geçiş bütün konumları aynı anda görür. Bedeli: her geçiş tüm dizi (486 konum) üzerinde, iki satırla.

Bir render: 32 × 77,4 ms ≈ 2,5 s ileri geçiş (iki geçiş: 486 + 214) + ~0,1 s çizelge/top-k/indirme. Motorun aynı aşaması 4,6 s.

## 3. Naive hâl ve ölçümü

İlk sürüm her adımda iki logit düzlemini (2 × 214 × 8200 × 4 B = **14 MB**) CPU'ya indiriyor, guided log-softmax'ı CPU'da yapıyordu:

- 3.4 milyon `exp` / adım, 32 adımda ~56 milyon.
- Ölçüm: sıcak s1 decode **4.25 s**; bunun **~0.65 s**'si CPU guidance ve indirme.

İkinci naive hata: maske kimliğini normalleştiriciden çıkarmak. Bu, her satırın güvenini farklı kaydırır ve ilk adımın 6'lı seçimini bozar; token payı %0.6'ya düşer. Doğrusu: maske toplamda kalır, argmax'ta çıkar.

## 4. Hızlandırma adımları

### 4.1 Guidance'ı GPU'ya taşı (4.25 → 3.74 s)

`guide_peak_f32`: çift başına bir blok (256 thread), iki geçişli (max/argmax, sonra `exp` toplamı), shared bellekte ağaç redüksiyonu ve `block_sum`. Çıktı yalnız `conf` + `pick`: adım başına **13.7 KiB** iner, 14 MiB değil.

Ölçüm: s1 **3.74 s**, s2 3.5 s, s3 3.2 s. Kazanç ~0.5 s. Tuzak: koşulsuz düzlem 486. satırda başlar, 256 bayta hizalı değildir → `slice_unaligned`; ve `vocab = 1025`, yani blok 1024 değerle sınırlı değildir.

### 4.2 Rezidüeli doğrudan f32'ye ekle (114.3 → 107.0 ms)

`o_proj` ve `down_proj` f16 bir tampona yazılıp sonra f32'ye ekleniyordu. `beta = 1` ile doğrudan `x`'e birikince 56 launch ve bir f16 yuvarlaması gider. Doğruluk da artar: `hidden_27` hatası 18.70 → 16.01.

### 4.3 K/V tekrarını kaldır (attention 1217.7 → 708.7 µs)

İki komşu sorgu kafası tek anahtar kafasını paylaşır ve bellekte bitişiktir. Bir anahtar kafası için `M = 2T` tek GEMM, K/V kopyası yok. **Ama dolgu maskesinin grid'i tüm skor düzlemini kapsamalıdır** (`b·hq`, `b·h_kv` değil): ilk denemede yarısı maskelenmedi ve yalnız koşulsuz satır bozuldu (%23.8). Ders: dolgu kuralı "görünmez" değil, hesaplanan değeri değiştirir.

Ölçüm yöntemi notu: buradaki 708,7 ve 1021,3 µs, geçişin içinde ölçüldü. İzole bench aynı kernel için 881,7 ve 1297,4 µs verir. Fark iki yöntemden gelir. İzole bench her çağrıyı soğuk cache ile başlatır. Geçiş içi ölçüm sıcak cache görür. Saat hızı da süreçler arasında değişir. İki sayıyı karşılaştırırken aynı yöntemi kullan.

### 4.4 Gereksiz işi at (4.32 → 4.23 s)

Açılmış konum için guided peak yok (%16.27 daha az çağrı), `k = 0` adımda geçiş yok, koşullu ve koşulsuz logit satırları bitişik olduğu için tek indirme.

### 4.5 Ürün modunda top-k'ı GPU'ya taşı

`class_temperature > 0` için CPU, 1712 çiftin 1025 logit'ini indiriyordu: adım başına 13,4 MiB. Yeni `product_topk_f32` kernel'i her çift için guided logit dizisini kurar. Maskeyi dışlar ve ilk 103 token'ı skor ile indeks sırasına göre yazar. Host yalnızca 103 kimlik ve 103 değer okur. İndirme adım başına yaklaşık 1,4 MiB'ye düşer. Gumbel gürültüsü host'ta kalır. Engine'in Philox akışı tekrarlanamaz. Bizim seed'imiz yine aynı sonucu verir.

İzole bench: 1712 çift için **2140 µs**, 7,2 GB/s (%4). Kernel hesap yoğun değil, 103 seçim adımında blok redüksiyonu yapıyor. Sınır burada bellek değil, seçim döngüsünün seri doğası. PTX'te `bar.sync` sayısı yüksektir. Olası hızlandırma: 103 yerine eşik tabanlı seçim veya blok başına birden fazla çift.

## 5. Kartın sınırı

RTX 3050 Ti Laptop, 192 GB/s, 15.2 TFLOP/s fp16.

| Parça | Süre (geçiş başına) |
|---|---|
| yedi GEMM, 972 satır | ~81 ms |
| attention, dolgulu | 28 × 1021 µs = 28.6 ms |
| SwiGLU | 28 × 110 µs = 3.1 ms |
| audio head (972 × 8200 × 1024) | ~1.0-1.35 ms |
| **iki geçiş, ölçülen** | **77,4 ms** |

Bench'ler her kerneli soğuk cache ile çağırır; toplamları ölçülen geçişten büyüktür. Bu tablo E1-A ve P5 öncesindendir. İki ayrı geçiş ve d128 flash ile adım 66-67 ms'ye indi. CUDA graph replay'i de launch yükünü azalttı. Sıcak product decode bugün 2,04 s; hedef 2,3 s.

## 6. PTX ve Nsight'ta neye bakmalı

- `guide_peak_f32`: iki kez `ld.global.f32`, `fma.rn.f32`, `ex2.approx.f32` (libm'in `expf`'i), `bar.sync` × ~10 (ağaç redüksiyonu). `vocab = 1025` olduğu için her thread 4-5 değer işler; blok başına 8 KB okuma.
- **Compute (SM) Throughput** guidance'ta düşük, **Memory** düşük: kernel küçük (1712 blok, 8 KB okuma). Asıl maliyet adım sayısıdır, tek çağrı değil.
- Çözme döngüsünde **en uzun kernel** audio head GEMM'i ve attention'dır; `ncu --set full -k regex:gemm` yerine Nsight Systems ile adımın zaman çizgisine bak: iki GEMM bloğu ve arada mask/softmax.
- Ürün modunun ilk sürümünde top-k ve `argmax` CPU'da 1712 değerle uğraşırdı: profilde görünmez, ama adım başına 13,4 MiB indirme vardı. `product_topk_f32` bu işi GPU'ya taşıdı; Nsight Systems'da indirme artık yaklaşık 1,4 MiB.

## 7. Alıştırmalar

1. **KV cache.** Prompt öneki 272 konum sabit. Yalnız hedef satırları için K/V'yi her adımda yeniden hesaplamak yerine önekin K/V'sini bir kez hesapla. Kazanç ne? (Dikkat: attention bidirectional olduğu için hedef token'lar değişince önekin **hidden** durumu da değişir; geleneksel KV cache bu modelde geçersizdir, yalnız önekin kodlanmış K/V'sini yeniden kullanmak tartışılabilir.)
2. **Çizelge.** `t_shift = 0.1` yerine kosinüs çizelgesi kullan; aynı 32 adımda token payı ve ses kalitesi (Phase 3 dinlemesi) nasıl değişiyor?
3. **Guidance.** `g = 0` ile `g = 2.5` arasında ilk adımın 6'lı seçimi nasıl değişiyor? Hangi konumlar kayboluyor?
4. **Top-k oranı.** Ürün modunda %10 yerine %50: token çeşitliliği ve çerçeve sayısı?
5. **Maskeyi normalleştiriciden çıkar.** Yanlış sürümü geri getir (maske `y[mask] = -inf` baştan) ve token payını ölç; farkı bir test iddiasına çevir.
