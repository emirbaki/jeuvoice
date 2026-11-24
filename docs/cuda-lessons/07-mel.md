# Ders 7: Log-mel spektrogram (matris çarpımı olarak DFT)

Bu derste ses örneklerini Whisper'ın gördüğü resme çeviriyoruz: `[128, 3000]` boyutlu log-mel spektrogram. İki büyük fikir var:

1. **DFT bir matris çarpımıdır.** FFT yazmadan, sabit bir "DFT matrisi" ile cuBLAS'a yaptırabiliriz.
2. **Önce doğru olanı kopyala, sonra hızlandır.** whisper.cpp'nin adımlarını önce CPU'da birebir yazdık (`mel_cpu.rs`). Sonra GPU sürümünü o koda karşı test ettik.

Kod: `crates/jeuvoice-whisper/src/mel_cpu.rs` (CPU, test kehaneti), `crates/jeuvoice-whisper/src/cuda/mel.rs` (GPU akışı), `crates/jeuvoice-gpu/src/ops/mel.rs` ve kernel'ler `mel_frames_f32`, `power_f32`, `log10_max_f32`, `max_f32`, `mel_window_f16`.

## 1. Ne yapar?

16 kHz ses örneklerinden, 10 ms'de bir, 128 frekans bandının log enerjisini çıkarır. whisper.cpp'nin (`log_mel_spectrogram`, `whisper.cpp:3170`) adımları:

1. **Dolgu:** başa 200 örnek yansıma (`x[200], x[199], …, x[1]`), sona 30 saniye + 200 örnek sıfır.
2. **Çerçeveler:** 160 örnekte bir (10 ms), 400 örneklik (25 ms) pencere. `n_len = (n + 480.000) / 160` çerçeve.
3. **Hann penceresi:** `0,5 · (1 − cos(2πj / 400))`. Çerçevenin kenarlarını yumuşatır, spektrumda "sızıntıyı" azaltır.
4. **Güç spektrumu:** 0–200 arası frekans kutuları (bin), `|X[k]|² = re² + im²`.
5. **Mel filtreleri:** 128 üçgen filtre (model dosyasında hazır, `[128, 201]`). Bant `b` = `Σ_k güç[k] · filtre[b][k]`. İnsan kulağı gibi alçak frekanslarda dar, yüksek frekanslarda geniş bantlar.
6. **Log ve normalleştirme:** `log10(max(v, 1e-10))`. **Bütün dosya** üzerindeki en büyük değer `m`. `m − 8`'den küçük her değer `m − 8` olur. Son olarak `(v + 4) / 4`.

**Önemli bir ayrıntı:** whisper.cpp en büyük değeri her 30 saniyelik pencerede değil, **bütün dosyada** bulur. Plan ilk başta "her pencereyi ayrı normalleştir" diyordu. Bu 30 saniyeden uzun dosyalarda whisper.cpp'den farklı sonuç verirdi. Kaynak kodu okuyunca düzelttik. Bu yüzden GPU tarafında iki adım var: `compute(samples)` bütün dosyayı hesaplar, `window_into(start, …)` bir pencereyi encoder'ın girdisine kopyalar.

## 2. Neden gerekli, ne sıklıkla çalışır?

Whisper ham ses dalgasını değil, bu spektrogramı "okur". Mel ölçeği ve log, sesi insan algısına yakın bir biçime çevirir. Dosya başına bir kez çalışır. Encoder'a göre çok küçük bir iştir, ama CPU'da yazarsak yavaş kalır.

## 3. İlk sürüm: CPU'da birebir kopya

`mel_cpu::log_mel` her adımı f64 ile ve en basit hâliyle yapar. Her çerçeve için 201 kutunun her biri 400 çarpım ister (doğrudan DFT, FFT değil):

| Sürüm | 30 s pencere |
|---|---|
| CPU, tek thread, f64, doğrudan DFT | 311,8 ms |

Bu sürümün işi hızlı olmak değil, **doğru** olmaktır. PyTorch (Hugging Face) referansıyla karşılaştırdık:

| Klip | Fark (ortalama) | Fark (en büyük) |
|---|---|---|
| `my_voice` (27,0 s) | 0,00000 | 0,0000 |
| `ses_001_0007` (8,4 s) | 0,00000 | 0,0000 |

Neden tam eşit? 30 saniyeden kısa seslerde whisper.cpp'nin "bütün dosya" en büyük değeri ile Hugging Face'in "30 saniyelik pencere" en büyük değeri aynıdır: dolgu sıfırları (log −10) hiçbir zaman en büyük değer olmaz.

## 4. GPU sürümü: DFT bir GEMM'dir

DFT'nin tanımı:

```
X[k] = Σ_j x[j] · (cos(2πkj/N) − i·sin(2πkj/N))
```

`k` ve `j` için `cos` ve `−sin` değerleri sabittir. Bunları bir kez `[400, 402]` boyutlu bir matrise yazarız (ilk 201 sütun `cos`, sonraki 201 sütun `−sin`). O zaman bütün çerçevelerin DFT'si tek bir matris çarpımıdır:

```
spec [n_çerçeve, 402] = çerçeveler [n_çerçeve, 400] · DFT [400, 402]
```

Mel filtreleri de aynı şekilde ikinci bir GEMM'dir:

```
mel [n_çerçeve, 128] = güç [n_çerçeve, 201] · filtreᵀ [201, 128]
```

GPU akışı (`cuda/mel.rs`), 3000 çerçevelik parçalar hâlinde:

| Adım | Ne | Nasıl |
|---|---|---|
| 1 | Çerçeve × Hann | `mel_frames_f32` (bir thread, bir değer) |
| 2 | DFT | `Blas::gemm_f32` |
| 3 | Güç | `power_f32` |
| 4 | Mel | `Blas::gemm_f32` (filtre `Op::T` ile) |
| 5 | log10 + en büyük | `log10_max_f32` (block başına en büyük), sonra `max_f32` (tek block) |
| 6 | Bir pencere | `mel_window_f16`: kırp, normalleştir, `[128, 3000]` biçimine çevir |

| Sürüm | 30 s pencere |
|---|---|
| CPU (f64, doğrudan DFT) | 311,8 ms |
| GPU (bütün adımlar, yükleme ve indirme dahil) | **3,24 ms** |

Yaklaşık 96 kat. Bu sürenin büyük kısmı GEMM'ler değil: arenayı ayırmak, sesi GPU'ya yüklemek ve sonucu indirmek. Hesap kısmı bir milisaniyenin altındadır.

### Neden parçalar hâlinde?

Bir saatlik ses yaklaşık 363.000 çerçevedir. Bütün çerçeveleri, spektrumu ve gücü aynı anda tutmak yaklaşık 1,7 GB eder. 4 GB'lık kartta model de var. Bu yüzden 1–4. adımlar 3000 çerçevelik parçalarla çalışır. Sadece sonuç (`[n_len, 128]`, bir saat için yaklaşık 194 MB) bütün olarak tutulur. Bir parçanın satırlarını yazmak için `DevSlice::range` ekledik.

### İki aşamalı en büyük değer

Bütün dosyada tek bir en büyük değer lazım. Bir block'un thread'leri `block_max` ile anlaşabilir (ders 3'teki gibi). Ama block'lar birbirini beklemez. Çözüm iki kernel:

1. `log10_max_f32`: her block 2048 değeri işler, en büyüğünü `partial[block]`'a yazar.
2. `max_f32`: tek bir block bütün `partial` değerlerini indirger.

İki kernel aynı stream'de sırayla çalışır, bu yüzden ikinci kernel birincinin bütün sonuçlarını görür.

## 5. f32 mi, f16 mı? Ölçerek karar ver

Bu kernel'lerde f16 yerine f32 kullandık. İlk gerekçemiz şuydu: normalleştirme `m − 8` ile log10 güçte 8 birimlik (80 dB) bir aralık tutar. f16'nın hassasiyeti yaklaşık 11 bit, yani −66 dB civarında bir gürültü tabanı. Sessiz bantlara gürültü karışabilir.

Sonra bunu **ölçtük**: örnekleri f16'ya yuvarlayıp mel'i yeniden hesapladık (`mel_timing_and_f16_error` testi):

| | Değer |
|---|---|
| Normalleştirilmiş mel'de en büyük fark | 0,0445 |
| 0,02'den büyük farklar | 768.000 değerden 20 |

Yani tahminimiz abartılıydı. Gerçek konuşmada çok az değer etkileniyor. Yine de f32'de kaldık: maliyeti küçük (3,24 ms'nin içinde) ve encoder'ı f16 hatalarından bağımsız test edebiliyoruz. Ders 1'deki gibi: **tahmin değil, ölçüm.**

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `log10_max_f32`: `libm::log10f` bir dizi `fma` ve `lg2.approx.f32` olur. Sonra `block_max` içinde `shfl.sync.bfly` ve `bar.sync`.
- `mel_window_f16`: `cvt.rn.f16.f32` ile f16'ya çevirme. Okuma `[çerçeve][bant]`, yazma `[bant][zaman]` sırasındadır. Yazma birleşik (coalesced), okuma değil. Pencere küçük olduğu için önemli değil.

Nsight Systems (zaman çizelgesi):

```powershell
nsys profile -o mel scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --release --test cuda_mel -- --ignored mel_timing
```

Zaman çizelgesinde iki `gemm` (cuBLAS SGEMM kernel'i), küçük kernel'lerimiz ve `cuMemcpyHtoD` / `cuMemAlloc` çağrılarını gör. Sürenin büyük kısmının hesap değil, kopyalama ve ayırma olduğunu orada görürsün.

## 7. Alıştırmalar

1. **Ayırmayı bir kez yap.** `compute` her çağrıda arenayı yeniden ayırır. En uzun dosya için bir kez ayır ve tekrar kullan. 3,24 ms ne kadar düşer?
2. **Simetriyi kullan.** Gerçek bir sinyalin DFT'sinde `cos` sütunları ve `sin` sütunları ayrı ayrı hesaplanır. Bir `[400, 201]` karmaşık matris yerine iki gerçek GEMM ile aynı işi yap ve hızını ölç.
3. **f16 DFT.** DFT GEMM'ini `gemm_f16` ile yap (çerçeveleri ve DFT matrisini f16'ya çevirerek). Hızı ve konuşma testindeki farkı ölç. Bu kez sadece örnekler değil, toplam da f16'da: fark büyür mü?
4. **FFT.** 400 = 16 × 25. Bu boyutta bir FFT kernel'i yazmak neden zordur? DFT GEMM'i hangi boyuttan sonra FFT'den yavaş kalır? (İpucu: DFT `O(N²)`, FFT `O(N log N)`. Ama GEMM tensor core ya da cuBLAS hızında çalışır.)
