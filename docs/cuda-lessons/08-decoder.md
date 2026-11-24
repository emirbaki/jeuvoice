# Ders 8: Whisper decoder'ı

Bu ders decoder'ın GPU kernel'lerini anlatır. İki bölümü var:

- **Bölüm A: token gömme (embedding).** Decoder'a giren her token'ı 1280 sayılık bir vektöre çeviren kernel.
- **Bölüm B: decoder attention ve KV cache.** Self-attention, cross-attention ve DTW için hizalama başlarının softmax'ı tek bir kernel ailesinde.

Kod: `crates/jeuvoice-gpu/src/ops/embed.rs`, `crates/jeuvoice-gpu/src/ops/attention_decode.rs` ve kernel'ler `embed_*`, `attn_decode_f16`, `attn_decode_split_f16`, `attn_decode_merge_f16` (`crates/jeuvoice-gpu/src/kernels/mod.rs`). Testler: `crates/jeuvoice-gpu/tests/embed.rs`, `crates/jeuvoice-gpu/tests/attention_decode.rs`.

---

# Bölüm A: Token gömme (embedding)

## 1. Ne yapar?

Decoder metni token token yazar. Her adımda elindeki token'ların numaralarını (id) alır ve onları vektöre çevirir:

```
x[i] = W_tok[id[i]] + P[pos0 + i]
```

- `W_tok`: token gömme tablosu, `[51866, 1280]`, f16. Her satır bir token'ın vektörüdür. Boyutu 51866 · 1280 · 2 bayt = **133 MB**.
- `P`: konum gömme tablosu, `[448, 1280]`, f32. Satır `p`, "bu token metnin `p`. konumunda" bilgisini taşır. Model dosyasında f32 olarak durur (2,3 MB).
- `pos0`: bu çağrıdaki ilk token'ın konumu. KV cache'te önceden kaç token varsa odur.

Sonuç `x`, f16 olarak decoder'ın ilk katmanına girer.

Bu bir **toplama (gather)** işlemidir: her satır, tablonun başka bir yerinden okunur. Hangi satırın okunacağını id belirler. Komşu satırlar bellekte birbirinden çok uzakta olabilir. Ama bir satırın içindeki 1280 değer yan yanadır, ve bir warp'ın 32 thread'i bir satırın komşu değerlerini okur. Bu yüzden okumalar yine birleşir (coalesced).

## 2. Neden gerekli, ne sıklıkla çalışır?

Transformer sayılarla çalışır, kelimelerle değil. Gömme, token numarasını modelin anladığı vektöre çeviren ilk adımdır. Aynı `W_tok` tablosu decoder'ın sonunda da kullanılır: logit'ler `x · W_tokᵀ` ile hesaplanır (Task 9).

Bir 30 saniyelik pencerede gömme şu kadar çalışır:

| Çağrı | Satır sayısı | Pencere başına |
|---|---|---|
| Prompt (`[SOT, tr, transcribe]` ve önceki pencerenin token'ları) | 3 ile 227 arası | her sıcaklık denemesinde 1 kez |
| Decoder adımı | 1 | en çok 220 kez |
| DTW geçişi (Ders 9) | metin token'ları + 4 | 1 kez |

Yani asıl sık çalışan durum **tek satırdır**. Bu, kernel'in hızını değil, başlatma (launch) maliyetini önemli yapar.

## 3. İlk sürüm: bir eleman, bir thread

`embed_f16` en basit hâldir. Her thread çıktının bir elemanını yazar:

```rust
let (row, c) = (i / d, i % d);
let id = *ids.add(row) as usize;
*out.add(i) = (*tok.add(id * d + c) as f32 + *pos.add(i)) as f16;
```

Id'ler GPU'ya bir `I32` tensör olarak yüklenir. Bunu `ops::embed` yapar: önce **her id'yi CPU'da denetler** (`id < 51866`), sonra yükler. Böylece bozuk bir id hiçbir zaman GPU'da tablonun dışını okutmaz. Test `embed_id_past_the_vocab_panics_before_the_gpu_reads` bunu gösterir.

Ölçüm (RTX 3050 Ti Laptop, 3 çalıştırmanın ortancası, id yüklemesi dahil):

| Sürüm | 1 satır | 224 satır |
|---|---|---|
| `embed_x1` | 19,7 µs | 25,1 µs |

## 4. Hızlandırma adımları

### 4.1. İki eleman, bir thread (`[f16; 2]`)

Ders 1'deki gibi her thread iki komşu değer yazar (`embed_f16x2`). `tok` ve `out` için `[f16; 2]`, `pos` için `[f32; 2]` okunur.

| Sürüm | 1 satır | 224 satır |
|---|---|---|
| `embed_x1` | 19,7 µs | 25,1 µs |
| `embed_x2` | 17,6 µs | 26,2 µs |

**Kazanç yok** (fark ölçüm gürültüsü içinde). Neden? Taşınan veri çok az: 224 satır 224 · 1280 · 8 bayt = 2,3 MB. Zamanın çoğu kernel'de değil, başka yerde geçiyor. Nerede olduğunu bulmak için parçaları ayrı ölçtük.

### 4.2. Parçaları ayrı ölçmek: id yüklemesi

`ids_upload` satırı yalnızca id'lerin GPU'ya kopyasını ölçer (448 tane `i32`, 1,8 KB):

| İş | Süre |
|---|---|
| `ids_upload` (kopyalama ve bekleme) | 10,0 µs |

1,8 KB'lık bir kopya 10 µs sürüyor. Süre veri boyutundan gelmiyor: `cuMemcpyHtoDAsync` çağrısı, sürücünün işi, ve kopyanın bitmesini beklemek (`stream.synchronize`). CPU bu sürede GPU'yu bekler, GPU da bir sonraki kernel'i bekler.

Çıkarma ile kernel'in kendi süresini **tahmin** edebiliriz: 224 satırda yaklaşık 26 − 10 = 16 µs. Bu 2,3 MB / 16 µs ≈ 145 GB/s eder. Tek satırda kernel yalnızca birkaç µs'dir.

### 4.3. Tek token için id'yi kernel argümanı yapmak (`embed_one`)

Decoder adımında yalnızca bir token var. Onu yüklemeye gerek yok: id'yi kernel'e **başlatma argümanı** olarak verebiliriz. Argümanlar kernel ile birlikte gider, ayrı bir kopya ve bekleme olmaz.

```rust
pub unsafe fn embed_one_f16x2(id: u32, tok: *const [f16; 2], pos: *const [f32; 2], out: *mut [f16; 2], d2: usize)
```

| Sürüm | 1 satır |
|---|---|
| `embed_x2` (id yüklemesi ile) | 17,6 µs |
| `embed_one` (id argümanda) | **9,0 µs** |

Neredeyse yarı yarıya. Pencere başına 220 adımda bu yaklaşık 220 · 8,6 µs ≈ **1,9 ms** kazanç demek. Daha önemlisi, adım içinde CPU artık GPU'yu beklemiyor. Bu, ileride bir adımın bütün kernel'lerini art arda kuyruğa koymak için gerekli.

Ürün iki yolu da kullanır: prompt ve DTW geçişi `embed` (çok satır, bir yükleme), decoder adımı `embed_one`.

## 5. Kartın sınırı ve ne kadar yaklaştık

Kartın bellek sınırı 192 GB/s. Tek satırda taşınan veri 1280 · 8 bayt = 10 KB. Sınırda bu 0,05 µs sürerdi. Ölçülen 9 µs bunun 180 katı. Demek ki tek satırda **sınır bellek değil, kernel başlatma maliyetidir**. Bir kernel'i başlatmak ve olayı kaydetmek bu sürücüde birkaç µs sürüyor.

224 satırda kernel'in kendisi, çıkarma ile tahminen ~145 GB/s, yani sınırın yaklaşık %75'i. Bu tahmin iki ölçümün farkıdır, ayrı bir ölçüm değildir.

Sonuç: decoder adımını hızlandırmanın yolu kernel'leri hızlandırmaktan çok **kernel sayısını ve beklemeleri azaltmaktır** (birleştirme, CUDA Graphs). Bunu decoder bütünüyle çalıştığında ölçeceğiz (Task 10 ve 14).

## 6. PTX ve Nsight'ta neye bakmalı

`target\oxide-ptx\jeuvoice_gpu.ptx` içinde kernel başına okuma ve yazma komutları:

| Kernel | Okumalar | Yazmalar | Bölme |
|---|---|---|---|
| `embed_f16` | 1 × `ld.global.s32` (id), 1 × `ld.global.b16`, 1 × `ld.global.b32` | 1 × `st.global.b16` | `div.u32` / `div.u64` |
| `embed_f16x2` | 1 × `ld.global.s32`, 2 × `ld.global.b16`, 2 × `ld.global.b32` | 2 × `st.global.b16` | `div.u32` / `div.u64` |
| `embed_one_f16x2` | 2 × `ld.global.b16`, 2 × `ld.global.b32` | 2 × `st.global.b16` | yok |

Üç şeye dikkat:

1. **`[f32; 2]` tek 8 baytlık okuma olmuyor**, iki `ld.global.b32` oluyor. Sebep Ders 1 ve Ders 4'teki ile aynı: okuma genişliği tipin hizalamasını izler, `[f32; 2]`'nin hizalaması 4 bayttır. L1 önbelleği bu iki okumayı komşu thread'lerinkiyle birleştirir.
2. **`i / d2` için iki bölme komutu var.** Derleyici önce iki sayının da 32 bite sığıp sığmadığını denetler. Sığarsa hızlı `div.u32`, sığmazsa yavaş `div.u64` çalışır. `embed_one`'da satır yok, bölme de yok.
3. Yerel bellek (`ld.local`) yok.

Nsight Systems'ta (`nsys profile`) zaman çizgisine bakın. `embed` çağrısında önce bir `cuMemcpyHtoDAsync`, sonra bir bekleme, sonra kernel görünür. `embed_one`'da yalnızca kernel vardır. Nsight Compute'ta tek satırlık kernel'in kendi süresi (Duration) birkaç µs'den azdır. Fark başlatma maliyetidir.

## 7. Alıştırmalar

1. `pos` için `#[repr(C, align(8))] struct F32x2([f32; 2])` gibi bir tip yaz. PTX'te `ld.global.v2.f32` görünüyor mu? 224 satırda süre değişiyor mu?
2. Prompt en çok 227 token. Id'leri bir yükleme yerine kernel argümanı olarak vermenin bir yolunu düşün (örneğin sabit boyutlu bir dizi argüman). Kernel argümanlarının boyut sınırı nedir? Ölç.
3. Gömmeyi decoder'ın ilk LayerNorm'u ile tek kernel'de birleştir. Kaç bayt okuma ve yazma kalkar? Tek satırda süre ne kadar değişir, ve neden çok değişmez?
4. 1 satırdan 448 satıra kadar `embed_x2` süresini çiz. Çizgi nerede yatay, nerede yükselen bir doğru? Kırılma noktası neyi gösterir?

---

# Bölüm B: Decoder attention ve KV cache

## 1. Ne yapar?

Decoder'ın her katmanında iki attention var. İkisi de aynı formül, baş (head) başına:

```
out[r] = softmax(q[r] · Kᵀ / 8) · V
```

| Durum | Sorgu (q) satırı | Anahtar (k) sayısı | Maske |
|---|---|---|---|
| Self-attention, bir adım | 1 | `geçmiş + 1` (en çok 448) | yok |
| Self-attention, prompt veya DTW geçişi | `n` | `n` | nedensel (causal) |
| Cross-attention | 1 veya `n` | 1500 (encoder çıktısı) | yok |

- **Self-attention:** yeni token, o ana kadar yazılmış token'lara bakar. Nedensel maske ile satır `r`, `0 … nk − nq + r` anahtarlarını görür, sonrakileri görmez.
- **Cross-attention:** token, encoder'ın 1500 ses konumuna bakar. Hangi sesi "dinlediğini" böyle seçer.

**KV cache:** Bir adımda yalnızca yeni token'ın `q`, `k`, `v` değerleri hesaplanır. Eski token'ların `k` ve `v` satırları değişmez, onları her adımda yeniden hesaplamak boşa iştir. Bu yüzden her katman için `[448, 1280]` boyutlu bir K ve bir V tamponu (cache) tutarız. Yeni token kendi satırını bu tampona yazar, attention bütün satırları okur. Cross-attention'ın K ve V'si pencere başına bir kez, encoder çıktısından hesaplanır.

**Veri düzeni:** `q`, `k`, `v` ve `out`, GEMM çıktısının düzeninde durur: `[satır, 1280]`. Baş `h`, `64h … 64h + 63` sütunlarıdır. Encoder'daki gibi `[baş, satır, 64]` düzenine kopyalamaya (`split_heads`) gerek yok: kernel başın sütunlarını doğrudan okur.

**DTW için softmax:** Ders 9'daki kelime zamanları için 6 "hizalama başının" cross-attention softmax'ı gerekir. Kernel, `probs_mask` ile seçilen başların softmax'ını ayrıca bir f32 tensöre yazar. Bit `h` açıksa baş `h`, `slot0 + (h'den küçük açık bit sayısı)` numaralı yuvaya gider. Bu sayıyı tek bir `popc` (bit sayma) komutu verir.

## 2. Neden gerekli, ne sıklıkla çalışır?

Attention, decoder'ın bağlamı kullandığı tek yerdir: self-attention metnin geri kalanını, cross-attention sesi getirir. Bir decoder adımında 4 katman · 2 attention = **8 çağrı** var. Bir pencerede en çok 220 adım, yani en çok 1760 çağrı. Bu yüzden asıl önemli durum **tek sorgu satırıdır**.

Tek satırda iş azdır ama bütün K ve V okunur. Cross-attention için bu 2 · 1500 · 1280 · 2 bayt = **7,7 MB**. Hesap yok denecek kadar az: attention burada **bellek sınırlıdır**.

## 3. İlk sürüm: satır ve baş başına bir blok

`attn_decode_f16` en basit hâldir. Her (sorgu satırı, baş) çifti için 256 thread'lik bir blok:

1. 64 sorgu değeri ortak belleğe (shared memory) gider.
2. Thread `t`, `t, t + 256, …` anahtarlarının puanını hesaplar. Her anahtar satırını baştan sona kendisi okur. Puanlar ortak bellekte durur (en çok 1536 · 4 bayt = 6 KB).
3. `block_max` ve `block_sum` ile satırın en büyüğü ve üstel toplamı (Ders 3).
4. `P · V`: 256 thread 64 boyutu 4 gruba böler, her grup anahtarların dörtte birini toplar, sonra ortak bellekte birleştirilir.

Ölçüm (20 baş, 3 çalıştırmanın ortancası):

| Durum | Naive | Sınırın yüzdesi |
|---|---|---|
| 1 × 448 (self, bir adım) | 56,2 µs | %21 |
| 1 × 1500 (cross, bir adım) | 175,5 µs | %23 |
| 224 × 224, nedensel (prompt) | 1024 µs | — |
| 224 × 1500, softmax çıktılı (DTW) | 12,6 ms | — |

Bir adımın 4 katmanı (56 + 175) · 4 ≈ **0,94 ms** tutar. Çok fazla. İki sebebi var:

1. **Çok az blok.** Tek satırda grid 1 × 20 = 20 blok. Kartta 20 SM var, yani her SM'de tek blok, 8 warp. Bellekten okuma isteklerini gizleyecek kadar çok warp yok.
2. **Dağınık okuma.** Thread `t`, anahtar satırı `t`'yi 2 baytlık okumalarla okur. Bir warp'ın 32 thread'i aynı anda 32 farklı satıra (her biri 2560 bayt uzakta) dokunur: 32 ayrı önbellek satırı.

Çok satırlı durumlar daha da kötü: her satır için K ve V yeniden okunur.

## 4. Hızlandırma: flash decoding

Fikir: **anahtarları parçalara böl, her parçayı ayrı bir blok yapsın, sonunda birleştir.** Buna "flash decoding" denir. Ders 5'teki online softmax'ın paralel hâlidir.

### Bölüm 1: parça kernel'i (`attn_decode_split_f16`)

Grid `(parça sayısı, baş, ⌈nq / 4⌉)`. Her blok 64 anahtarlık bir parça ve 4 sorgu satırı alır:

1. K ve V'nin `64 × 64` parçasını ortak belleğe yükler. Ardışık thread'ler bir anahtar satırının ardışık değerlerini okur: **birleşik (coalesced) okuma**. Satır adımı 66 f16'dır, böylece puan döngüsünde 32 thread 32 farklı bankaya düşer (Ders 5'teki `FLASH_LD` ile aynı hile).
2. Thread `t` = sorgu satırı `t / 64`, anahtar `t % 64`. Puanı ortak bellekten hesaplar.
3. Bir satırın 64 thread'i iki warp'tır. Her warp kendi en büyüğünü `shfl` ile bulur, sonra ikisi ortak bellekte birleşir.
4. Her satır için bir **ara sonuç** yazar: 64 toplam `Σ exp(s − m) · v`, parçanın en büyüğü `m`, üstel toplam `l`.

Tek sorgu ve 1500 anahtarda grid 24 · 20 = **480 blok** olur. Her SM'ye 24 blok düşer.

### Bölüm 2: birleştirme kernel'i (`attn_decode_merge_f16`)

Satır ve baş başına bir blok, parçaları birleştirir:

```
M = max_c m_c
L = Σ_c l_c · exp(m_c − M)
out = Σ_c acc_c · exp(m_c − M) / L
```

Bu, online softmax'taki "eski toplamı yeni en büyüğe göre ölçekle" adımının aynısıdır. Nedensel maskede bir satırın hiç göremediği bir parça `m = −∞`, `l = 0` verir. `exp(−∞ − M) = 0` olduğu için o parça hiçbir şey katmaz. Parça 0 her satırda en az bir anahtar (anahtar 0) içerdiği için `M` hep sonludur. Test `split_causal_probs_and_partial_chunks` tam bu durumu dener.

DTW softmax'ı için parça kernel'i ham puanları yazar (maskeli anahtar için `−∞`). Birleştirme kernel'i onları `exp(s − M) / L`'ye çevirir.

Ara sonuçlar bir "scratch" tensörde durur: `baş · nq · parça · 66` f32. Tek satırda 20 · 24 · 66 · 4 = 127 KB.

### Ölçüm

| Durum | Naive | Split | Kazanç | Split, sınırın yüzdesi |
|---|---|---|---|---|
| 1 × 448 (self, bir adım) | 56,2 µs | 22,9 µs | 2,5× | %52 |
| 1 × 1500 (cross, bir adım) | 175,5 µs | **59,1 µs** | 3,0× | **%68** |
| 224 × 224, nedensel (prompt) | 1024 µs | 482 µs | 2,1× | — |
| 224 × 1500, softmax çıktılı (DTW) | 12,6 ms | 3,47 ms | 3,6× | — |

Bir adımın 4 katmanı artık (22,9 + 59,1) · 4 ≈ **0,33 ms**. Ürün `Split` kullanır.

## 5. Kartın sınırı ve ne kadar yaklaştık

Tek sorgulu cross-attention 7,7 MB okur. 192 GB/s sınırında bu **40 µs** sürer. `Split` 59 µs ile sınırın **%68**'inde. Geri kalanın bir kısmı iki kernel başlatmasıdır (Bölüm A'da gördüğümüz birkaç µs'lik maliyet, burada iki kez). Bir kısmı da parça yüklemesindeki 2 baytlık okumalardır (bkz. 6. bölüm).

Çok satırlı durumlarda tablo "sınırın yüzdesi" vermiyor, çünkü orada sınır bellek değil. 224 × 1500 DTW geçişi yaklaşık 224 · 1500 · 20 · 64 · 2 · 2 ≈ 0,86 GFLOP skaler f32 işlem yapar. 3,47 ms'de bu ~250 GFLOP/s eder. Kartın skaler f32 gücünün çok altında. Sebep: puan döngüsünde her çarpma-toplama iki ortak bellek okuması ister, yani ortak bellek bant genişliği sınırlıdır. Bunun doğru çözümü tensör çekirdekleridir (Ders 5'teki `FlashTc` gibi). Ama bu durumlar pencere başına yalnızca birkaç kez çalışır: DTW 4 pencerede ~14 ms, toplam 4,3 saniyelik bütçenin küçük bir parçası. Bu yüzden şimdilik bıraktık.

## 6. PTX ve Nsight'ta neye bakmalı

| Kernel | Global okuma | Ortak bellek | Diğer |
|---|---|---|---|
| `attn_decode_f16` | 6 × `ld.global.b16` | 27 × `ld.shared.b32` | 6 × `bar.sync`, 10 × `shfl`, 1 × `popc` |
| `attn_decode_split_f16` | 3 × `ld.global.b16` | 7 × `ld.shared.b16`, 11 × `ld.shared.b32` | 3 × `bar.sync`, 10 × `shfl`, 1 × `popc` |
| `attn_decode_merge_f16` | 8 × `ld.global.b32` | yok | 3 × `ex2.approx.ftz.f32` |

Dikkat edilecekler:

1. **Yerel bellek (`ld.local`) yok.** Diziler yazmaçlarda kaldı.
2. **`count_ones()` tek bir `popc.b32` oldu.** Yuva numarası için döngü gerekmiyor.
3. **`libm::expf` bir `ex2.approx.ftz.f32` oldu:** `exp(x) = 2^(x · log2 e)`. Donanımın hızlı yaklaşık üs komutu. Doğruluğu softmax için yeter, testler bunu gösteriyor.
4. **64'e bölme yok.** `i / 64` ve `i % 64` kaydırma ve maskeye dönüştü, çünkü 64 bir sabit ve 2'nin kuvveti.
5. **Parça yüklemesi 2 baytlık (`ld.global.b16`).** Birleşik olsa da her okuma komutu az veri taşır. Bu, sınıra kalan mesafenin bir sebebi olabilir. Alıştırma 1'e bakın.

Nsight Compute'ta `attn_decode_split_f16` için "Memory Workload Analysis" bölümüne bakın: L2'den gelen bayt, DRAM'den gelen bayta yakın olmalı (her K ve V satırı bir kez okunur). `attn_decode_f16`'da ise "Occupancy" düşük görünür: tek satırda SM başına tek blok.

## 7. Alıştırmalar

1. Parça kernel'inde K ve V'yi `Q16` (16 baytlık) okumalarla yükle (Ders 4). PTX'te `ld.global.v4` görünüyor mu? 1 × 1500 süresi 40 µs'ye ne kadar yaklaşıyor?
2. Parça boyutunu 64'ten 128'e çıkar. Blok sayısı yarıya iner, birleştirme işi azalır. Tek satırda hangisi daha hızlı? Ortak bellek kullanımı 48 KB sınırını geçiyor mu?
3. Birleştirme kernel'ini parça kernel'ine katmanın bir yolunu düşün (örneğin son biten blok birleştirsin, bir atomik sayaçla). Bir kernel başlatması kaç µs kazandırır?
4. DTW geçişi (224 × 1500) için puanları `mma.sync.m16n8k8` ile hesaplayan bir sürüm tasarla. `FlashTc`'den farkı ne olmalı (nedensel maske, `nq ≠ nk`, softmax çıktısı)?
5. KV cache'i f16 yerine f32 tutsaydık bir adımın okuduğu bayt ne kadar artardı? Süre?
