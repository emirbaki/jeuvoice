# Ders 8: Whisper decoder'ı

Bu ders decoder'ın GPU kernel'lerini anlatır. İki bölümü var:

- **Bölüm A: token gömme (embedding).** Decoder'a giren her token'ı 1280 sayılık bir vektöre çeviren kernel.
- **Bölüm B: decoder attention ve KV cache.** Plan Task 8 ile eklenecek.

Kod: `crates/jeuvoice-gpu/src/ops/embed.rs` ve kernel'ler `embed_f16`, `embed_f16x2`, `embed_one_f16x2` (`crates/jeuvoice-gpu/src/kernels/mod.rs`). Testler: `crates/jeuvoice-gpu/tests/embed.rs`.

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
