# Ders 9: Kelime zamanları — hizalama başları ve DTW

Bu ders diğerlerinden farklı: iş **CPU'da** ve "GPU'ya taşıyalım mı?" sorusunun cevabı **hayır**. Burada öğrendiğimiz şey iki tane:

1. Bir algoritmayı 8 kat hızlandıran şey donanım değil, **veri yapısı** olabilir (medyan filtresi).
2. Her iş GPU'ya taşınmaz. 10 ms'lik bir işi GPU'ya taşımak, kazandırmadığı gibi karmaşıklık getirir. Bunu ölçerek gösteriyoruz.

Kod: `crates/jeuvoice-whisper/src/decode/dtw.rs` (CPU) ve `crates/jeuvoice-whisper/src/cuda/decoder.rs` (`alignment` geçişi, GPU).

Ölçümler: aynı makine, `--release`, tek pencere (6 baş, 130 token, 1500 ses konumu). `scripts\cargo.cmd test -p jeuvoice-whisper --release --test decode_dtw dtw_cpu_cost -- --nocapture`.

## 1. Ne yapıyor?

Decoder token'ları üretirken iki tür zaman bilgisi verir:

| Bilgi | Nereden | Duyarlılık |
|---|---|---|
| `t0`, `t1` (santimetre) | timestamp token'ları `<\|0.00\|>` … `<\|30.00\|>` | 20 ms'lik kuantalama, segment sınırında |
| `t_dtw` (kelime başı) | cross-attention ağırlıkları + DTW | ses konumu = 20 ms |

Token zamanı `t0`, bir token'ın segment içindeki payına göre bölünür (`token_level_times`). Bu kaba bir tahmindir: 20 ms kuantalama ve segment içi eşit dağıtım varsayımı vardır.

**DTW** daha iyisini yapar. Whisper'ın bazı cross-attention başları ("hizalama başları") bir token'ı yazarken sesin **hangi konumuna** baktığını gösterir. large-v3-turbo için bu 6 baş `(katman, baş)` olarak `(2,4), (2,11), (3,3), (3,6), (3,11), (3,14)`'tür.

Adımlar (`whisper_exp_compute_token_level_timestamps_dtw`):

```
w[h][t][a]   = DTW geçişinin cross-attention olasılıkları   [6][token][1500]
w            = token'lar boyunca normalize (ortalama 0, varyans 1)
w            = her satırda medyan filtresi (genişlik 7)
cost[t][a]   = -ortalama_h w[h][t][a]                       [token-3][1500]
yol          = DTW(cost)             # (0,0) → (token-4, 1499), en ucuz yol
t_dtw[token] = yolun o token satırına girdiği ses konumu × 20 ms
```

DTW (dynamic time warping) yolunun kuralı: her adımda sağa, aşağıya veya çapraz ilerle. Üç komşudan **en ucuzunu** seç. Bu, "token'lar sırayla, ses de ileri doğru" kısıtını korur. Yolun bir token satırına girdiği ilk sütun, o token'ın başlangıç zamanıdır.

## 2. Neden gerekli, ne sıklıkla çalışır?

Uygulama altyazı gösterir ve kelimeyi vurgular. Yanlış başlangıç, altyazının kayması demektir. Faz 1'in 2. kriteri bu yüzden var: **medyan kelime başı farkı ≤ 50 ms**. Ölçtüğümüz değer 0 ms.

Çalışma sıklığı: her 30 saniyelik pencere için **bir kez** DTW geçişi, sonra segment başına bir kez `token_level_times`. Test setinde 4 pencere var, yani 4 DTW geçişi.

Maliyet: bir pencere için 190 bin hücrelik bir dinamik programlama ve 1,17 milyon cross-attention değeri. Aşağıda ölçüyoruz.

## 3. Naive hâl: her konum için bir sıralama

İlk sürüm doğruydu ve whisper.cpp'nin portuydu. Medyan filtresi şöyleydi:

```rust
for k in 0..n {                      // ses konumu
    let mut window = Vec::with_capacity(7);
    for off in -3..=3 {
        let mut i = k + off;         // yansıtmalı kenar
        if i < 0 { i = -i } else if i >= n { i = 2 * (n - 1) - i }
        window.push(row[i as usize]);
    }
    window.sort_by(|a, b| a.total_cmp(b));
    out[k] = window[3];
}
```

Ölçüm (tek pencere, release):

| Aşama | Süre |
|---|---|
| Token'lar boyunca normalize | 3,25 ms |
| **Medyan filtresi (genişlik 7)** | **67,3 ms** |
| Maliyet matrisi, tüm aşamalar | 75,2 ms |
| DTW yolu + geri iz | 2,03 ms |
| **Toplam** | **77,2 ms** |

İşin **%87'si medyan filtresinde**. Neden?

- Her konum için bir `Vec` dolduruluyor, sıralanıyor, sonra atılıyor. 1,17 milyon kez.
- `sort_by` bir **karşılaştırma fonksiyonu** çağırır (`total_cmp`). Yedi eleman için bu, gerçek işin kat kat üstünde bir yük.
- Kenar yansıtması her eleman için dallanma (`if i < 0 …`) demek. Dallanma tahmini bozulur.

## 4. Hızlandırma: sıralama ağı (77,2 → 9,9 ms)

Yedi elemanın medyanı için sıralama rutini gerekmez. **Sıralama ağı** (sorting network) yeterlidir: sabit sayıda karşılaştırma-değiştirme, dallanma yok, ayırma yok.

```rust
#[inline]
fn median7(mut v: [f32; 7]) -> f32 {
    fn ce(v: &mut [f32; 7], i: usize, j: usize) { if v[i] > v[j] { v.swap(i, j); } }
    ce(&mut v, 0, 6); ce(&mut v, 2, 3); ce(&mut v, 4, 5);
    ce(&mut v, 0, 2); ce(&mut v, 1, 4); ce(&mut v, 3, 6);
    ce(&mut v, 0, 1); ce(&mut v, 2, 5); ce(&mut v, 3, 4);
    ce(&mut v, 1, 2); ce(&mut v, 4, 6);
    ce(&mut v, 2, 3); ce(&mut v, 4, 5);
    ce(&mut v, 1, 2); ce(&mut v, 3, 4); ce(&mut v, 5, 6);
    v[3]                                  // sıralanmış dizinin ortası
}
```

19 karşılaştırma, 7 elemanı tamamen sıralar (Knuth, TAOCP 5.3.4). Ağ **tam sıralama** yapar, yani medyanı almak için ilk 3 elemanı da sıralamış oluruz. Daha az karşılaştırmalı bir "sadece medyan" ağı da vardır (alıştırma 1).

İkinci değişiklik: **kenar yansıtmasını satırın uçlarına taşıdık.** Satırın ortasında (`3 ≤ k < n-3`) dallanma yok, sadece 7 ardışık okuma:

```rust
} else {
    let base = k - 3;
    v.copy_from_slice(&row[base..base + 7]);
}
```

Ölçüm (aynı pencere):

| Aşama | Önce | Sonra | Kazanç |
|---|---|---|---|
| Normalize | 3,25 ms | 2,32 ms | — |
| **Medyan filtresi** | **67,3 ms** | **3,71 ms** | **18×** |
| Maliyet matrisi, tümü | 75,2 ms | 8,02 ms | 9,4× |
| DTW yolu | 2,03 ms | 1,86 ms | — |
| **Toplam (pencere)** | **77,2 ms** | **9,9 ms** | **7,8×** |

18 kat, yedi elemanı sıralamanın maliyetinden geliyor: 19 karşılaştırma, 7 yükleme, hiç ayırma. `sort_by` sürümü her çağrıda Vec kontrolü, fonksiyon çağrısı ve sınır kontrolü yapıyordu.

**Doğruluk:** `median_filter_width_7_matches_a_sort_on_random_rows` testi, 5 farklı satır uzunluğunda (8'den 1500'e) ve 3 satırda, ağın sonucunu sıralamalı referansla **birebir** karşılaştırır. Üstelik gerçek ses karşılaştırması da geçiyor: `t_dtw` medyan farkı **0 cs**, en büyük 4 cs (whisper.cpp'ye karşı, 3 klibin 379 token'ı).

## 5. Sınır ve neden GPU'ya taşımıyoruz

Tek pencere 9,9 ms. 4 pencerelik test setinde yaklaşık **40 ms**, yani 3,9 saniyelik toplam sürenin **%1'i**.

Uçtan uca ölçüm bunu doğruluyor: aynı klibi DTW açık ve kapalı koşturduk (`dtw_cost_in_a_transcription` testi). Sonuçlar 1,305 s ile 1,927 s arasında **sıçradı**, DTW'li koşular DTW'sizlerden yavaş çıkmadı. Yani DTW'nin maliyeti, GPU saat hızının koşudan koşuya oynamasının (yaklaşık ±0,1 s) altında kalıyor.

Bu yüzden DTW CPU'da kalıyor:

1. **İş küçük.** 190 bin hücrelik bir dinamik programlama. Bir GPU kernel'i için çekirdek başına yeterli paralellik yok: DP zinciri çaprazda **seri**dir, ancak dalga cephesi (anti-diyagonal) ile paralelleşir ve o zaman her adımda senkronizasyon gerekir.
2. **Bağımlılık.** Girdi, decoder'ın ürettiği cross-attention olasılıklarıdır. O zaten GPU'da hesaplanıyor (hizalama geçişi, pencere başına 3,5 ms). Kalan iş CPU için yeterince küçük.
3. **Doğruluk riski.** Medyan filtresi ve DTW adım kuralı whisper.cpp ile birebir aynı olmalı. Portu değiştirmek, kriter 2'yi (50 ms) riske atar. Kazanç %1'in altındayken bu risk alınmaz.

**Ders:** GPU her zaman cevap değildir. Önce **ölç**, sonra taşı. 10 ms'lik bir işi taşımak, kod ve risk ekler, hız eklemez.

## 6. Nereye bakmalı (CPU tarafı)

PTX yok. Bunun yerine:

```powershell
# Derleyicinin ürettiği assembly: minss/maxss (SSE) veya vminps/vmaxps (AVX) görmelisin.
cargo asm --release jeuvoice_whisper::decode::dtw::median_filter
```

- **Beklenen:** `median7` satır içine alınmış, `ce` çağrıları `minss`/`maxss` komutlarına dönmüş, döngü vektörleşmiş olabilir (LLVM ardışık konumları birlikte işleyebilir).
- **Beklenmeyen:** `call` komutları (fonksiyon çağrısı kaldıysa ağ satır içine alınmamış) ve `movups` yığın kopyaları.
- **DTW yolu** (`dtw_path`) iç döngüsü: 3 karşılaştırma, 1 toplama. 190 bin hücre 1,86 ms ise hücre başına ~10 ns: gecikme sınırlı, bant genişliği değil. `cost` ve `trace` matrisleri 127 × 1501 × (4 + 1) bayt ≈ 950 KB, L2'de kalır.

Windows'ta profil: Windows Performance Analyzer (`wpa`) veya Visual Studio'nun CPU profiler'ı. Linux'ta `perf record` ve `cargo flamegraph`.

## 7. Alıştırmalar

1. **Sadece medyan ağı.** Tam sıralama yerine medyanı veren daha kısa bir ağ yaz (7 eleman için en az 13 karşılaştırma yeterlidir). Ölç, kazanç var mı?
2. **Kayan pencere.** Sıralı 7'lik listeyi koru: her adımda bir eleman çıkar, bir eleman ekle (insertion). Konum başına O(1). Ölç ve karşılaştır: küçük pencerede kazanç var mı, bellek erişimi nasıl değişiyor?
3. **Vektörleştir.** 8 konumu aynı anda işle (yapı-dizileri, SoA). `cargo asm` ile `vminps`/`vmaxps` gördün mü? Hız ne oldu?
4. **Medyanı at.** Medyan filtresini kapatıp `t_dtw`'nin whisper.cpp'den ne kadar kaydığını ölç. Kriter 2 (50 ms) hâlâ tutuyor mu? Demek ki filtre neyi düzeltiyor?
5. **Bellek.** `dtw_path` tüm `cost` ve `trace` matrislerini tutuyor (≈ 950 KB). Yolu geri izlemek için daha az bellek yeter mi (Hirschberg böl ve yönet)? Ne kazanırdın?
6. **GPU'da DTW.** Anti-diyagonal dalga cephesi ile bir kernel yaz. 10 ms'lik iş için kazanç var mı? Senkronizasyon maliyeti ne kadar?
