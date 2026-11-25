# Ders 13: 128 genişliğinde kafa için flash attention (d128)

OmniVoice'un (Qwen3) attention'ı ders 5'teki Whisper attention'ından iki yönde ayrılır: kafa genişliği **128** (64 değil) ve maske **iki yönlü** (causal değil). Ders 5'in tensor-core flash kernel'i 64 genişliğinde çalışıyordu. Bu ders aynı fikri 128'e taşıyor, nerede pahalandığını ölçüyor ve üretimde iki yolu karşılaştırıyor.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`flash_attn_tc128_f16`), `crates/jeuvoice-gpu/src/ops/attention.rs` (`attention_flash128`, `attention_bidir`, `attention_bidir_grouped`).

Ölçümler: RTX 3050 Ti Laptop, 20 SM, bellek sınırı 192 GB/s, fp16 tensor zirvesi 15,2 TF/s. `target\release\gpu-bench.exe attn_p5`. Üretim şekilleri: `b=1`, 16 sorgu kafası, 8 anahtar kafası (grup 2), 486 ve 214 konum.

## 1. Ne yapıyor?

```
out[b, h, p, :] = softmax_p'( q[b, h, p, :] · k[b, h/2, p', :] / √128 ) · v[b, h/2, p', :]
```

Her sorgu kafası, kendi anahtar kafasının **bütün** konumlarını görür (iki yönlü). Nedeni modelde: OmniVoice maskeli yinelemeli çözmede (ders 12) her adımda bütün diziyi yeniden işler, geçmişi bir KV cache'de tutmaz. Bu yüzden maske yok ve ileriye de bakılır.

Şekiller: `q`, `out` `[1, 16, 486, 128]` f16, `k`, `v` `[1, 8, 486, 128]` f16. Bir kafa 128 değer taşır: bir `m16n8k8` mma için gereken k adımı sayısı `128 / 8 = 16`.

## 2. Neden gerekli, ne sıklıkla çalışır?

Attention, transformer bloğunun tek "her konum her konumu görür" katmanıdır; modelin bağlamı buradan gelir. Maliyeti konum sayısının karesiyle büyür: 486 konumda bir kafa için `486 · 486 · 128 · 2 · 2 = 968 MFLOP` (q·kᵀ ve probs·v).

Sıklık: 28 katman × adım başına iki geçiş (koşullu 486, koşulsuz 214) × 32 adım = **1792 kez** bir render'da. Adım başına yaklaşık 14,7 ms; 77 ms'lik adımın %19'u. Bu, GEMM'lerden sonra ikinci en büyük kalem.

## 3. Naive hâl: skorları DRAM'e yaz

Bugüne kadarki yol (grup attention, `attention_bidir_grouped`) üç adım:

1. cuBLAS ile `S = q·kᵀ` (16 kafa, her biri `486×486`), f16 olarak DRAM'e yazılır: **7,6 MB**.
2. `softmax_rows` bu düzlemi okur ve yerinde yazar: 7,6 MB okuma + 7,6 MB yazma.
3. `probs · v` ikinci bir cuBLAS çağrısı.

Ölçüm (`b=1`, 16 kafa):

| Konum | Süre | Hız | Zirvenin yüzdesi |
|---|---|---|---|
| 486 | 468 µs | 4,1 TF/s | %27 |
| 214 | 164 µs | 2,3 TF/s | %15 |

Yani kartın tensor çekirdekleri %73 boş duruyor: iş, skor düzleminin DRAM turuna takılıyor. 214 konumda oran daha da kötü, çünkü GEMM'ler küçülüyor ama düzlem trafiği aynı oranda küçülmüyor.

## 4. Hızlandırma adımları

### 4a. Tensor core'lu flash, d=128

Ders 5'teki `flash_attn_tc_f16` (d=64) temel alındı. Aynı iskelet: blok başına 64 sorgu satırı (4 warp, her warp 16 satır), anahtarlar 32'lik dilimlerde shared memory'de, çevrimiçi softmax, skor düzlemi hiç DRAM'e çıkmaz. Değişen üç şey:

1. **16 Q fragmenti** (d64'te 8). `qa[ks]` bu sefer `ks = 0..15`, her biri 8 kafa boyutu.
2. **Her `ldmatrix_x4` 32 boyut kapsıyor**, 128 boyut için dört grup: sütun 0, 32, 64, 96. Skor için `s += qa[ks]·b0[ks] + qa[ks+4]·b1[ks] + qa[ks+8]·b2[ks] + qa[ks+12]·b3[ks]`.
3. **16 O akümülatörü** (d64'te 8): `o[dt]` her biri 4 f32, `dt = 0..15`. `V` için `ldmatrix_x4_trans` yine dört grupta okunur.

Kayıt bütçesi: `qa` 32, `o` 64, `s` 16, `m/l` 4, adresler ve indeksler ~30 → **yaklaşık 150 kayıt/thread**. PTX'te `ld.local` ve `st.local` **sıfır** (bu şart: ders 5'te local memory'ye taşma 7,5 kat yavaşlatmıştı; tüm fragment ve akümülatör dizileri `unroll!` ile sabit indekslenir).

Shared memory: `K` ve `V` dilimleri `32 × (128 + 8)` f16 = 8,7 KB, ikisi 17,4 KB. 150 kayıt ve 17,4 KB ile SM başına **3 blok** (12 warp) sığar: %19 doluluk.

Ölçüm:

| Konum | Grup yol | Flash d128 | Kazanç |
|---|---|---|---|
| 486 | 468 µs (4,1 TF/s) | 323 µs (6,0 TF/s) | −%31 |
| 214 | 164 µs (2,3 TF/s) | 94 µs (4,0 TF/s) | −%43 |

### 4b. Anahtar dilimi 64: kazanç yok

İlk deneme 32 anahtar/dilim. "Daha büyük dilim, daha az bariyer" hipoteziyle 64 denendi: shared memory 34,8 KB olur, SM başına **1 blok** kalır (12 warp → 4 warp).

Ölçüm: 486'da 331 µs (fark yok), 214'te 103 µs (32'lik dilimle aynı, ilk ölçümlerdeki 95 µs'e göre kötü). Karar: **32 kalır**. Bu derste d64'ün 64 anahtarı iyiydi; d128'de aynı sayı shared memory'yi ikiye katlıyor ve doluluğu öldürüyor. Sayılar boyuta bağlı, kural değil.

### 4c. Tuzak: ölçek

İlk sürüm `ATTN_SCALE` (yani `1/√64 = 0.125`) kullanıyordu. 128 genişliğinde doğrusu `1/√128 = 0.0884`. Test bunu yakaladı: `mean_abs 0.011`, `max_abs 0.083`, yani sonuç "neredeyse doğru" görünüyordu. Skorlar %41 büyük olunca softmax sivrileşiyor ve düşük ağırlıklı değerler kayıyor. Denklem: `softmax(a·s)` ile `softmax(s)` aynı değildir; hata mesajı vermez, sadece kaliteyi düşürür.

## 5. Kartın sınırı ve ne kadar yaklaştık

Zirve 15,2 TF/s; flash 5,8 TF/s ile %38'inde. Kalan boşluğun üç sebebi:

1. **Doluluk**: SM başına 12 warp (64'ün %19'u). 3 blok sınırını kayıtlar belirliyor (150 × 128 × 3 ≈ 57,6 bin kayıt).
2. **Bariyer**: her dilimde `sync_threads`, ardından çevrimiçi softmax'ın sıralı zinciri (`max → shuffle → exp → shuffle → ölçekle`) iki mma grubunu birbirinden ayırıyor. Bu zincir sırasında tensor çekirdekleri boş.
3. **K/V yeniden okuma**: her satır bloğu kendi kafasının bütün K/V'sini okur: 8 satır bloğu × 249 KB × 16 kafa = 32 MB (dilim başına). 4 MB L2 bunun çoğunu tutar (aynı kafanın 8 bloğu arka arkaya çalışır), ama yine de bellek yolunu kullanır.

Bir adımın payı: 323 + 94 = 417 µs × 28 katman = **11,7 ms**; eskiden 468 + 164 = 632 µs × 28 = 17,7 ms. Yani adım başına ~6 ms, 32 adımda **~0,19 s** kazanç. Son 10 µs ve 323 µs, paket 1'in `rcp.approx.f32` değişikliğinden geliyor (`div.rn` yerine tek komut).

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `flash_attn_tc128_f16` içinde **`ld.local` ve `st.local` sayısı 0 olmalı**. Bir tane bile varsa `o[16]` veya `qa[16]` local memory'ye düşmüş demektir; `unroll!` listelerini ve döngüleri kontrol et.
- `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` sayısı: dilim başına 128 (skor 64, P·V 64). 16 k adımı olması d64'ün 4 katı.
- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` (K için) ve `.trans` (V için) dörder kez.

Nsight Compute:

```powershell
ncu --set full -k regex:flash_attn_tc128 -c 2 target\release\gpu-bench.exe attn_p5
```

- **Tensor (SM) Throughput**: ~%38. Zirvenin yarısına bile gelmiyoruz; sebep 1 ve 2.
- **Occupancy**: 12 warp / 64 → "%18,75". `launch__registers_per_thread` ~150 çıkar.
- **Warp State → Barrier**: `sync_threads` beklemesi. Anahtar dilimini büyütmek bunu azaltır ama doluluğu düşürür (4b).
- **Memory Workload Analysis → L2 Hit Rate**: K/V için yüksek olmalı; düşükse blok sıralaması bozulmuş demektir.

## 7. Alıştırmalar

1. **Satır başına 32 sorgu.** `FT8_ROWS = 128` ve warp başına 2 satır bloğu (32 satır). Kayıt ~210'a çıkar, SM başına 1 blok kalır. K/V okuması yarıya iner; süre ne olur? Ölç.
2. **8 warp, 128 satır.** `FT8_THREADS = 256`, `FT8_ROWS = 128`. Kayıt/thread aynı kalır, SM başına 1 blok (8 warp) olur. Doluluk 12 warp'tan 8'e iner; bariyer maliyeti yarıya iner. Hangisi kazanır?
3. **Dilim 16.** `FT8_KEYS = 16`: shared 8,7 KB, daha çok bariyer, daha iyi doluluk. 214 konumda (7 dilim yerine 14) nasıl?
4. **Ölçeği unut.** `scale` parametresini 1,0 yap ve testin hangi eşikte patladığını gör. Bu, 4c'deki hatanın neden sessiz olduğunu gösterir.
5. **Maske ekle.** İki yönlü attention'da `valid` (kısa satır) maskesi ekle: satır `valid`'den sonra yalnız kendini görmeli (ders 12). Bunu flash'ta yapmak, grup yolu korumaktan daha mı kolay?
6. **K'yı shared'da tut, V'yi akıt.** V dilimini shared'a koymadan doğrudan global'den oku (her P·V adımında 32 anahtar). Shared yarıya iner, SM başına 6 blok sığar. Bant genişliği mi, doluluk mu kazanır?
