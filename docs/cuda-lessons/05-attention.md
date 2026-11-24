# Ders 5: Attention ve flash attention

Attention, encoder'ın en pahalı ikinci işidir. İki yolu vardır: **skor matrisini yazıp softmax'ı ayrı yapmak** (klasik yol, cuBLAS ile) ve **skorları hiç belleğe yazmadan tek kernel'de yapmak** (flash attention). Bu derste ikisini de yazıp ölçüyoruz. Sonuç ders kitabındaki gibi çıkmıyor: bu kartta, bu boyutta klasik yol **10 kat** hızlı. Nedenini adım adım görüyoruz.

Kod: `crates/jeuvoice-gpu/src/kernels/mod.rs` (`flash_attn_f16`, `split_heads_f16`, `merge_heads_f16`) ve `crates/jeuvoice-gpu/src/ops/attention.rs`.

Ölçümler: RTX 3050 Ti Laptop, `heads = 20`, `seq = 1500`, `d = 64`. `target\release\gpu-bench.exe attention`.

## 1. Ne yapıyor?

Bir kafa için, `q`, `k`, `v` her biri `[seq, 64]`:

```
scores = q · kᵀ / 8            // [seq, seq]
probs  = softmax(scores)       // her satırda
out    = probs · v             // [seq, 64]
```

`1/8 = 1/√64` ölçektir (ders 3). Sezgi: her sorgu satırı, anahtarlarla ne kadar uyuştuğuna göre değerleri karıştırır. "Hangi token'a ne kadar bakacağım" sorusunun cevabı budur.

Encoder'da 20 kafa vardır, yani 20 ayrı `[1500, 1500]` skor matrisi. Bunlar bağımsızdır: bir kafa diğerini etkilemez.

## 2. Neden gerekli, ne sıklıkla çalışır?

Her blokta bir kez, ama 20 kafayla birlikte. Yani bir blokta 20 × 1500 = 30 000 satır softmax. 32 blok için **960 000 satır**.

İşlem sayısı: `2 · 2 · heads · seq² · d = 4 · 20 · 1500² · 64 = 11,5 GFLOP` bir blok için; 32 blok için **368 GFLOP**. GEMM'lerin (640 GFLOP) yanında ikinci büyük kalem.

Aradaki fark şu: attention'da **skor matrisi** diye bir ara tensör var. `[20, 1500, 1500]` f16 = **90 MB**. Bu tensörü yazmak, okumak, tekrar yazmak, tekrar okumak 360 MB trafik eder. Kartta bu ~2 ms'dir. Flash attention'ın var olma sebebi tam olarak bu tensörü hiç oluşturmamaktır.

## 3. Klasik yol: skor matrisi + softmax + ikinci GEMM

`ops::attention(..., AttnKind::Naive)` üç adım yapar:

1. `blas.gemm_f16_batched`: her kafa için `q · kᵀ` → `scratch` (90 MB yazma).
2. `softmax_rows` (ders 3'ün register kernel'i): yerinde, satır satır (90 MB oku + 90 MB yaz).
3. `blas.gemm_f16_batched`: `probs · v` (90 MB oku + 17 MB yaz).

Ölçüm: **2,48 ms, 4,6 TFLOP/s, tensor sınırının %31'i.**

Üç adımın hepsi kendi başına iyidir: GEMM'ler %90 civarı, softmax %94. Ama toplam sürenin **yaklaşık %80'i scratch tensörünün trafiğidir**, hesaplama değil. Bu tam olarak flash attention'ın ortadan kaldırdığı şey.

## 4. Flash attention: skor matrisi yok, online softmax var

`AttnKind::Flash` tek kernel çalıştırır: `flash_attn_f16`.

- Bir blok, bir kafanın **8 sorgu satırını** işler (`FLASH_BR = 8`), 64 thread'i vardır (thread = boyut).
- Anahtarlar 64'lük dilimlerle gelir (`FLASH_BK`). K ve V dilimi shared memory'ye bir kez yüklenir ve 8 satır paylaşır.
- Skorlar **register'da** kalır. Satırın maksimumu ve üssel toplamı (`m`, `l`) her dilimde güncellenir: ders 3'teki online softmax.
- Çıkış `acc` de register'dadır: thread `t`, çıkışın `t`'inci boyutunu tutar.
- `scratch` hiç okunmaz ve yazılmaz.

Kritik nokta: **her satır grubu için `q` shared memory'de** durur, çünkü bir skor 64 boyutun tamamına bakar. İlk sürümde bu yoktu ve sonuç yanlıştı: kernel yalnızca kendi boyutunu çarpıyordu. Test bunu yakaladı (`attention_single_row`).

Ölçüm: **26,2 ms, 0,4 TFLOP/s, tensor sınırının %3'ü.**

Klasik yoldan **10 kat yavaş**. Flash'ın sildiği 360 MB trafiğe rağmen. Neden?

### 4.1. Ardışık bağımlılık (36,4 → 26,2 ms)

İlk sürümde her satır için tek bir toplama vardı:

```rust
dot += q[dd] * k[dd];   // 64 kez, hepsi aynı `dot`'a bağlı
```

Her toplama bir öncekini bekler. Shared memory okuması ~30 devir gecikmeli, toplama 4 devir. Zincir 64 eleman için en az 256 devir. GPU bu gecikmeyi ancak *başka işlerle* saklar; thread'in başka işi yoksa bekler.

Çözüm: **dört ayrı toplama** (`dot0..dot3`), 4'erli açılım. Birbirinden bağımsız dört zincir, gecikmenin 3/4'ünü saklar. Aynı şey `acc` döngüsünde de yapıldı.

Ölçüm: **36,4 → 26,2 ms.**

### 4.2. Sorgu bloğu büyütmek (36,4 → 35,3 ms, fark yok)

İlk flash sürümü blok başına **tek sorgu satırı** işliyordu. O zaman her satır K ve V'nin tamamını yeniden okur: `30 000 satır × 2 × 192 KB = 11,5 GB` trafik. Korkutucu görünür.

Blok başına 8 satıra çıkardık (`FLASH_BR = 8`): aynı dilim 8 satıra hizmet eder, trafik 8'e bölünür. Beklenen kazanç büyüktü.

Ölçüm: **36,4 → 35,3 ms.** Neredeyse hiç değişmedi.

Neden? Çünkü bir kafanın K ve V'si `2 × 192 KB = 384 KB` ve **L2 cache 4 MB**'dır. Aynı kafanın tüm sorgu blokları arka arkaya çalıştığı için (grid'de `blockIdx.y = kafa`), K ve V zaten L2'de durur. Yani "tekrar okuma" hiç DRAM'e inmiyordu. Ders 3'teki aynı sürpriz: kazandığını sandığın trafik zaten cache'te.

### 4.3. Gerçek darboğaz: doluluk (occupancy) ve skaler FMA

Flash 26 ms'de 26,2 ms × 1,76 GHz ≈ 46 milyon devir harcıyor. Yapılan iş 5,9 milyar FMA. 20 SM × 128 şerit = 2560 FMA/çevrim kapasitesiyle bu iş **1,3 ms** sürer. Yani 20 kat boşluk var.

İki sebep:

1. **Doluluk.** Blok 64 thread (2 warp) ve ~20 KB shared memory kullanıyor. Bir SM'de 100 KB shared var → 5 blok → **10 warp / 48**. Yani SM'lerin %80'i boş. Shared bellek gecikmesini saklayacak warp yok.
2. **Skaler FMA.** Klasik yolun GEMM'leri cuBLAS'ta **tensor core** kullanır: aynı komutta 1024 çarpma-toplama. Flash bizim yazdığımız gibi skaler FMA kullanır: 1 çarpma-toplama. Ders 4'teki farkın aynısı.

## 5. Kartın sınırı ve ne kadar yaklaştık

| Yol | Süre | Hız | Tensor sınırının yüzdesi | 90 MB skor matrisi |
|---|---|---|---|---|
| `attention_naive_f16` | **2,48 ms** | 4,6 TF/s | %31 | yazılır ve 3 kez okunur |
| `attention_flash_f16` (1 satır/blok) | 36,4 ms | 0,3 TF/s | %2 | hiç yok |
| `attention_flash_f16` (8 satır/blok, ILP) | 26,2 ms | 0,4 TF/s | %3 | hiç yok |

Klasik yol kazanıyor, çünkü onun ağır kısmını **cuBLAS tensor core'ları** yapıyor. Flash ise bizim skaler kernel'imiz. Flash'ın gerçek kazancı (360 MB trafik) bu farkın yanında küçük kalıyor.

**Karar:** Phase 1a ve 1b'de `AttnKind::Naive` kullanılır. Flash, tensor core'lu bir sürüm yazıldığında anlamlı olur (`mma.sync.m16n8k8`, ders 4'ün 5. alıştırması; `ptx_asm` engeli çözülünce).

Bu dersin asıl kazancı şu: "flash attention daha hızlıdır" cümlesi **tek başına doğru değildir**. Doğrusu: "flash attention ara belleği siler; kazancı, matris çarpımını tensor core'larda yaptığınız sürece ortaya çıkar."

## 6. PTX ve Nsight'ta neye bakmalı

PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`):

- `flash_attn_f16`: iç döngüde 4 × `fma.rn.f32` ve 8 × `ld.shared.u16` (q ve k için), `shfl.sync` (8 satırın maksimumu ve toplamı için), `bar.sync` × 4 her dilimde.
- `ld.local` **olmamalı**: `acc[8]`, `m[8]`, `l[8]`, `s[8]` register'da kalmalı. Kalmıyorsa register baskısı dizileri local memory'ye taşımıştır ve her şey yavaşlar.
- `split_heads_f16` çok basit bir kopyadır: `ld.global.b16` + `st.global.b16`, indeks bölmeleri. `[1500, 1280]` için ~5 µs.

Nsight Compute:

```powershell
ncu --set full -k regex:flash -o fla target\release\gpu-bench.exe attention
```

- **Occupancy → Achieved Occupancy**: ~%15-20 beklenir. **Theoretical** değeri ile karşılaştır: blok 2 warp olduğu için tavan düşüktür.
- **Warp State Statistics → Stall Long Scoreboard ve Stall MIO Throttle**: shared bellek bekleyen warp'lar. ILP ekleyince bu düşer.
- **Speed Of Light**: bellek %20'nin altında, hesap %10 civarı; yani hiçbir kısıt dolu değil. Bu, "latency bound" demektir.
- `attention_naive_f16` için: timeline'da üç kernel görürsün (GEMM, softmax, GEMM). İlk GEMM'in ardından gelen softmax 90 MB okur. Toplam sürenin çoğu buradadır.

## 7. Alıştırmalar

1. **Doluluk.** Bloğu 256 thread yap: 4 grup × 64 thread, her grup 8 satır. K/V dilimi dört gruba birden hizmet eder, SM'de 4 kat daha fazla warp olur. 26 ms'den nereye iniyor?
2. **Sekiz satır yerine 16.** `FLASH_BR = 16`. Register sayısı ikiye katlanır (`acc`, `m`, `l` 16'şar). Hangi noktada `ld.local` görünür ve hız düşer? PTX'ten kontrol et.
3. **`acc` döngüsünü kaydır.** `SP` yerine üsleri register'da tut (thread `t` yalnızca kendi anahtarının üssünü bilir). Bu, ikinci shared okumasını siler mi? Neden zor?
4. **Küçük `seq`.** `seq = 448` (decoder boyu) ile ölç. Skor matrisi 90 MB değil 8 MB olur. Flash yine kaybediyor mu? Nerede kazanmaya başlar?
5. **Tensor core.** Ders 4'ün 5. alıştırmasındaki `mma` kernel'ini attention'a uygula: `queries · keysᵀ` fragment'larla. Flash'ın klasik yolu geçtiği noktayı ölç.
6. **`split_heads` / `merge_heads` maliyeti.** Bunlar ayrı kernel'ler; toplam süreye etkisi ne? `[1500, 1280]` için ölç.
