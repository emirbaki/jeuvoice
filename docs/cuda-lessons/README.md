# CUDA dersleri

Bu dersler JeuVoice'un Rust GPU kernel'lerini anlatır. Her ders bir kernel'i alır: modelde ne iş yaptığını, neden gerektiğini, önce basit (naive) hâlini, sonra her hızlandırma adımını ölçümleriyle gösterir.

Kernel'ler Rust ile yazılır ve NVIDIA'nın **cuda-oxide** derleyicisi onları PTX'e çevirir. Kod `crates/jeuvoice-gpu` içindedir.

## Dersler

| No | Konu | Durum |
|---|---|---|
| 0 | [GPU temelleri ve araç zinciri (RMSNorm)](00-gpu-temelleri.md) | hazır |
| 1 | [Eleman eleman kernel'ler: bias, GELU, residual](01-eleman-eleman.md) | hazır |
| 2 | [LayerNorm](02-layernorm.md) | hazır |
| 3 | Softmax ve "online softmax" | Faz 1 |
| 4 | GEMM: önce cuBLAS, sonra kendi `mma.sync` GEMM'imiz | Faz 1 |
| 5 | Attention ve flash attention | Faz 1 |
| 6 | conv1d | Faz 1 |
| 7 | Log-mel spektrogram (matris çarpımı olarak STFT) | Faz 1 |
| 8 | Decoder döngüsü: KV cache, argmax, CUDA Graphs | Faz 1 |
| 9 | Kelime zamanları: hizalama başları ve DTW | Faz 1 |

## Çalıştırma

Testler (GPU gerekir):

```powershell
scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
```

Ölçüm tablosu:

```powershell
scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
```

```powershell
target\release\gpu-bench.exe rmsnorm
```

Son komuttaki `rmsnorm` bir süzgeçtir. Boş bırakırsan bütün satırlar çalışır.

Gerekli araçların listesi `docs/development-log.md` içinde, "Status (Rust CUDA, Phase 0)" bölümündedir.
