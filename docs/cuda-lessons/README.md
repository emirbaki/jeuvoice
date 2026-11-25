# CUDA dersleri

Bu dersler JeuVoice'un Rust GPU kernel'lerini anlatır. Her ders bir kernel'i alır: modelde ne iş yaptığını, neden gerektiğini, önce basit (naive) hâlini, sonra her hızlandırma adımını ölçümleriyle gösterir.

Kernel'ler Rust ile yazılır ve NVIDIA'nın **cuda-oxide** derleyicisi onları PTX'e çevirir. Kod `crates/jeuvoice-gpu` içindedir.

## Dersler

| No | Konu | Durum |
|---|---|---|
| 0 | [GPU temelleri ve araç zinciri (RMSNorm)](00-gpu-temelleri.md) | hazır |
| 1 | [Eleman eleman kernel'ler: bias, GELU, residual](01-eleman-eleman.md) | hazır |
| 2 | [LayerNorm](02-layernorm.md) | hazır |
| 3 | [Softmax ve "online softmax"](03-softmax.md) | hazır |
| 4 | [GEMM: önce cuBLAS, sonra kendi tel döşememiz](04-gemm.md) | hazır |
| 5 | [Attention ve flash attention](05-attention.md) | hazır |
| 6 | [conv1d: doğrudan ve im2col + GEMM](06-conv1d.md) | hazır |
| 7 | [Log-mel spektrogram: matris çarpımı olarak DFT](07-mel.md) | hazır |
| 8 | [Decoder: token gömme, KV cache ve decoder attention](08-decoder.md) | hazır |
| 9 | [Kelime zamanları: hizalama başları ve DTW](09-dtw.md) | hazır |
| 10 | [RoPE ve kafa başına RMSNorm (Qwen3)](10-rope-qk-norm.md) | hazır |
| 11 | [SwiGLU](11-swiglu.md) | hazır |

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
