# Rust CUDA Phase 2: handoff after Task 7 (2026-11-25)

The branch is `rust-cuda-phase2`. The last commit is **`1e7dcc5`** (`feat(tts): run the OmniVoice forward pass on the GPU`). The tree is clean.

## 1. The state in one paragraph

The OmniVoice main model (Qwen3-0.6B, 28 layers, bidirectional, 8 audio codebooks) runs on the Rust CUDA kernels. One forward pass over the real first-pass input of script s1 matches the PyTorch fixtures: the audio-head argmax agrees on **1692 of 1712** positions (98.8%), and every hidden tap is closer to the f32 model than the engine's own fp16 run is. The text path (tokenizer, layout, frame estimate) and the deterministic masked decode are written and tested; the product mode (Task 10) and the measurements/lesson/review (Task 12) remain.

## 2. What each task produced

| Task | State | Where |
|---|---|---|
| 1, the reference fixtures | done | `tools/reference/omnivoice_reference.py`, `omnivoice_scripts.json`, `scripts\tts-baseline.ps1` |
| 2, config and index | done | `crates/jeuvoice-omnivoice/src/{config,index,paths}.rs` |
| 3, RoPE and q/k-norm, lesson 10 | done | `ops/rope.rs`, lesson `10-rope-qk-norm.md` |
| 4, SwiGLU, lesson 11 | done | `ops/swiglu` in `ops/llm.rs`, lesson `11-swiglu.md` |
| 5, the 8-codebook gather | done | `embed_mixed`, `embed_audio_sum` |
| 6, weights and the LoRA merge | done | `src/{lora,weights}.rs` |
| 7, the forward pass | done | `src/forward.rs`, `tests/forward.rs` |
| 8, bidirectional GQA attention | done | `ops/attention.rs` (`attention_bidir`) |
| 9, the masked decode loop | done (deterministic) | `src/decode.rs`, `tests/decode.rs`, `tests/decode_gpu.rs` |
| 10, the product path and seeds | **next** | `src/sample.rs` (does not exist) |
| 11, text layout and the frame estimate | done | `src/text.rs`, `src/duration.rs`, `src/lang.rs`, `tests/text.rs` |
| 12, render tool, measurements, lesson 12, review | later | `examples/render_tokens.rs` (does not exist) |

Task 11 must come before Task 9: the decode needs the step-0 sequence, and that sequence is the text layout. Recorded in the ledger.

## 3. Measurements (RTX 3050 Ti Laptop, 4 GB)

The forward pass against the fixtures (`--test forward`, 2 rows, 486 positions):

| Tap | ours vs f32, max / mean | the engine's fp16 run vs f32 |
|---|---|---|
| `embed` | 0.0007 / 0.000020 | 0.0007 / 0.000020 |
| `hidden_00` | 0.0117 / 0.000155 | 0.0215 / 0.000188 |
| `hidden_13` | 0.2119 / 0.001000 | 0.2449 / 0.001509 |
| `hidden_27` | 18.70 / 0.03278 | 47.26 / 0.05215 |
| `post_norm` | 0.2294 / 0.000792 | 0.3675 / 0.001345 |
| the logits | 98.8% argmax, max 0.059 | 94.2% argmax (fp16 vs f32) |

Other numbers:

| Item | Value |
|---|---|
| one pass, `[2, 486]` | ~112 ms (first pass 656 ms: cuBLAS and module setup) |
| the engine's token stage | 4.6 s at `num_step = 32` |
| our token stage, projected | 32 × 112 ms ≈ 3.6 s (**above the 2.3 s criterion**) |
| weights, f16 matrices and f32 norms | 1225 MB |
| activations (planned for `T_MAX = 1024`, 2 rows) | 138 MB |
| arena total | 1364 MB |
| load | 6.4 s (criterion: 25 s) |
| tests | 423 plain, 111 GPU, 70 whisper GPU |

## 4. Rulings that the next agent must not re-litigate

1. **RoPE uses the half-split pair `(i, i + d/2)`** (Hugging Face `rotate_half`), not the interleaved `(2i, 2i+1)` of GPT-J. The interleaved kernel was shape-correct and speed-correct and still gave 6.5% logit agreement. `RopeKind` is `HalfSplit | HalfSplitShared | Interleaved`; the product path uses `HalfSplit` (the forward pass) and should use `HalfSplitShared` if `d <= 128`.
2. **The adapter's `audio_embeddings` and `audio_heads` are full `modules_to_save` tables**, not LoRA. The loader replaces them (`lora::replaced_name`). The base head gives 40.7% argmax against the model's own logits; the adapter's gives 95.6%.
3. **The fixtures must be saved contiguous.** `cond_logits_00` was a strided slice of `batch_logits`, and safetensors wrote the storage, so codebooks 1 to 7 were garbage. Fixed with `.contiguous()`.
4. **The tap test asserts a ratio, not an absolute tolerance**: our error against f32 must be at most twice the engine's own fp16 error, plus `2e-2` (max) and `1e-5` (mean). At `hidden_27` the magnitude has a mean of 54.9, so an absolute tolerance is meaningless.
5. **The attention is bidirectional over the packed sequence.** Pad rows see only themselves (`omnivoice.py:1339-1347`).
6. **The residual stream is f32, the GEMM input f16, the norms f32.** A weight name that contains `norm` stays f32.
7. **The audio tokenizer stays on the CPU.** On the 4 GB card it thrashed: 202 s against 6 s.
8. **Alignment:** the ids, mask and lengths of a second batch row start off the 256-byte alignment. Use `Tensor::rows_unaligned` and `Arena::slice_unaligned` for scalar-only tensors, never for cuBLAS or vectorised kernels.
9. **Batch independence is a test**: the two identical rows of one pass must give identical logits. It caught a key-position table that was sliced from the query array.
10. **The product mode** compares the frame count and the token share (criterion 2); the RNG stream of the engine is not reproducible (W1).

## 5. Open items

- **criterion 1 is not reachable as written, and the measurement says so.** The engine's own fp16 run keeps only 20.0% of the tokens of its own fp32 run (s1; 14.9% and 14.1% on s2 and s3). The score of a position is `log_prob_max - 5·codebook`, and the six best positions of the first step span 0.05 with a 0.002 gap at the top, so any change of arithmetic reorders them and the whole trajectory follows. We keep 19.8% of the fp16 tokens and 23.7% of the f32 tokens on s1, within 0.3 points of the engine's own dtype change. The supervisor must decide how to restate criterion 1: per-position accuracy (98.9% argmax), similarity to the f32 model, or a by-ear check in Phase 3.
- **The time criterion.** 112 ms per pass is about 3.6 s at 32 steps. Task 12 must find about 1.3 s. The attention bench says 28 × 1.2 ms ≈ 34 ms; the rest is the GEMMs (small M = 972) and the layout copies.
- **The decode time.** s1 takes 4.6 s for 214 frames against the 2.3 s criterion. The forward pass is 3.6 s of it; the guided peak on the CPU is about 1 s. Task 12 must move the guidance to a kernel and find the rest.
- **The prompt frames** come from the engine's audio tokenizer (a fixture). Encoding a new voice is Phase 4.

## 6. Commands

```
scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels
scripts\cargo.cmd test
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
```

The real-model tests need the 4 GB card alone: run one file per process, `--test-threads 1`.

## 7. The files that matter

| File | Why |
|---|---|
| `crates/jeuvoice-omnivoice/src/forward.rs` | the pass, 28 blocks, the final norm, the audio head |
| `crates/jeuvoice-omnivoice/src/weights.rs` | the arena plan, `Activations`, `BATCH = 2`, `T_MAX = 1024` |
| `crates/jeuvoice-gpu/src/ops/attention.rs` | `attention_bidir` and `BidirScratch` |
| `crates/jeuvoice-gpu/src/ops/rope.rs` | the three rope kinds |
| `engine/.venv/Lib/site-packages/omnivoice/models/omnivoice.py` | the behaviour reference: `_prepare_inference_inputs:1198`, `_generate_iterative:1275`, `_predict_tokens_with_scoring:1429` |
| `engine/.venv/Lib/site-packages/omnivoice/utils/duration.py` | `RuleDurationEstimator:208` |
| `.superpowers/sdd/phase2/progress.md` | the local ledger with every ruling and measurement (gitignored) |
