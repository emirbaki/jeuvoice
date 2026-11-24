# Handoff: Rust CUDA inference (Whisper and OmniVoice in Rust kernels)

Date: 2026-11-24. Written by the previous agent for the next agent. Read `AGENTS.md` at the repo root first. It has the rules and the commands.

## 1. Start here

1. Check out the branch: `git switch rust-cuda-phase1a`. The last commit is `3e1cd12 fix: close the three Important findings of the Phase 1a review`.
2. Run the GPU tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`. Expected: 58 tests pass.
3. Run the plain tests: `scripts\cargo.cmd test`. Expected: 348 tests pass, with no failures.
4. Run the Whisper GPU tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels -- --include-ignored --test-threads 1`. Expected: all pass. The first run builds whisper.cpp with CUDA (about 20 minutes).
5. **Plan 1a is complete and reviewed.** The branch is not merged to `main` yet: the user decides. Continue with section 4.

## 2. The goal and why

The user wants three things, in this order:

1. **Learn CUDA kernel programming in practice.** This is the main reason. Every kernel gets a lesson in Turkish in `docs/cuda-lessons/`. The lessons say what the kernel does, why the model needs it, and how it gets faster, with measurements. A kernel without its lesson is not done.
2. **Remove the Python setup** (8 GB, 10–15 minutes on first start). Today the voice engine is a Python process (`engine/`, PyTorch + OmniVoice).
3. **Faster generation and less VRAM.**

The user decided: all inference code is Rust, and the kernels are also Rust, written with NVIDIA cuda-rust (**cuda-oxide**, the SIMT model). Large matrix multiplies use **cuBLAS**. Do not switch a kernel to CUDA C++.

The full design: `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md`. Read it before you plan anything new.

## 3. Where the work is

| Phase | Content | State |
|---|---|---|
| 0. GPU base | toolchain, `jeuvoice-gpu`, cuBLAS, arena, test and bench helpers, RMSNorm, lesson 0 | **done**, merged to `main`, pushed |
| 1a. Whisper encoder | plan `docs/superpowers/plans/2026-11-24-rust-cuda-phase1a-encoder.md`, 10 tasks | **done and reviewed** on `rust-cuda-phase1a`, not merged |
| 1b. Whisper decoder | decoder, decoding rules, DTW, `CudaWhisper` in the app, packaging, speed | no plan yet |
| 2. OmniVoice main model | Qwen3-0.6B backbone, masked iterative decoding, LoRA merge | not started |
| 3. Audio decoder | DAC decoder of the Higgs Audio v2 tokenizer | not started |
| 4. Clone encoder | HuBERT + DAC encoder | not started |
| 5. Python removal | port `dataprep.py` and `fetch_adapter`, remove `engine/` | not started |

### Plan 1a task state

| Task | State | Commit |
|---|---|---|
| 1. Phase 0 follow-ups | done | `693ea98` |
| 2. ggml model reader | done | `84f9152` |
| 3. Reference tensors | done | `48208ff` |
| 4. Element-wise kernels, lesson 1 | done | `ed30802` |
| 5. LayerNorm, lesson 2 | done | `4344d52` |
| 6. Softmax, lesson 3 | done | `0c885a1` |
| 7. GEMM lessons and attention, lessons 4-5 | done, tensor-core kernel deferred (see 7) | `d45ca6e`, `0947f50` |
| 8. conv1d, lesson 6 | done | `6ce1727` |
| 9. Log-mel, lesson 7 | done | `b6ad128` |
| 10. The encoder | done | `f714732` |
| Encoding repair (double-encoded comments, BOMs) | done | `2045375` |
| Final review of the branch | done: 3 Important fixed, 10 Minor deferred | `3e1cd12` |

## 4. The next steps, in order

1. **Ask the user to merge `rust-cuda-phase1a` to `main`.** Merge and push only when the user says so.
2. **Tensor-core task (not in a plan yet).** The user's main goal is learning CUDA, and tensor cores are central to it. Write a small plan for: (a) a `ptx_asm!` wrapper macro over the marker call, so inline PTX builds in the test profile, (b) `gemm_tc_f16` with `mma.sync.aligned.m16n8k8` (`sm_75` form) for lesson 4, (c) a tensor-core flash attention for lesson 5. Measure both against cuBLAS and the naive attention. The product keeps cuBLAS and `AttnKind::Naive` unless a measurement says otherwise.
3. **Write the Phase 1b plan.** The spec, Phase 1 section, lists what it must cover. Section 8 below lists the facts for it. It must also handle the deferred findings M1 and M6 (section 7).

## 5. Rulings made so far (decisions not in the plans)

Phase 0:

- Arena kernels take raw device pointers and a length. The synchronous cuda-oxide launch takes only a whole `&DeviceBuffer<T>` or raw pointers. An arena tensor is a part of one buffer.
- `Arena::slice` returns `DevSlice<'_, T>` (a typed device pointer and length), not a `DeviceBufferView`.
- `jeuvoice-gpu` has `[lib] test = false`, and all its tests are in `tests/`. cuda-oxide writes the PTX of a crate to one file per crate name. Cargo built the lib and the lib test at the same time, and one build failed with "embedded artifact payload is empty".
- `scripts\cargo.cmd oxide` sets `CUDA_OXIDE_PTX_DIR=target\oxide-ptx` and `CUDA_OXIDE_TARGET=sm_75`. The cuda-oxide default `sm_80` does not load on Turing (RTX 20) cards.
- `#![feature(f16)]` in `jeuvoice-gpu` for the built-in `f16` type in kernels.
- The no-GPU test uses `CUDA_VISIBLE_DEVICES=-1`. An empty value counts as "not set" on Windows.
- `gpu-bench` runs as a build, then the exe. `cargo oxide run` gives the words after `--` to the program, not to cargo.

Phase 1a:

- Task 4: `bias_gelu` has three widths. A `[f16; 2]` load compiles to two 2-byte loads, and it is the fastest (178.1 µs, 90%). A real 4-byte `u32` load was slower (190.8 µs, 84%) because of the pack and unpack instructions. The product uses `[f16; 2]`. The `u32` kernel stays for lesson 1 (`Width::TwoPacked`).
- Task 3: the `libm` dependency went into the Task 3 commit.
- Task 5: the three-pass LayerNorm measured 63% at `[1500, 1280]`, under the plan's 80% target. Two more kernels: `layernorm_regs_*` (row in registers, 78%) and `layernorm_regs_f16x2` (pairs, 84%). The product uses the pair kernel for F16, and `ops::layernorm_with(..., LayerNormKind)` falls back above 2048 columns.
- Task 6: the online softmax lost to the three-pass kernel (1.32 vs 1.08 ms): it costs one extra `exp` per element and has a serial chain, while the extra reads already hit L1. A register kernel won at 993 µs (94%) and is the product default.
- Task 6: `libm::expf` compiles to two `fma` plus `ex2.approx.ftz.f32`. `exp` is cheap on this card.
- Task 7: `gemm_lesson` takes the shapes from the tensors, not as `m`, `n`, `k` arguments.
- Task 7: the tensor-core GEMM is not written yet. `ptx_asm!` does not build in the test profile on rev `5e5c77b` (the template reaches the MIR importer as a temporary, `Move(_3)`), and the ready-made `mma` wrappers in `cuda_device` need `sm_80` shapes. A direct marker call does build. Lesson 4 uses cuBLAS as the tensor-core row. Details in `docs/development-log.md`.
- Task 7: attention uses `AttnKind::Naive` (2.48 ms). Flash attention measured 26.2 ms at `20 x 1500 x 64`, because two-warp blocks with 20 KB of shared memory reach about 20% occupancy and the math is scalar FMA. Raising the query block from 1 to 8 rows changed nothing: one head of K and V (384 KB) already fits in the 4 MB L2.

Phase 1a, Tasks 8-10 and the review (this agent):

- Task 8: the conv tests compare 2000 random output positions with the CPU (a full CPU conv2 is 7.4 billion multiply-adds). Im2col + cuBLAS is 21-46 times faster than the direct kernel. The encoder uses it.
- Task 9: the mel covers **the whole file**, as whisper.cpp does (the max for the clamp is over the whole file, not per 30 s window). `GpuMel::compute(samples)` makes the whole mel, and `window_into(start, arena, out)` copies one window into the encoder input.
- Task 9: the DFT and the mel projection run as **f32** GEMMs (`Blas::gemm_f32`). A later measurement showed that f16 frames would change only 20 of 768,000 values by more than 0.02, so the f16 risk was small. f32 stays, at little cost.
- Task 9: `GpuMel` works in chunks of 3000 frames, so one hour of audio needs about 194 MB, not 1.7 GB. `DevSlice::range`, the `first`/`count` arguments of `mel_frames`, and a second arena argument of `mel_window` came from this.
- Task 10: the encoder weights and activations share **one arena**, because each op takes one arena for all its tensors. The decoder in 1b can join it.
- Task 10: activations are **time first** (`[1500, 1280]`). `conv1d_with_layout` reads the mel channels first and writes time first through `colᵀ · wᵀ`, with no transpose kernel.
- Task 10: block 31 and `ln_post` differ from PyTorch fp32 for 0.5-2.4% of values. Two hypotheses were tested: fp16 attention scores (rejected: the f32-score flash kernel gave the same error) and fp16 activations (confirmed: PyTorch itself in fp16 gives the same share at the same rows). The test allows the PyTorch fp16 level with a margin. The table is in `crates/jeuvoice-whisper/tests/cuda_encoder.rs`.
- Review: the attention scale moved into the `Q·Kᵀ` GEMM alpha, so a raw score cannot pass the f16 limit. `GgmlModel::check_gpu_encoder` rejects models that are not large-v3 shaped. `merge_heads` checks its width.

## 6. Measurements so far (RTX 3050 Ti Laptop, 192 GB/s, 15.2 TFLOP/s fp16)

| Kernel | Size | Time | % of limit |
|---|---|---|---|
| cuBLAS `gemm_f16` | 1500x1280x1280 | 287-364 µs | 89-113% |
| cuBLAS `gemm_f16` | 4096x1024x1024 | 449-594 µs | 95-126% |
| `gemm_naive_f16` (lesson 4) | 1500x1280x1280 | 9.3-10.9 ms | 3% |
| `gemm_tiled_f16` (lesson 4) | 1500x1280x1280 | 6.7-7.2 ms | 4-5% |
| `gemm_regtile_f16` (lesson 4) | 1500x1280x1280 | 2.5 ms | 13% |
| `rmsnorm_naive_f32` | 4096x1024 | 938 µs | 19% |
| `rmsnorm_f32` | 4096x1024 | 200 µs | 88% |
| `rmsnorm_f16` | 4096x1024 | 99 µs | 89% |
| `bias_gelu` one per thread | 1500x5120 | 285 µs | 56% |
| `bias_gelu` `[f16; 2]` | 1500x5120 | 178 µs | 90% |
| `layernorm_3pass_f16` | 1500x1280 | 63.7 µs | 63% |
| `layernorm_f16_regs` | 1500x1280 | 51.1 µs | 78% |
| `layernorm_f16` (pairs) | 1500x1280 | 47.4 µs | 84% |
| `layernorm_f16` (pairs, 4x rows) | 6000x1280 | 175.5 µs | 91% |
| `layernorm_f32` | 1500x1280 | 89.6 µs | 89% |
| `softmax_3pass_f16` | 30000x1500 | 1.08 ms | 87% |
| `softmax_online_f16` | 30000x1500 | 1.32 ms | 71% |
| `softmax_regs_f16` | 30000x1500 | 993 µs | 94% |
| `attention_naive_f16` | 20x1500x64 | 2.48 ms | 31% |
| `attention_flash_f16` | 20x1500x64 | 26.2 ms | 3% |
| `conv1d_direct_f16` | 1280>1280, t3000, stride 2 | 30.8-43.6 ms | 2-3% |
| `conv1d_im2col_f16` | 1280>1280, t3000, stride 2 | 942 µs | 103% |
| `conv1d_im2col_f16` | 128>1280, t3000 | 247 µs | 79% |
| GPU mel, one 30 s window | upload, compute, download | 3.24 ms | — |
| CPU mel port (f64, plain DFT) | one 30 s window | 311.8 ms | — |
| **The encoder** | one 30 s window | **252 ms** | about 140 ms GEMM, 80 ms attention |
| Encoder memory | weights / activations | 1214 MB / 165 MB | — |

The % column for cuBLAS can pass 100. `DeviceInfo` reports 15.2 TFLOP/s fp16, and the card boosts above that. A first run on an idle GPU reads one step low. Run the table twice.

whisper.cpp baseline (the Phase 1b target): 69 s of Turkish audio in 3 files transcribes in 4.3 s.

## 7. Open issues

Deferred review findings (from the Phase 0 review):

- M5: a real out-of-memory at allocation returns `GpuError::Driver`, not `OutOfMemory`.
- M7: `scripts\cargo.cmd` picks the CUDA folder by name order (v13.4 before v13.10), drops `!` from PATH entries, and fails with a PATH near 8 KB.
- M8: `cargo test --workspace` and `cargo clippy --workspace` include `jeuvoice-gpu`, which builds only with `cargo oxide`.
- M10: `check::compare` ignores NaN in `max_rel` (`assert_close` still fails on NaN).

Deferred from the Phase 1a review (the full list is in the ledger):

- M1: mel frames past the end get the floor value. whisper.cpp writes 0 into the encoder input there. The 30 s tail makes this almost unreachable. Fix it in plan 1b with the window loop.
- M6: `Blas` holds a raw cuBLAS handle, so it is `!Send` and `!Sync`. `CudaWhisper` must satisfy `Transcriber: Send + Sync`: plan a `Mutex` with an `unsafe impl Send`, or create `Blas` on the worker thread.
- M7: the other-thread test covers only upload and download. Add a kernel launch and a GEMM.
- Other minors: two constant doc comments on the wrong item, one rustfmt line, `attention()` reads the shape before it checks the rank, the ggml reader allocates the filter buffer before the size check, lesson 7 has no card-limit part, `reference` makes `safetensors` a production dependency.
- The fp16 residual stream: if plan 1b misses the word error target, try an f32 residual stream first (whisper.cpp keeps activations in f32).
- The encoder margin on `ses_001_0007` at `ln_post` is thin: 2.37% against a 2.5% limit.

Found in Phase 1a, Task 7:

- `ptx_asm!` does not build in the test profile on the pinned cuda-oxide rev. `cargo oxide build` works, `cargo oxide test` fails with `ptx_asm template must be a byte string literal, got MIR operand Move(_3)`. A direct call to the marker function builds in both. The tensor-core GEMM waits for one of these: a wrapper macro over the marker call, a fix in cuda-oxide, or a patched fork. See `docs/development-log.md`, "Inline PTX (`ptx_asm!`) does not build in the test profile".
- `DeviceInfo::fp16_tensor_tflops` (15.2) is below what cuBLAS reaches (up to 19). The bench percent for a compute kernel can read over 100.

## 8. Facts for the Phase 1b plan

From the whisper.cpp source in the cargo cache (`%USERPROFILE%\.cargo\registry\src\index.crates.io-*\whisper-rs-sys-0.15.0\whisper.cpp\src\whisper.cpp`):

| Subject | Where | Fact |
|---|---|---|
| Defaults | `whisper_full_default_params`, line 5906 | `no_context=true`, temperature 0.0 with fallback steps of 0.2, `entropy_thold 2.4`, `logprob_thold -1.0`, `no_speech_thold 0.6`, `suppress_blank true`, `max_initial_ts 1.0`, `length_penalty -1` |
| The app's settings | `crates/jeuvoice-whisper/src/lib.rs` | greedy, `best_of: 1`, `token_timestamps: true`, DTW preset `LargeV3Turbo`, the language from the settings |
| Token rules | `whisper_process_logits`, line 6142 | suppress blank at the start, `<|notimestamps|>`, SOT, task, language and previous tokens. Timestamps come in pairs. Initial timestamp limit. |
| The window loop | `whisper_full_with_state`, line 6792 | 30 s windows, seek by the last timestamp, temperature fallback, no-speech check |
| Alignment heads | line 395 | large-v3-turbo: (2,4), (2,11), (3,3), (3,6), (3,11), (3,14) as (decoder layer, head) |
| DTW | `whisper_exp_compute_token_level_timestamps_dtw`, line 8802, `median_filter` 8767, `dtw_and_backtrace` 8677 | |
| Mel | `log_mel_spectrogram`, line 3170 | reflect pad 200 at the start, 200 + 480,000 zeros at the end, Hann 400, hop 160, log10, clamp to max − 8, then (v + 4) / 4 |

The app's word times use `t_dtw` when it is set, else `t0`/`t1` (centiseconds). `tokens_to_words` in `crates/jeuvoice-whisper/src/words.rs` turns tokens into words.

Success criteria for Phase 1 (from the spec, decided by the user): **Phase 1 is not done until all of them pass.** Until then `CudaWhisper` runs only with `JEUVOICE_CUDA_WHISPER=1`.

1. Word error rate vs whisper.cpp output: 2% or less on each test file.
2. Median word start-time difference: 50 ms or less.
3. The 69 s test set in 4.3 s or less.
4. Peak VRAM not more than whisper.cpp.
5. The existing tests and the `#[ignore]` real-model tests pass with `CudaWhisper`.
6. The app transcribes on the CPU (whisper-rs) when CUDA is not available.

## 9. Files and data on this machine

| What | Where |
|---|---|
| Whisper model (ggml, f16) | `%APPDATA%\Jeu\JeuVoice\data\models\ggml-large-v3-turbo.bin` |
| Test audio | `D:\MLProjelerim\tts_lab\data\ref\my_voice.ogg`, `my_voice2.mp4`, `sesim\ses_001_0007.wav` (27.0 + 33.6 + 8.4 = 69 s) |
| Reference tensors (not committed) | `target\reference\whisper\<clip>.safetensors` and `.f32`. Rebuild with `tools\reference\README.md` |
| Python for references only | `engine\.venv\Scripts\python.exe` (torch 2.14 cu126, transformers 5.19). Run it with `-I`. |
| cuda-rust source at the pinned commit | `D:\MLProjelerim\cuda-rust-spike\cuda-rust` (read it for API questions) |
| Toolchain test (throwaway) | `D:\MLProjelerim\cuda-rust-spike` |
| vcpkg with libffi | `D:\vcpkg` |

## 10. How the previous agent worked

- One plan task at a time: write the tests, run them and watch them fail, write the code, run all GPU tests, add the bench row, write the lesson, commit.
- A progress ledger lives in `.superpowers/sdd/<plan>/progress.md` (git ignores it). It has one line per finished task and one `Ruling:` line for each decision that is not in the plan. Keep it.
- When a measurement does not match the expectation, look at the PTX (`target\oxide-ptx\jeuvoice_gpu.ptx`) and measure again before you change the plan. Task 4 is an example: the "faster" kernel was slower.
- After the last task of a plan: a fresh reviewer reads the whole branch. Fix each Critical or Important finding with a test that fails first. List the Minor ones as deferred.
- Merge to `main` and push only when the user says so.
