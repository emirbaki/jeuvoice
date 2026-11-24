# Handoff: Rust CUDA inference (Whisper and OmniVoice in Rust kernels)

Date: 2026-11-24. Written by the previous agent for the next agent. Read `AGENTS.md` at the repo root first. It has the rules and the commands.

## 1. Start here

1. Check out the branch: `git switch rust-cuda-phase1a`. The last commit is `4fa9b03 feat(gpu): add the two-pass LayerNorm kernels`.
2. Run the GPU tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`. Expected: 34 tests pass.
3. Run the plain tests: `scripts\cargo.cmd test`. Expected: 340 or more tests pass, with no failures.
4. Continue with **plan 1a, Task 5, Step 3** (section 4 below).

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
| 1a. Whisper encoder | plan `docs/superpowers/plans/2026-11-24-rust-cuda-phase1a-encoder.md`, 10 tasks | **in progress** on `rust-cuda-phase1a`, Tasks 1–4 done, Task 5 half done |
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
| 5. LayerNorm, lesson 2 | **steps 1–2 done** (kernels and tests pass). Steps 3–5 open | `4fa9b03` |
| 6. Softmax, lesson 3 | open | — |
| 7. GEMM lesson kernels and attention, lessons 4–5 | open | — |
| 8. conv1d, lesson 6 | open | — |
| 9. Log-mel, lesson 7 | open | — |
| 10. The encoder | open | — |
| Final review of the branch | open | — |

## 4. The next steps, in order

1. **Task 5, Step 3:** add a `layernorm_f16 1500x1280` row to `crates/jeuvoice-gpu/src/bin/gpu-bench.rs`. Copy the pattern of `bench_rmsnorm`. Bytes = read x + write y. Target: 80% or more of the memory limit.
2. **Task 5, Step 4:** write lesson 2, `docs/cuda-lessons/02-layernorm.md`, in Turkish. Use the 7 parts of lesson 0 (see `AGENTS.md`). Explain why the two-pass form is needed (`layernorm_large_offset` test). Add it to `docs/cuda-lessons/README.md`.
3. **Task 5, Step 5:** commit `feat(gpu): add LayerNorm and lesson 2`.
4. **Tasks 6–10** as the plan says. Each task: tests first (watch them fail), then code, then the bench row, then the lesson, then one commit.
5. **Final review** of the branch, then fix the Critical and Important findings with a test each, then merge to `main` only when the user says so.
6. **Write the Phase 1b plan.** The spec, Phase 1 section, lists what it must cover. Section 8 below lists the facts for it.

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

## 6. Measurements so far (RTX 3050 Ti Laptop, 192 GB/s, 15.2 TFLOP/s fp16)

| Kernel | Size | Time | % of limit |
|---|---|---|---|
| cuBLAS `gemm_f16` | 1500x1280x1280 | 364 µs | 89% |
| cuBLAS `gemm_f16` | 4096x1024x1024 | 611 µs | 92% |
| `rmsnorm_naive_f32` | 4096x1024 | 938 µs | 19% |
| `rmsnorm_f32` | 4096x1024 | 200 µs | 88% |
| `rmsnorm_f16` | 4096x1024 | 99 µs | 89% |
| `bias_gelu` one per thread | 1500x5120 | 285 µs | 56% |
| `bias_gelu` `[f16; 2]` | 1500x5120 | 178 µs | 90% |

whisper.cpp baseline (the Phase 1b target): 69 s of Turkish audio in 3 files transcribes in 4.3 s.

## 7. Open issues

Deferred review findings (from the Phase 0 review):

- M5: a real out-of-memory at allocation returns `GpuError::Driver`, not `OutOfMemory`.
- M7: `scripts\cargo.cmd` picks the CUDA folder by name order (v13.4 before v13.10), drops `!` from PATH entries, and fails with a PATH near 8 KB.
- M8: `cargo test --workspace` and `cargo clippy --workspace` include `jeuvoice-gpu`, which builds only with `cargo oxide`.
- M10: `check::compare` ignores NaN in `max_rel` (`assert_close` still fails on NaN).

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
