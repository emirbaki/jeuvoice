# Handoff: Rust CUDA inference, Phase 1b (the Whisper decoder)

Date: 2026-11-25. Written by the previous agent for the next agent. Read `AGENTS.md` first. It has the rules and the commands. The older handoff (`2026-11-24-rust-cuda-handoff.md`) has the history of Phase 0 and Phase 1a. Its sections 2, 5 and 9 still apply.

## 1. Start here

1. Switch to the branch: `git switch rust-cuda-phase1b`. It is 30 commits ahead of `main`. It is **not merged and not pushed**. Merge and push only when the user says so.
2. Run the plain tests: `scripts\cargo.cmd test`. Expected: 407 pass, 0 fail.
3. Run the GPU kernel tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`. Expected: 93 pass.
4. Run the Whisper GPU tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels`. Expected: 70 pass (the rest are `#[ignore]`).
5. Run the real-model tests one file at a time, with `--release --test-threads 1` (the card has 4 GB):
   `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --release --test cuda_transcribe -- --ignored --nocapture --test-threads 1`
   and the same for `cuda_decoder` and `cuda_encoder`.
6. Read section 3. One criterion is open, and **only the user can decide about it**.

## 2. Where the work is

The plan: `docs/superpowers/plans/2026-11-25-rust-cuda-phase1b-decoder.md` (16 tasks). The ledger with one line per task and every ruling: `.superpowers/sdd/phase1b/progress.md` (git ignores it, it is on this machine). The measurements: `docs/development-log.md`, section "Status (Rust CUDA, Phase 1b)". The lessons are 0 to 9 in `docs/cuda-lessons/`; 9 is new.

| Task | Content | State |
|---|---|---|
| 0 to 12 | fixtures, mel fix, vocab, token rules, window loop, token times, DTW, embed and attention kernels, logits GEMM, GPU decoder, CudaWhisper, the comparison tool | done |
| 13 | the app path and the package | done, `8af4c63` |
| 14 | close the gaps (the f32 residual stream) | done, `278f3af` (criterion 1 open, see 3) |
| 15 | lesson 9, docs, the branch review | done: lesson `17c5f5e`, review fixes `ac897d8`, docs `7d33b53` |

## 3. The open criterion: 1, the WER against whisper.cpp

| Criterion | Limit | whisper.cpp | CudaWhisper | Pass |
|---|---|---|---|---|
| 1. WER on each file | 2% | — | worst 2.86% (my_voice2) | **no** |
| 2. Median word-start difference | 50 ms | — | 0 ms | yes |
| 3. Time of the 69 s set | 4.3 s | 3.954 s | 2.813 s | yes |
| 4. Peak VRAM (whole card, peak minus idle) | not more than whisper.cpp | 2157 MiB | 1798 MiB | yes |
| 5. The tests and the real-model tests with CudaWhisper | all pass | - | 70 + 4 real-model | yes |
| 6. The app transcribes on the CPU without CUDA | works | — | two tests (the fallback and the no-GPU load) | yes (no manual UI run) |

Criterion 1 is **one token** of my_voice2 window 1:

| | Token 80 | Probability |
|---|---|---|
| PyTorch f32 (transformers, teacher forced) | `uy` (7493) | 0.5448 |
| PyTorch f32, the second candidate | `u` (84) | 0.4528 |
| CudaWhisper | `uy` (7493) | 0.509 |
| whisper.cpp (the reference) | `u` (84) | 0.517 |

The model prefers `uy`. Our token is the model answer and whisper.cpp is the outlier at this step, because whisper.cpp converts its f32 activations to f16 and runs `cublasGemmEx` with half alpha and a half output, so it accumulates its matmuls in f16. Our matmuls accumulate in f32. Matching whisper.cpp would mean being less accurate.

The user was asked and decided: **go on with Task 13 first**. The decision about criterion 1 is still open. Do not change a criterion. Only the user can.

## 4. What is left

1. **The review is done** (`docs/development-log.md`, "The branch review"): no Critical, two Important fixed with tests, the minors listed as deferred. Read that list before changing the decoder.
2. **The merge.** Ask the user before `git push` / a merge to `main`.
3. **The Phase 1b handoff for Phase 2.** If the user accepts criterion 1 as "explained", Phase 1 is done and the next work is Phase 2 (the OmniVoice Qwen3-0.6B backbone): write its plan from the spec, Phase 2 section.
4. **Optional: the manual UI check of criterion 6.** Run the app with `CUDA_VISIBLE_DEVICES=-1` and `JEUVOICE_CUDA_WHISPER=1`, transcribe one file, and see the result come from whisper.cpp on the CPU.

## 5. The watch list

| Item | State |
|---|---|
| W1: the no-speech prob of whisper.cpp | **settled** (Task 0). whisper.cpp reads row 0 of its logits buffer, which holds the last 1-token decode or zeros. The window loop copies this. A test checks both cases. |
| W2: the fallback windows (our random draws differ from `std::mt19937`) | 0 fallbacks on all 4 windows of the test set. `CudaWhisper::last_fallback_windows()` and the report column show the count. |
| W3: the VRAM | 1798 MiB against whisper.cpp 2157 MiB. The f32 residual stream costs about 12 MiB: the decoder and encoder weigh more, the activations `x` are f32. |

## 6. How the code fits together

| Part | File | Note |
|---|---|---|
| The token rules, sampler, score | `crates/jeuvoice-whisper/src/decode/rules.rs`, `sample.rs` | ports of `whisper_process_logits`, `whisper_sample_token`, `whisper_sequence_score` |
| The window loop | `decode/window.rs` | `transcribe_tokens(model, vocab, samples, params, progress)`, the trait `WhisperModel` |
| The token times, DTW | `decode/token_times.rs`, `decode/dtw.rs` | called from the window loop. DTW is CPU code (lesson 9) |
| The vocab | `decode/vocab.rs` | ids for large-v3 (n_vocab 51866): EOT 50257, SOT 50258, BEG 50365 |
| The GPU decoder | `cuda/decoder.rs`, `weights.rs` | one arena; the residual stream `x` is f32, the norm weights are f32 |
| The GPU encoder | `cuda/encoder.rs` | `x` f32 from the second convolution on, norms and the positional embedding f32 |
| The GPU model | `cuda/model.rs` (`GpuWhisper`) | mel + encoder + decoder as the `WhisperModel` |
| The Transcriber | `cuda/transcriber.rs` (`CudaWhisper`) | one worker thread owns the GPU objects |
| The app path | `crates/jeuvoice-app/src/main.rs` | `LazyWhisper` holds `Arc<dyn Transcriber>`; `CudaWhisper` only with `JEUVOICE_CUDA_WHISPER=1` and the `cuda-kernels` feature |
| The package | `packaging/windows/collect.ps1` | builds with `cargo oxide build -- -p jeuvoice-app --release --features cuda-kernels`; the PTX is inside the exe |
| The comparison | `src/compare.rs`, `examples/compare_whisper.rs`, `scripts/compare-whisper.ps1` | WER with Levenshtein after the Turkish lower case |
| The token dumps | `examples/dump_whisper_rs.rs` (`GPU=0` for the CPU), `examples/dump_cuda_tokens.rs` | same JSON shape, so the two compare token by token |
| New kernels | `crates/jeuvoice-gpu/src/kernels/mod.rs` | `embed_*`, `attn_decode_*`, `layernorm_to_f16`, `add_f16_into_f32`, `add_f32_into_f32`, `f16_to_f32` |

## 7. Measurements (RTX 3050 Ti Laptop, 192 GB/s)

| What | Time | Note |
|---|---|---|
| one decoder step | 2.49 ms median | |
| the cross K/V of one window | 3.1 ms | |
| logits GEMM, 1 x 51866 x 1280 | 718 µs | 96% of the memory limit |
| DTW of one window, before / after lesson 9 | 77.2 / 9.9 ms | the median filter was 67 ms, now 3.7 ms |
| the 69 s set | 2.813 s | whisper.cpp 3.954 s, same run |
| peak VRAM | 1798 MiB | whisper.cpp 2157 MiB |

Agreement with the PyTorch f32 reference after the f32 residual stream: decoder step logits max 0.041 (my_voice), 0.028 (ses_001_0007); encoder ln_post max 1.28, block_15 max 0.0016.

## 8. Rulings of Phase 1b (the important ones)

The ledger has all of them. When the plan and the whisper.cpp source disagree, the source wins.

- `no_context = true` clears only the context of an earlier `whisper_full` call. Inside one call the prompt carries the tokens of the earlier windows.
- The random generator is seeded once per transcription, not once per window.
- The token-level VAD reads the energy of the whole file at indexes relative to the segment start (a whisper.cpp quirk). The port keeps it.
- The DTW step rule sends a tie to the left. A window with 7 or fewer audio positions skips DTW.
- The decoder reference uses only the clips under 30 s, because the Hugging Face feature extractor normalizes each window and whisper.cpp the whole file.
- The median filter of lesson 9 uses a sorting network for width 7 and handles the reflect padding at the row ends. The output is identical to a plain sort (a test holds it).
- **The residual stream of both halves is f32**, the norm weights and the encoder positional embedding too. It made both halves closer to the f32 reference and did not change the criterion-1 token.
- `LazyWhisper` holds `Arc<dyn Transcriber>`. The `cuda-kernels` app feature and `JEUVOICE_CUDA_WHISPER=1` decide the path, and every load error falls back to whisper.cpp.
- A UTF-8 file read and written with PowerShell `Get-Content` + `Set-Content` turns Turkish letters and `·` into mojibake, and `Set-Content -Encoding utf8` adds a BOM. Use the edit tool, or `[System.IO.File]::WriteAllText` with `UTF8Encoding($false)`, then check the bytes.

## 9. Open issues

- Criterion 1, section 3. The user's decision.
- From Phase 1a: `ptx_asm!` does not build in the test profile on the pinned cuda-oxide rev, so the tensor-core lesson kernel (`mma.sync.m16n8k8`) is deferred. Lesson 4 uses cuBLAS as the tensor-core row. Details in `docs/development-log.md`.
- Deferred Phase 0 review minors: M5 (a real out-of-memory comes back as a driver error), M7 (`cargo.cmd` limits), M8 (`--workspace` commands include `jeuvoice-gpu`), M10 (`compare` ignores NaN in `max_rel`).
- `DeviceInfo::fp16_tensor_tflops` (15.2) is below what cuBLAS reaches (up to 19), so a bench percent can read over 100.

## 10. Files and data on this machine

The table of the older handoff, section 9, still applies. New in Phase 1b, all in `target\reference\whisper\` (not committed):

| File | Made by |
|---|---|
| `<clip>.rs.json`: the whisper-rs segments and tokens | `scripts\cargo.cmd run -p jeuvoice-whisper --release --example dump_whisper_rs -- <audio> <out.json>` (`GPU=0` for whisper.cpp on the CPU) |
| `<clip>.cuda.json`: our tokens with the probabilities | `dump_cuda_tokens` (needs the GPT kernels feature) |
| `<clip>.decoder.safetensors`: decoder logits and alignment softmax | `tools\reference\whisper_decoder_reference.py` |
| `my_voice2.win1.rs.json`: the first five segments only, so the reference tool accepts the clip | made by hand from `my_voice2.rs.json` |
| `target\compare\rs.json`, `cuda.json` | `scripts\compare-whisper.ps1` |

## 11. How the previous agent worked

- One plan task at a time: write the tests, run them and watch them fail, write the code, run the tests of **both** `jeuvoice-gpu` and `jeuvoice-whisper`, add the bench row, write the lesson, commit.
- Read the whisper.cpp source before each port, and check the port line by line after it. Record each difference as a ruling.
- Measure before a change. Two examples that paid off: the tie diagnosis (the reference, not us, is off) and the median filter (77 ms to 10 ms per window).
- After each task, report to the user: what works, the measurements, the state of W1, W2 and W3. Then ask before the next large step.
- All text in STE style (`AGENTS.md`, rule 8). The user reads English replies. The lessons are Turkish.
