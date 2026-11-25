# Handoff: Rust CUDA inference, Phase 1b (the Whisper decoder)

Date: 2026-11-25. Written by the previous agent for the next agent. Read `AGENTS.md` first. It has the rules and the commands. The older handoff (`2026-11-24-rust-cuda-handoff.md`) has the history of Phase 0 and Phase 1a. Its sections 2, 5 and 9 still apply.

## 1. Start here

1. Switch to the branch: `git switch rust-cuda-phase1b`. It has 16 commits on top of `main`. It is **not merged and not pushed**. Merge and push only when the user says so.
2. Run the plain tests: `scripts\cargo.cmd test`. Expected: 398 pass, 0 fail.
3. Run the GPU kernel tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`. Expected: 89 pass.
4. Run the Whisper GPU tests: `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels`. Expected: 67 pass (the rest are `#[ignore]`).
5. Run the real-model tests one file at a time, with `--release --test-threads 1` (the card has 4 GB). Example: `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels --release --test cuda_transcribe -- --ignored --nocapture --test-threads 1`.
6. Read section 3. **The next step is a decision of the user.**

## 2. Where the work is

The plan: `docs/superpowers/plans/2026-11-25-rust-cuda-phase1b-decoder.md` (16 tasks, 0 to 15). The ledger with one line per task and every ruling: `.superpowers/sdd/phase1b/progress.md` (git ignores it, it is on this machine). The measurements: `docs/development-log.md`, section "Status (Rust CUDA, Phase 1b)".

| Task | Content | State | Commit |
|---|---|---|---|
| 0 | whisper-rs fixtures, baseline | done | `2e53205` |
| 1 | mel frames past the end are 0.0 (M1) | done | `04575e2`, test fix `a692994` |
| 2 | vocab, special tokens, language table | done | `4c373ab` |
| 3 | token rules, sampler, sequence score | done | `b105d1c`, plan fix `75db921` |
| 4 | window loop (fallback, past prompt, no-speech row) | done | `f2ab44b` |
| 5 | token-level times (t0, t1) | done | `b449402` |
| 6 | DTW | done | `5b5e945` |
| 7 | `Tensor::rows`, embedding kernels, lesson 8 part A | done | `c4001c4` |
| 8 | decoder attention (flash decoding), lesson 8 part B | done | `40c180a` |
| 9 | f16 GEMM with an f32 output (logits) | done | `0388240` |
| 10 | GPU decoder, `GpuWhisper`, decoder reference tool | done | `c371005` |
| 11 | `CudaWhisper` (worker thread, Send + Sync, M6, M7) | done | `3cf1ab1` |
| 12 | measurement tool and the first criteria table | done | `1eadd31` |
| 13 | the app path and the package | **not started** | |
| 14 | close the gaps | **not started**: criterion 1 fails | |
| 15 | lesson 9 (DTW), docs, the branch review | **not started** | |

## 3. The open decision: criterion 1 fails on one near tie

The first measurement (`powershell -ExecutionPolicy Bypass -File scripts\compare-whisper.ps1`):

| Criterion | Limit | whisper.cpp | CudaWhisper | Pass |
|---|---|---|---|---|
| 1. WER on each file | 2% | — | worst 2.86% (my_voice2) | **no** |
| 2. Median word-start difference | 50 ms | — | 0 ms | yes |
| 3. Time of the 69 s set | 4.3 s | 3.861 s | 2.820 s | yes |
| 4. Peak VRAM (whole card, peak minus idle) | not more than whisper.cpp | 2147 MiB | 1787 MiB | yes |
| 5. The tests with CudaWhisper | all pass | | needs Task 13 | — |
| 6. CPU fallback in the app | works | | needs Task 13 | — |

The user decided: **Phase 1 is not done until all six pass.** Do not change a criterion. Only the user can do that.

The cause of the WER failure: my_voice2 has one near tie at token 80 of window 1. whisper-rs takes "u" (id 84, p 0.517), CudaWhisper takes "uy" (id 7493, p 0.511). The next token joins the two sequences again ("uyla" against "u ile"). That gives 1 substitution and 1 deletion of 70 words: 2.86%. The other two clips have 0% WER. All other 377 of 379 tokens are equal.

The previous agent proposed two ways on, and the user had not chosen yet:

1. **Diagnose the tie first** (recommended, about 1 to 2 hours):
   1. Run whisper.cpp on the CPU (`WhisperRs::load(&model, false)`) on my_voice2. Compare its tokens with its own GPU fixture `target\reference\whisper\my_voice2.rs.json`. If the CPU and the GPU of whisper.cpp also differ at this tie, the tie is rounding noise.
   2. Try an f32 residual stream in the decoder (4 blocks: `x` in f32, the LayerNorm reads f32, the residual add writes f32). Measure if the tie goes to "u", and the cost in time and VRAM. The encoder is the other place for an f32 residual stream (32 blocks, more cost).
   3. Show the result to the user before a large change (plan Task 14, step 2).
2. **Task 13 first** (the app path and the package), then Task 14.

Ask the user which one. Write the answer in the ledger.

## 4. The remaining tasks

The plan has the steps. Short notes from the work so far:

- **Task 13:** `LazyWhisper` in `crates/jeuvoice-app/src/main.rs` holds `Arc<WhisperRs>` today. Make it hold `Arc<dyn Transcriber>` and an "on GPU" flag. Load `CudaWhisper` only with `JEUVOICE_CUDA_WHISPER=1`, and use `WhisperRs` when `CudaWhisper::load` fails (every load failure is `WhisperError::ModelLoad`). Add the app feature `cuda-kernels = ["jeuvoice-whisper/cuda-kernels"]`. `packaging/windows/collect.ps1` builds the release exe at line 75: switch it to `scripts\cargo.cmd oxide build -- -p jeuvoice-app --release --features cuda-kernels`. Do not change the app UI (hard rule 11).
- **Task 14:** see section 3. Also run `scripts\compare-whisper.ps1` again after each change.
- **Task 15:** lesson 9 (DTW, Turkish, 7 parts). The DTW is CPU code in `crates/jeuvoice-whisper/src/decode/dtw.rs`. Measure its CPU time per window for the lesson. Then a fresh reviewer reads the whole branch.

## 5. The watch list (the user asked to keep these in view)

| Item | State |
|---|---|
| W1: the no-speech prob of whisper.cpp | **settled** (Task 0). whisper.cpp reads row 0 of its logits buffer, which holds the last 1-token decode or zeros. The window loop copies this. A test checks both cases. |
| W2: the fallback windows (our random draws differ from `std::mt19937`) | 0 fallbacks on all 4 windows of the test set, for both whisper.cpp (indirect evidence) and CudaWhisper. `CudaWhisper::last_fallback_windows()` and the report column show the count. |
| W3: the VRAM | 1787 MiB against 2147 MiB for whisper.cpp. The decoder runs the multi-row attention in groups of 32 rows (scratch 4 MB, not 29 MB), and the logits are only the last row. |

## 6. How the code fits together

| Part | File | Note |
|---|---|---|
| The token rules, sampler, score | `crates/jeuvoice-whisper/src/decode/rules.rs`, `sample.rs` | ports of `whisper_process_logits` (line 6142), `whisper_sample_token`, `whisper_sequence_score` |
| The window loop | `decode/window.rs` | `transcribe_tokens(model, vocab, samples, params, progress)`, the trait `WhisperModel` |
| The token times, DTW | `decode/token_times.rs`, `decode/dtw.rs` | called from the window loop after each segment and each window |
| The vocab | `decode/vocab.rs` | ids for large-v3 (n_vocab 51866): EOT 50257, SOT 50258, BEG 50365 |
| The GPU decoder | `crates/jeuvoice-whisper/src/cuda/decoder.rs`, `weights.rs` | one arena: weights 328 MB, caches and activations 53 MB |
| The GPU model | `cuda/model.rs` (`GpuWhisper`) | mel + encoder + decoder as the `WhisperModel` |
| The Transcriber | `cuda/transcriber.rs` (`CudaWhisper`) | one worker thread owns the GPU objects |
| The comparison | `src/compare.rs`, `examples/compare_whisper.rs`, `scripts/compare-whisper.ps1` | WER with Levenshtein after the Turkish lower case |
| New kernels | `crates/jeuvoice-gpu/src/kernels/mod.rs` | `embed_f16`, `embed_f16x2`, `embed_one_f16x2`, `attn_decode_f16`, `attn_decode_split_f16`, `attn_decode_merge_f16` |

## 7. Measurements of Phase 1b (RTX 3050 Ti Laptop, 192 GB/s)

| What | Time | Note |
|---|---|---|
| `embed_one` (one decoder step) | 9.0 µs | `embed_x2` with the ids upload: 17.6 µs |
| decoder attention, 1 x 1500 (cross step) | 59.1 µs split, 175.5 µs naive | 68% of the memory limit |
| decoder attention, 1 x 448 (self step) | 22.9 µs split, 56.2 µs naive | |
| DTW pass attention, 224 x 1500 | 3.47 ms split, 12.6 ms naive | scalar math, not memory-bound |
| logits GEMM, 1 x 51866 x 1280 | 718 µs | 96% of the memory limit |
| one decoder step | 2.49 ms median | |
| the cross K/V of one window | 3.1 ms | |
| the 69 s set | 2.820 s | whisper.cpp 3.861 s |

## 8. Rulings of Phase 1b (the important ones)

The ledger has all of them. When the plan and the whisper.cpp source disagree, the source wins.

- `no_context = true` clears only the context of an earlier `whisper_full` call. Inside one call, the prompt carries the tokens of the earlier windows at temperatures below 0.5 (`[PREV, last 223 past tokens, SOT, lang, transcribe]`). The plan said "no past tokens". The plan is corrected.
- The random generator is seeded once per transcription, not once per window.
- The timestamp-mass rule masks every id below BEG (EOT too), and it does not normalize the logprobs again.
- The token-level VAD reads the energy of the whole file at indexes relative to the segment start (a whisper.cpp quirk). The port keeps it. All 379 token times equal whisper-rs.
- The DTW step rule sends a tie to the left step. A window with 7 or fewer audio positions skips DTW (whisper.cpp asserts there).
- `Tensor::rows` panics on a row that does not start on the 256-byte arena alignment. Every Whisper row of 1280 f16 values is aligned.
- The decoder reference uses only the clips under 30 s, because the Hugging Face feature extractor normalizes each window and whisper.cpp the whole file.
- The decoder test limits are 2 times the error of PyTorch fp16 against fp32 (the table is in `tests/cuda_decoder.rs`).

## 9. Files and data on this machine

The table of the older handoff, section 9, still applies. New in Phase 1b, all in `target\reference\whisper\` (not committed):

| File | Made by |
|---|---|
| `<clip>.rs.json`: the whisper-rs segments and tokens | `scripts\cargo.cmd run -p jeuvoice-whisper --release --example dump_whisper_rs -- <audio> <out.json>` |
| `<clip>.decoder.safetensors`: decoder logits and alignment softmax (my_voice, ses_001_0007 only) | `tools\reference\whisper_decoder_reference.py`, see `tools\reference\README.md` |
| `target\compare\rs.json`, `cuda.json` | `scripts\compare-whisper.ps1` |

## 10. How the previous agent worked

- One plan task at a time: write the tests, run them and watch them fail, write the code, run the tests of **both** `jeuvoice-gpu` and `jeuvoice-whisper`, add the bench row, write the lesson, commit. Two misses are in the ledger: a Task 1 test in the other crate, and `Tensor::prefix_as` with no failing run first.
- Read the whisper.cpp source before each port, and check the port line by line after it. Record each difference as a ruling.
- Trace a test by hand before you trust its expected value. Three test values were wrong in Tasks 3, 4 and 6, and the code was right.
- After each task, report to the user: what works, the measurements, the state of W1, W2 and W3. Then ask before the next task.
- All text in STE style (`AGENTS.md`, rule 8). The user reads English replies. The lessons are Turkish.
