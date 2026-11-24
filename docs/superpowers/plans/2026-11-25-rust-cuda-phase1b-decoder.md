# Phase 1b: Whisper decoder, decoding, DTW and the app path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `CudaWhisper`, a `Transcriber` on the Rust kernels that gives the same words and word times as whisper.cpp, runs in the app behind `JEUVOICE_CUDA_WHISPER=1`, and passes the six Phase 1 success criteria.

**Architecture:**

- The GPU does the numbers: the encoder (Phase 1a), the decoder (4 layers, a self-attention KV cache, a cross-attention K/V per window) and the logits.
- The CPU does the decisions, as whisper.cpp does: the token rules, the sampling, the window loop with the temperature fallback, the segments, the token times and DTW.
- The CPU parts are pure Rust in `crates/jeuvoice-whisper/src/decode/`. They compile without the `cuda-kernels` feature, so `scripts\cargo.cmd test` tests them with no GPU.
- The window loop talks to the model through a trait. A fake model with scripted logits tests the loop.
- `CudaWhisper` owns one worker thread. That thread owns the `Gpu`, the `Blas` (which is `!Send`) and the model. This gives `Send + Sync` (deferred minor M6).

**Tech stack:** Rust 1.99.1, cuda-oxide at the pinned rev, cuBLAS 13, the `jeuvoice-gpu` ops, the whisper.cpp source in the cargo cache as the behavior reference.

**Spec:** `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md`. **Handoff:** `docs/superpowers/handoff/2026-11-24-rust-cuda-handoff.md`, sections 7 and 8.

## Global Constraints

- All rules of `AGENTS.md` apply. The most important: Rust kernels only, device code only in `jeuvoice-gpu`, raw device pointers and a `SAFETY` comment on each `unsafe` block, PTX `sm_75`, fp16 data with f32 accumulate, tests first, a Turkish lesson for each kernel, a `gpu-bench` row for each kernel, English comments in STE style, UTF-8 without a BOM.
- Branch: `rust-cuda-phase1b`. Commit after each task. Keep the ledger in `.superpowers/sdd/phase1b/progress.md`.
- Do not change the app UI (hard rule 11). The app change in Task 13 is in the model loader only.
- whisper-rs stays: it is the fallback and the baseline. Do not remove it.
- "Same as whisper.cpp" means the same as the whisper.cpp source in `whisper-rs-sys-0.15.0`. When this plan and that source do not agree, the source wins. Record the case as a `Ruling:` line in the ledger.

## The whisper.cpp facts this plan uses

Source: `%USERPROFILE%\.cargo\registry\src\index.crates.io-1949cf8c6b5b557f\whisper-rs-sys-0.15.0\whisper.cpp\src\whisper.cpp`. Line numbers are for that file.

**Special tokens** (line 439, shifted at line 1622 for multilingual models). For large-v3-turbo, `n_vocab = 51866`, so `num_languages = 51866 − 51765 − 1 = 100` and the shift is `dt = 100 − 98 = 2`:

| Token | Id |
|---|---|
| EOT | 50257 |
| SOT | 50258 |
| first language (`en`) | 50259. The language with index `i` is `SOT + 1 + i`. |
| translate / transcribe | 50359 / 50360 |
| SOLM / PREV / NOSP / NOT (no timestamps) | 50361 / 50362 / 50363 / 50364 |
| BEG (timestamp 0.00) | 50365. Timestamp token `BEG + k` is `k × 0.02 s`. |

The ggml file stores fewer words than `n_vocab`. whisper.cpp names the extra ids `[_EOT_]`, `[_SOT_]`, `[_TT_k]` and so on (line 1647). `tokens_to_words` skips text that starts with `[_`.

**The app's parameters** (`crates/jeuvoice-whisper/src/lib.rs` and `whisper_full_default_params`, line 5906): greedy, `best_of 1`, `token_timestamps true`, `no_context true`, `temperature 0.0` with steps of `0.2` up to `1.0`, `entropy_thold 2.4`, `logprob_thold −1.0`, `no_speech_thold 0.6`, `suppress_blank true`, `max_initial_ts 1.0`, `length_penalty −1`, `thold_pt 0.01`, `thold_ptsum 0.01`, `max_len 0`, `suppress_nst false`, DTW preset large-v3-turbo with `medfilt_width 7`.

**The prompt** (line 6955): `[SOT, lang, transcribe]`. `no_context` is true, so there are no past tokens. The NOT token is not in the prompt, because timestamps are on.

**The token rules** (`whisper_process_logits`, line 6142), in this order:

1. Divide the logits by the temperature when it is above 0.
2. At the first sampled token (`suppress_blank`): set EOT and the token for `" "` to −∞.
3. Set NOT, SOT, NOSP, SOLM (no tinydiarize), translate, transcribe, PREV and all language tokens to −∞.
4. Timestamp pairs. Let `last` be the last sampled token and `penult` the one before. If `last` is a timestamp: if `penult` is also a timestamp, or there is only one sampled token, set all timestamps to −∞. Else set all text tokens below EOT to −∞.
5. Initial timestamp limit: at the first sampled token, set the timestamps from `BEG + round(1.0 / 0.02) + 1` up to −∞.
6. Increasing timestamps: when the decoder has had a timestamp (`has_ts`), set `BEG .. BEG + seek_delta / 2` to −∞.
7. Log-softmax. If `logsumexp(timestamp logprobs) > max(logprobs below BEG)`, set the logits and the logprobs of all ids below BEG to −∞ (EOT and the special tokens too). The logprobs are not normalized again.
8. Then the probs and logprobs are computed again from the result.

Read the full function before Task 3. These notes are a summary, not the source.

**Sampling** (`whisper_sample_token`, line 6425): `tid` is the best timestamp token, `pt = max_ts / (sum_ts + 1e−10)`, `ptsum = sum_ts`. At temperature 0 the token is the arg max of the probs. Above 0 it is a random draw from the probs. When the token is a timestamp, `tid = id` and `pt = p`.

**The sequence score** (`whisper_sequence_score`, line 6560): `sum_logprobs` and `avg_logprobs` over `result_len` tokens. The entropy uses the counts of the last 32 token ids.

**The window loop** (`whisper_full_with_state`, line 6792): times are in centiseconds ("frames" of 10 ms).

- `seek` starts at 0. `seek_end` is the mel length of the original audio. `delta_min = 10`.
- For each window: encode at `seek`. For each temperature: decode at most `n_text_ctx / 2 − 4 = 220` steps.
- The no-speech prob comes from the first decode of the prompt, before the token rules (line 7154). See Task 0, step 4: whisper.cpp reads row 0 of the logits buffer there, and only the last row is filled.
- A timestamp token `BEG + k` gives `seek_delta_new = 2k`. If `has_ts` and `seek_delta > seek_delta_new` and `result_len < i`, the decoder fails. Else `seek_delta = seek_delta_new` and `result_len = i + 1`.
- EOT, or `has_ts && seek + seek_delta + delta_min >= seek_end`, ends the window. With `result_len == 0`: at the end of the audio `result_len = i + 1`, else the decoder fails.
- At step 219, if `result_len == 0` or `seek_delta < 1500`, the decoder fails (repetition loop).
- After the steps: if `result_len > 32` and `entropy < 2.4`, the decoder fails.
- The last temperature is always kept. At each other temperature, the loop moves to the next temperature if the decoder failed, or if `avg_logprobs < −1.0 && no_speech_prob < 0.6`.
- `is_no_speech = no_speech_prob > 0.6 && avg_logprobs < −1.0`. A no-speech window gives no segments.
- Segments split at timestamp tokens (line 7605). A segment runs from `t0 = seek + 2 × (first.tid − BEG)` to the next timestamp. The last open text gets `t1 = seek + seek_delta`.
- `single_timestamp_ending`: if the last two tokens are a text token and then a timestamp, `seek_delta = min(seek_end − seek, 3000)`.
- Then `seek += seek_delta`.
- Temperatures above 0 sample with `std::mt19937`. Our RNG gives other draws, so a fallback window can give other text. This is the only expected difference. Record how many windows use a fallback in the measurements.

**Language detection** (`whisper_lang_auto_detect_with_state`, line 4021): encode the first window, decode `[SOT]`, softmax over the 100 language logits, take the best. The language table `g_lang` is near the top of the file.

**Token times without DTW** (`whisper_exp_compute_token_level_timestamps`, line 8420, with `get_signal_energy`, line 8390): this gives `t0` and `t1` for each token from `pt`, `ptsum` and the signal energy. The app uses `t1` for the word end.

**DTW** (`whisper_exp_compute_token_level_timestamps_dtw`, line 8802, `median_filter` line 8767, `dtw_and_backtrace` line 8677):

1. Tokens: `[SOT, lang, NOT]`, then the text tokens (`id < EOT`) of all new segments of the window, then EOT.
2. Run the decoder on these tokens in one batch, with a new self-attention cache and the window's cross K/V. Keep the cross-attention softmax of the alignment heads (decoder layer, head): (2,4), (2,11), (3,3), (3,6), (3,11), (3,14).
3. Keep the first `n_frames / 2` audio positions, with `n_frames = min(3000, seek_delta, seek_end − seek)`.
4. For each head and each audio position, normalize over the tokens: `(x − mean) / sqrt(var + 1e−9)`, with the population variance.
5. Median filter of width 7 over the audio positions, with reflect padding.
6. The mean over the heads, times −1.
7. Drop the first 2 token rows (SOT, lang) and the last row (EOT). The NOT row stays as row 0.
8. DTW and backtrace on `[n_tokens, n_audio]`. The ties go to "left" (`t = 2`) as in the source.
9. Walk the path: each time the token index changes (it starts at 0, so the NOT row gets no time), give the next text token `t_dtw = 2 × audio_index + seek`.

**The app's word times** (`lib.rs`): start `= t_dtw` when it is set, else `t0`. End `= max(t1, t_dtw)`. The word prob is the lowest token `p` in the word.

---

### Task 0: Branch, baseline and whisper-rs fixtures

**Files:** `crates/jeuvoice-whisper/examples/dump_whisper_rs.rs` (new), `docs/development-log.md`.

- [ ] **Step 1:** Make the branch `rust-cuda-phase1b` from `main`.
- [ ] **Step 2:** Write the example `dump_whisper_rs`. It runs `WhisperRs` with the app's parameters on a clip and writes `target/reference/whisper/<clip>.rs.json`: the language, and for each segment `t0`, `t1`, the no-speech prob and the tokens with `id`, `tid`, `p`, `plog`, `pt`, `ptsum`, `t0`, `t1`, `t_dtw`. Add the window count and the seek values if whisper-rs exposes them. Else leave them out.
- [ ] **Step 3:** Run it on the 3 test clips (handoff, section 9). The later tests read these files.
- [ ] **Step 4: The no-speech prob question.** whisper.cpp line 7154 reads `state->logits` from index 0, but `whisper_decode_internal` copies only the rows with `batch.logits[i] = 1` (the last prompt row). Find out the value that whisper.cpp really uses: print the segment no-speech probs from step 3. If they are near `1 / 51866` or equal to an old row, whisper.cpp does not use the SOT row. Record the ruling: copy what whisper.cpp does, because criterion 1 compares with whisper.cpp.
- [ ] **Step 5: Baseline.** Measure whisper-rs on the 69 s set: the time without the model load (warm, the median of 3 runs), and the peak VRAM of the process with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv -lms 100`. Write both in the development log. These numbers are the targets of criteria 3 and 4.
- [ ] **Step 6: Commit** `chore(whisper): dump whisper-rs fixtures for phase 1b`.

### Task 1: Mel frames past the end (deferred minor M1)

**Files:** `crates/jeuvoice-whisper/src/cuda/mel.rs` or `crates/jeuvoice-gpu/src/kernels/mod.rs` (`mel_window_f16`), `crates/jeuvoice-whisper/src/mel_cpu.rs`, `crates/jeuvoice-whisper/tests/cuda_mel.rs`.

- [ ] **Step 1: Failing test:** a window that starts 1000 frames before the end of the mel. The frames past the end must be what whisper.cpp gives (read `log_mel_spectrogram` and `whisper_encode_internal` first to confirm the value: the handoff says 0). Test the GPU window and the CPU oracle.
- [ ] **Step 2: Implement** the fix. Run the mel and encoder tests.
- [ ] **Step 3: Commit** `fix(whisper): fill mel frames past the end as whisper.cpp does`.

### Task 2: Vocab, special tokens and the language table

**Files:** `crates/jeuvoice-whisper/src/decode/mod.rs`, `decode/vocab.rs` (new), `crates/jeuvoice-whisper/src/lib.rs` (`pub mod decode`), `crates/jeuvoice-whisper/tests/decode_vocab.rs`.

- [ ] **Step 1: Failing tests:** `Vocab::from_ggml(&GgmlModel)` gives the ids of the table above for `n_vocab = 51866`. The text of id 50365 is `[_BEG_]` and of 50366 is `[_TT_1]` (as whisper.cpp names them). `lang_id("tr")` and `lang_code(i)` agree with whisper.cpp's `g_lang`. A model with `n_vocab = 51864` (English only) gives an error in plain words: the GPU path supports multilingual large-v3 models only.
- [ ] **Step 2: Implement.** Copy `g_lang` as a `const` table of `(code, index)`. Keep the token text as bytes. Give `token_text_lossy(id)` with the same lossy conversion as whisper-rs `to_str_lossy`, so the words match.
- [ ] **Step 3: Commit** `feat(whisper): add the vocab and the special tokens`.

### Task 3: The token rules and the sampler

**Files:** `decode/rules.rs`, `decode/sample.rs` (new), `crates/jeuvoice-whisper/tests/decode_rules.rs`.

- [ ] **Step 1: Failing tests,** one for each rule of the list above, on small made-up logit vectors of the real `n_vocab`:
  - suppress blank only at the first token
  - the special and language tokens are always −∞
  - the timestamp pairs (the 3 cases: one token, two timestamps, text then timestamp)
  - the initial limit (`BEG + 51` is the first masked id)
  - increasing timestamps with `seek_delta`
  - the timestamp-mass rule
  - the temperature division
  - `TokenData` from the sampler: `id`, `p`, `plog`, `tid`, `pt`, `ptsum` for a hand case, and the timestamp case `tid = id`
  - `SequenceScore`: `avg_logprobs` and the entropy of the last 32 tokens on a hand case
- [ ] **Step 2: Implement** the port of `whisper_process_logits`, `whisper_sample_token` and `whisper_sequence_score`. The math is in f32 as in the source, with `f64` sums where the source uses `double`. The random draw uses a small seeded generator in the crate (no new dependency), seeded with 0 for each window.
- [ ] **Step 3:** Check the port line by line against the source once more. Record each difference as a ruling.
- [ ] **Step 4: Commit** `feat(whisper): port the whisper.cpp token rules`.

### Task 4: The window loop

**Files:** `decode/window.rs` (new), `crates/jeuvoice-whisper/tests/decode_window.rs`.

The loop talks to the model through this trait:

```rust
/// The numeric part of Whisper, as the window loop sees it.
pub trait WhisperModel {
    /// Encodes the 30 s mel window that starts at `seek` (in frames).
    fn encode(&mut self, seek: usize) -> Result<(), WhisperError>;
    /// Starts a new self-attention cache and decodes `tokens`. Returns the logits of the
    /// rows that the caller asks for in `logit_rows`, `n_vocab` values for each row.
    fn decode_first(&mut self, tokens: &[u32], logit_rows: &[usize]) -> Result<Vec<f32>, WhisperError>;
    /// Decodes one more token with the cache. Returns its `n_vocab` logits.
    fn decode_next(&mut self, token: u32) -> Result<Vec<f32>, WhisperError>;
    /// The DTW pass: decodes `tokens` in one batch with a new cache. Returns the cross-attention
    /// softmax of the alignment heads as `[heads][tokens][n_audio]` for `n_audio` positions.
    fn alignment(&mut self, tokens: &[u32], n_audio: usize) -> Result<Vec<f32>, WhisperError>;
}
```

- [ ] **Step 1: Failing tests** with a `ScriptedModel` that returns logits with one high value at a scripted token for each step:
  - one window of text with timestamps gives the right segments and `seek`
  - an audio of 45 s gives two windows, and `seek` follows the last timestamp
  - the single-timestamp ending skips the window
  - a failed temperature (low `avg_logprobs`) moves to 0.2 and so on, and the last temperature is kept
  - the entropy failure and the repetition failure (220 steps)
  - a no-speech window gives no segments
  - the end of the audio stops the window (`seek + seek_delta + delta_min >= seek_end`)
  - an input shorter than 100 ms gives an empty transcript, as whisper.cpp does
  - language `"auto"` calls `decode_first(&[SOT], ..)` and takes the best language
  - progress values rise from 0.0 to 1.0
- [ ] **Step 2: Implement** `transcribe_tokens(model, vocab, samples_len, opts, progress) -> Result<Vec<Segment>, WhisperError>` as the port of the loop. `Segment` has `t0`, `t1`, the tokens (`TokenData` plus `t0`, `t1`, `t_dtw`) and the no-speech prob.
- [ ] **Step 3: Commit** `feat(whisper): port the whisper.cpp window loop`.

### Task 5: Token times without DTW

**Files:** `decode/token_times.rs` (new), `crates/jeuvoice-whisper/tests/decode_token_times.rs`.

- [ ] **Step 1: Failing tests:** `signal_energy` on a short hand signal. Then a fixture test: take the segments and token data of `<clip>.rs.json` (Task 0), clear `t0`/`t1`, run `token_level_times`, and compare with the whisper-rs values. They must be equal, because the input is the same.
- [ ] **Step 2: Implement** the port of `get_signal_energy` and `whisper_exp_compute_token_level_timestamps`, with the state that it keeps across segments (`t_beg`, `t_last`, `tid_last`).
- [ ] **Step 3: Commit** `feat(whisper): port the token-level times`.

### Task 6: DTW

**Files:** `decode/dtw.rs` (new), `crates/jeuvoice-whisper/tests/decode_dtw.rs`.

- [ ] **Step 1: Failing tests:**
  - `dtw_path` on 3 hand matrices (a diagonal, a wide matrix, a tie case) against paths worked out by hand from the source rules
  - `median_filter` with width 7 and the reflect padding at both ends
  - `normalize` over the tokens
  - `place_dtw_times`: the NOT row gets no time, text tokens of 2 segments get their times in order, non-text tokens are skipped
- [ ] **Step 2: Implement** the 9 DTW steps above as `dtw_times(model, segments, seek, n_frames)`. It calls `WhisperModel::alignment`.
- [ ] **Step 3:** Add a `ScriptedModel` case to the Task 4 tests: with a diagonal alignment, each text token gets the expected `t_dtw`.
- [ ] **Step 4: Commit** `feat(whisper): port the DTW word times`.

### Task 7: `Tensor::rows` and the embedding kernel

**Files:** `crates/jeuvoice-gpu/src/tensor.rs`, `crates/jeuvoice-gpu/src/kernels/mod.rs` (`embed_f16`), `crates/jeuvoice-gpu/src/ops/embed.rs` (new), `crates/jeuvoice-gpu/tests/embed.rs`, `gpu-bench.rs`, `docs/cuda-lessons/08-decoder.md` (new, first part).

The decoder works on 1 to 448 rows of tensors that the arena plans at the full size.

- [ ] **Step 1: Failing tests:** `t.rows(start, n)` gives a tensor with the offset of row `start` and the shape `[n, ..]`. It panics in plain words past the end. Then `embed`: `out[i] = tok_emb[tokens[i]] + pos_emb[pos0 + i]`, with `tok_emb` f16 `[51866, 1280]`, `pos_emb` f32 `[448, 1280]` (check the ggml types first), `out` f16. Test against a CPU f64 loop with ids near 0 and near `n_vocab`, and an id past the end must give an error, not a bad read.
- [ ] **Step 2: Implement.** The token ids go to the GPU in a small `u32` tensor.
- [ ] **Step 3: Bench row** `embed_f16` for 1 and 224 rows.
- [ ] **Step 4: Lesson 08, part 1** (Turkish): what the embedding is, the gather read pattern, the measured speed against the copy limit.
- [ ] **Step 5: Commit** `feat(gpu): add the token embedding kernel`.

### Task 8: The decoder attention kernel

**Files:** `kernels/mod.rs` (`attn_decode_f16`), `ops/attention.rs` (`attention_decode`), `tests/attention_decode.rs`, `gpu-bench.rs`, `docs/cuda-lessons/08-decoder.md`.

One kernel covers the 3 decoder cases:

| Case | Queries | Keys | Mask |
|---|---|---|---|
| self-attention, one step | 1 | `past + 1` (at most 448) | none |
| self-attention, prompt or DTW pass | `n` | `n` | causal |
| cross-attention | 1 or `n` | 1500 | none |

Q, K and V are read in the `[rows, 1280]` layout of the GEMM outputs and the caches. Head `h` is columns `64h .. 64h + 64`. So the decoder needs no split or merge of the heads. One block handles one (head, query row). The scores of one row fit in shared memory (1500 f32 = 6 KB). The softmax is in f32. An optional f32 output gets the softmax of the chosen heads, for DTW.

- [ ] **Step 1: Failing tests** against a CPU f64 reference: the 3 cases of the table, `n = 37` with the causal mask, `past = 447`, keys = 1500, a large-score case, and the probs output for heads 4 and 11 only.
- [ ] **Step 2: Implement** the naive version first and measure it. Then make it faster in measured steps (for example: a warp per key block, `[f16; 2]` loads, a split over the keys for the 1-query case). Stop when the 1-query cross-attention case is near the memory limit, and record how near.
- [ ] **Step 3: Bench rows** for the 3 cases.
- [ ] **Step 4: Lesson 08, part 2** (Turkish): the KV cache and why the decoder step is memory-bound, the naive kernel, each measured change, the limit, the PTX, exercises.
- [ ] **Step 5: Commit** `feat(gpu): add the decoder attention kernel`.

### Task 9: The f16 GEMM with an f32 output

**Files:** `crates/jeuvoice-gpu/src/blas.rs`, `tests/blas.rs`, `gpu-bench.rs`.

The logits are `x [rows, 1280] · tok_embᵀ` with `tok_emb [51866, 1280]`. whisper.cpp keeps them in f32. f16 logits can change the arg max and the log-softmax.

- [ ] **Step 1: Failing test:** `gemm_f16_f32out` (f16 A and B, f32 C, f32 compute) against a CPU f64 product at `1 x 51866 x 1280` and `7 x 51866 x 1280`.
- [ ] **Step 2: Implement** with `cublasGemmEx` (C type `CUDA_R_32F`).
- [ ] **Step 3: Bench row** at `1 x 51866 x 1280`: this reads the 133 MB embedding once, so it shows the memory limit of one decoder step.
- [ ] **Step 4: Commit** `feat(gpu): add the f16 GEMM with an f32 output`.

### Task 10: The decoder on the GPU

**Files:** `crates/jeuvoice-whisper/src/cuda/decoder.rs` (new), `cuda/weights.rs` (`DecoderWeights`), `crates/jeuvoice-whisper/tests/cuda_decoder.rs`, `tools/reference/whisper_reference.py` (decoder references), `tools/reference/README.md`.

The decoder has its own arena: weights, the self-attention cache `[4][2][448, 1280]`, the cross K/V `[4][2][1500, 1280]`, the activations for 448 rows and the logits `[448, 51866]` f32 (93 MB) or fewer rows (decide by measurement: the DTW pass needs no logits). The only reads across the two arenas are the 8 cross K/V GEMMs from the encoder output. These go through `Blas` with `DevSlice`s, so the one-arena rule of the ops still holds.

- [ ] **Step 1: Reference data.** Add `--decoder` to `whisper_reference.py`: for each clip, run the PyTorch decoder (fp16 and fp32) on the first window with the whisper-rs tokens of that window from `<clip>.rs.json`. Write the logits of each row and the alignment-head cross-attention softmax.
- [ ] **Step 2: Failing tests** (`#[ignore]`, they need the model):
  - `DecoderWeights::plan` finds every decoder tensor with the right shape. A missing tensor gives an error in plain words.
  - `decode_first` on the prompt, then `decode_next` for each token, against the reference logits: the arg max is the same on each row, and the logits are in a tolerance calibrated from the fp16 reference (as for the encoder taps).
  - One batch decode of all tokens gives the same logits as the step by step decode (the KV cache test).
  - `alignment` against the reference softmax.
- [ ] **Step 3: Implement** `Decoder::load`, `set_cross_kv(encoder_output)`, `decode_first`, `decode_next` and `alignment`. Each layer: layernorm, Q/K/V GEMMs into the cache rows, `attention_decode` self, out GEMM, add, layernorm, Q GEMM, `attention_decode` cross, out GEMM, add, layernorm, MLP with GELU, add. The end: layernorm, logits GEMM.
- [ ] **Step 4:** Implement the `WhisperModel` trait for a `GpuWhisper { gpu, blas, mel, encoder, decoder }`.
- [ ] **Step 5:** Measure one decoder step (the median of 200) and the cross K/V time per window. Write both in the development log.
- [ ] **Step 6: Commit** `feat(whisper): add the GPU decoder`.

### Task 11: `CudaWhisper`

**Files:** `crates/jeuvoice-whisper/src/cuda/transcriber.rs` (new), `crates/jeuvoice-whisper/tests/cuda_transcribe.rs`.

- [ ] **Step 1: Failing tests:**
  - A compile-time check that `CudaWhisper: Send + Sync`.
  - `CudaWhisper::load` with a missing model or an English-only model gives an error in plain words. With no GPU (`CUDA_VISIBLE_DEVICES=-1` in a child process) it gives `WhisperError::ModelLoad`.
  - `#[ignore]`: each test clip gives the same token ids as `<clip>.rs.json` in at least 98% of the positions, and the same word count within 2%.
  - `#[ignore]`: `transcribe` from two threads, one after the other, works (deferred minor M7: the other-thread test).
  - `#[ignore]`: a progress callback that panics gives `WhisperError::Transcribe`, as `WhisperRs` does.
- [ ] **Step 2: Implement.** `CudaWhisper::load(model_path)` starts the worker thread. The thread opens the `Gpu`, the `Blas`, `GpuMel`, the `Encoder` and the `Decoder`, then waits for jobs. `transcribe` sends the samples and gets back progress messages and the result over a channel. The caller thread calls `progress` (with the same `catch_unwind` rule as `WhisperRs`). The words come from `tokens_to_words` with the start and end rule of the app.
- [ ] **Step 3: Commit** `feat(whisper): add CudaWhisper`.

### Task 12: The measurement tool

**Files:** `crates/jeuvoice-whisper/examples/compare_whisper.rs` (new), `scripts/compare-whisper.ps1` (new), `docs/development-log.md`.

- [ ] **Step 1:** Write the example. `compare_whisper --backend rs|cuda <clips..>` loads the model once, runs each clip 3 times, and writes JSON: the words with times, the time of each run without the load, and the number of fallback windows.
- [ ] **Step 2:** Write a unit test for the WER function (word Levenshtein after Turkish lower case, `İ` to `i` and `I` to `ı`, and no punctuation) and for the start-time match (the words that the Levenshtein path matches).
- [ ] **Step 3:** Write the script. It runs each backend in its own process, samples `nvidia-smi --query-compute-apps` every 100 ms for the peak VRAM, and prints one table: WER for each clip, the median start difference, the time of the 69 s set, the peak VRAM.
- [ ] **Step 4:** Run it. Write the table in the development log, with the date.
- [ ] **Step 5: Commit** `feat(whisper): add the whisper comparison tool`.

### Task 13: The app path and the package

**Files:** `crates/jeuvoice-app/Cargo.toml` (feature `cuda-kernels`), `crates/jeuvoice-app/src/main.rs` (`LazyWhisper`), `packaging/windows/collect.ps1`, `packaging/windows/README.md`, `AGENTS.md` (the command).

- [ ] **Step 1: Failing tests** in `main.rs` or a small module: with `JEUVOICE_CUDA_WHISPER=1` and a loader that fails, `LazyWhisper` loads `WhisperRs`. Without the variable it never tries `CudaWhisper`. `loaded_on_gpu()` is true for `CudaWhisper`. Use loader functions as parameters so the tests need no model.
- [ ] **Step 2: Implement.** `LazyWhisper` holds `Arc<dyn Transcriber>` and the GPU flag. The `cuda-kernels` feature of the app turns on `jeuvoice-whisper/cuda-kernels`. Without the feature the app is as it is today.
- [ ] **Step 3:** Change `collect.ps1` to build with `scripts\cargo.cmd oxide build -- -p jeuvoice-app --release --features cuda-kernels`. Check that the release exe has the PTX in it (no PTX file next to the exe), and that it runs on this machine with and without the variable.
- [ ] **Step 4:** Run `scripts\cargo.cmd test` (the default members, no feature) to show that the normal build did not change.
- [ ] **Step 5: Commit** `feat(app): run CudaWhisper behind JEUVOICE_CUDA_WHISPER`.

### Task 14: Close the gaps

This task depends on the Task 12 numbers. Do not start a speed change before a measurement shows where the time goes.

- [ ] **Step 1:** If a criterion fails, find the cause with a measurement first: Nsight Systems for the time (the kernel gaps, the host waits, the logits download), the arena sizes for the VRAM, the Task 12 word lists for the WER.
- [ ] **Step 2:** Write the cause and the planned change in the ledger, then show it to the user before a large change. Known candidates:
  - Time: the logits download for each step (207 KB), the CPU rules, the launch count per step. A GPU arg max with the rules on the GPU is possible but larger.
  - VRAM: fewer logits rows, the cross K/V in the encoder arena, the encoder activations reused by the decoder.
  - WER: the fp16 residual stream of the encoder (handoff section 7: try an f32 residual).
- [ ] **Step 3:** Each change: a test or a bench row first, the change, a new measurement, a commit.
- [ ] **Step 4:** Run Task 12 again until all six criteria pass. Write the final table in the development log.

### Task 15: Lesson 09, docs and the review

**Files:** `docs/cuda-lessons/09-dtw.md` (new), `docs/cuda-lessons/README.md` or the index, `docs/development-log.md`, the handoff, `AGENTS.md`.

- [ ] **Step 1: Lesson 09** (Turkish): what DTW does for the word times, the alignment heads, why it runs on the CPU here (measure the CPU time per window), the 7 parts as far as they apply, with exercises (for example: a GPU median filter).
- [ ] **Step 2:** Update the handoff (what is done, what is next: Phase 2 or the TTS) and the development log status section.
- [ ] **Step 3:** A fresh reviewer reads the whole branch. Fix each Critical or Important finding with a test that fails first. List the Minor ones as deferred.
- [ ] **Step 4:** Ask the user before the merge to `main`.

## Watch list

The user approved the plan on 2026-11-25 and asked to keep these risks in view. Each task report says the state of each item.

- **W1 The no-speech prob.** Settled in Task 0: whisper.cpp reads row 0 of its logits buffer. That row holds the raw logits of the last 1-token decode (a step of the window before, or the language detection), or zeros in a new state. Task 4 copies this, and a `ScriptedModel` test checks both cases.
- **W2 The fallback windows.** Above temperature 0, our random draws are not the whisper.cpp draws. Count the fallback windows in each measurement, and report them next to the WER.
- **W3 The VRAM.** `CudaWhisper` needs about 1.7 GB. Compare it with the whisper.cpp peak of Task 0 after each task that adds memory (Tasks 10 and 11).

## Review Focus

1. The token rules: the order, and the case of the first sampled token. One rule in the wrong order changes the text.
2. Centiseconds against audio positions (one position is 2 frames, 20 ms). A factor of 2 error moves every word.
3. The KV cache rows: a decode step writes row `past`, and the causal mask allows keys up to `past + row`.
4. The DTW row and column order, and the NOT row that gets no time.
5. Ids at the vocab end: `BEG + 1500` is the last timestamp (30.00 s), and no id may read past `n_vocab`.
6. The worker thread: a panic in the thread must give an error to the caller, not a hang.
7. The fallback: any load error of `CudaWhisper` must give a working `WhisperRs`.

## Self-check of this plan

- Every Phase 1b item of the handoff has a task: decoder (10), rules (3), window loop and fallback (4), DTW (6), `CudaWhisper` (11), app (13), package (13), criteria (12, 14), lessons 8 and 9 (7, 8, 15), M1 (1), M6 (11), M7 (11).
- Open facts that a task must settle, not guess: the no-speech row (Task 0), the mel value past the end (Task 1), the type of the positional embedding (Task 7), the logits rows (Task 10).
