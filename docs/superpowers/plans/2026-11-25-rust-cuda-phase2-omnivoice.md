# Phase 2: OmniVoice main model (Qwen3 backbone and masked iterative decoding) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the OmniVoice text-to-speech *main model* runs on the Rust kernels: text and an optional voice prompt in, **audio tokens** out (8 codebooks, 25 frames per second), with the same tokens as the Python engine and faster than it. The audio decoder (tokens to a waveform) is Phase 3, so Phase 2 produces tokens and a reference comparison, not a WAV file.

**Architecture:**

- One new host crate `crates/jeuvoice-omnivoice` (no device code, hard rule 2). It reads the safetensors weights, merges the LoRA adapter, runs the masked decoding loop on the CPU, and calls `jeuvoice-gpu` ops for the numbers.
- The model is a Qwen3-0.6B transformer, 28 layers, run **bidirectionally** over one packed sequence: `[special tokens, text, prompt audio frames, masked target frames]`. Each decoding step runs one full forward pass over the whole sequence, for both the conditioned and the unconditioned input (`2 · B` rows, classifier-free guidance).
- The GPU does the matmuls (cuBLAS), RMSNorm, RoPE, SwiGLU, attention and the audio head. The CPU does the schedule, the mask bookkeeping, the priority score, the sampling and the text-to-frames estimate.
- The Python engine stays the product path: the app keeps calling it (Phase 3 and 5 change that). Phase 2 adds a developer tool and tests only.

**Tech stack:** Rust 1.99.1, cuda-oxide at the pinned rev, cuBLAS 13, the existing `jeuvoice-gpu` ops and kernels, `safetensors` for the weights, the Python engine (`engine/.venv`) as the reference.

**Spec:** `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md` (roadmap line 23; Phases 0 and 1 are done). **Handoff:** `docs/superpowers/handoff/2026-11-25-rust-cuda-phase1b-handoff.md`, section 4.

## Global Constraints

- All rules of `AGENTS.md` apply. The most important: Rust kernels only, device code only in `jeuvoice-gpu`, raw device pointers and a `SAFETY` comment on each `unsafe` block, PTX `sm_75`, fp16 data with f32 accumulate, tests first, a Turkish lesson for each kernel, a `gpu-bench` row for each kernel, English comments in STE style, UTF-8 without a BOM.
- Branch: `rust-cuda-phase2`. Commit after each task. Ledger: `.superpowers/sdd/phase2/progress.md`.
- The app UI does not change (hard rule 11). The app keeps the Python engine as its TTS backend in Phase 2.
- The Python engine is the behaviour reference. When this plan and the engine source disagree, the engine source wins. Record the case as a `Ruling:` line in the ledger.
- Model files: `k2-fsa/OmniVoice` snapshot `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` under `%USERPROFILE%\.cache\huggingface\hub\models--k2-fsa--OmniVoice\snapshots\<rev>\`. The LoRA adapter: `%APPDATA%\Jeu\JeuVoice\data\models\emirbakid__jeuvoice-omnivoice-tr2\`. Both are already on this machine. Nothing new is downloaded.
- VRAM: 4 GB must work. The engine uses 2.4 GB today, so the weights go up as **f16** (about 1.23 GB) with **f32 norms** and **f32 accumulate** (hard rule 5), not as the f32 file (2.45 GB).

## The OmniVoice facts this plan uses

Checked in `engine/.venv/Lib/site-packages/omnivoice/models/omnivoice.py` (line numbers for that file) and the checkpoint config on 2026-11-25.

**The checkpoint.** `config.json` holds an OmniVoice config with a nested `llm_config`; `model.safetensors` is 2,450,344,112 B, **all F32**, 313 tensors, **612,577,288 params**, and has **no `lm_head`** (text logits are never needed). Plus `audio_tokenizer/` (805,665,628 B, Higgs Audio v2, 8 codebooks of 1024 at 25 frames per second, 24 kHz) — that one is **Phase 3**.

**The LLM (`config.json:20-96`).** `num_hidden_layers` **28**, `hidden_size` **1024**, `num_attention_heads` **16**, `num_key_value_heads` **8** (GQA, 2 q heads per kv head), `head_dim` **128**, `intermediate_size` **3072**, `rope_theta` **1e6**, `rope_type` "default" (no scaling), `rms_norm_eps` **1e-6**, `vocab_size` **151676**, `tie_word_embeddings` true, every layer `full_attention` (no sliding window). **qk-norm**: `self_attn.q_norm` and `k_norm`, shape `[128]`, 28 each, applied to every head before RoPE (RMSNorm, the same eps).

**The audio side.** `audio_embeddings` `[8200, 1024]` = 8 codebooks × 1025 (code 1024 is the mask), `audio_heads` a linear `[8200, 1024]` with no bias. A position's embedding is the **sum of its 8 codebook embeddings**, codebook `c` adding offset `1025 · c` (`omnivoice.py:470-490`). Mask id **1024**.

**The generation loop** (`_generate_iterative`, `omnivoice.py:1275-1427`).

- Every step runs **one full forward pass over the whole sequence**; there is **no KV cache between steps** (all positions change). Cost per step is one 28-layer pass over `T ≈ 100 + prompt frames + target frames`.
- The attention is **bidirectional**: `batch_attention_mask[i, :, :c_len, :c_len] = True` (`omnivoice.py:1339`). Padding positions get a diagonal-only mask (`:1345-1347`). This is not a causal LM pass.
- Batch is `2 · B` rows: conditioned rows `0..B` and unconditioned rows `B..2B`. The unconditioned row is only the last `target_len` tokens, without the style, text and prompt (`:1342-1344`).
- `num_step` **32** (the app's preset, `preset.rs:148`), mask schedule `linspace(0, 1, num_step + 1)` shifted by `t_shift = 0.1` (`t / (0.1 + 0.9 t)`), `k = ceil(masked_total · (t[k+1] - t[k]))` positions decoded per step, the last step takes the remainder (`:1356-1378`, `_get_time_steps:1639-1648`).
- Position priority at a step: `score = max log prob - 5.0 · layer_id`, plus Gumbel noise scaled by `position_temperature = 5.0`, where `layer_id = id / 1025` (`:1407-1415`).
- Guidance (the app's 2.5): `lp_c = log_softmax(cond)`, `lp_u = log_softmax(uncond)`, `logits = lp_c + g · (lp_c - lp_u)`, then `log_softmax`; the mask id gets `-inf` (`:1429-1452`).
- Sampling: `variety = 0` means argmax. Above 0: top-k with `k = ceil(0.1 · 1025) = 103`, then Gumbel (`:1624-1636`).
- The RNG is `torch.manual_seed` per part plus `torch.rand_like` (Philox).

**The prompt layout** (`_prepare_inference_inputs:1198-1273`): `[denoise] <|lang_start|>{lang}<|lang_end|> <|instruct_start|>{instruct}<|instruct_end|> <|text_start|>{ref_text + " " + text}<|text_end|>[prompt audio frames][masked target frames]`, each block repeated over the 8 codebook rows. Ids: denoise **151669**, lang_start **151670**, lang_end **151671**, instruct_start **151672**, instruct_end **151673**, text_start **151674**, text_end **151675**, eos **151645**, pad **151643**.

**The LoRA adapter** (`utils/lora.py:32-58`): `r = 32`, `alpha = 64` (scale 2.0), targets `q, k, v, o, gate, up, down` for all 28 layers (196 A/B pairs), `modules_to_save = audio_embeddings, audio_heads` (replaced, not merged).

**Text to frames** (`RuleDurationEstimator`, `utils/duration.py:208-248`): the frame count comes from character weights, scaled by `speed`. Text longer than 30 s is split into 15 s chunks with a cross-fade (`omnivoice.py:186-187, 913-1022`); that path can wait for Phase 3.

**Today's speed** (README, RTX 3050 Ti): model load 25.5 s and 2.4 GB VRAM; 5.6 s of audio in 12.1 s wall time. **Measured in process on 2026-11-25** (`scripts\tts-baseline.ps1`, the fixtures of Task 1): load 7.2 s, the voice prompt 2.2 s, a render of 8.37 s of audio in 6.1 s, of which the token stage is 4.6 s and the audio decode 1.7 s, and 2034 MiB above idle. The sequence of that script is 486 positions with a 186-frame prompt.

## Success criteria

Decided with the user before Task 1. The numbers below are the starting proposal; the user can change them, and only the user can.

1. **Tokens**: with the deterministic reference mode (`position_temperature = 0`, `variety = 0`), every audio token of 3 test scripts equals the Python reference, and the frame count matches exactly.
2. **Product mode** (noise on, `guidance 2.5`, `num_step 32`, `variety 0`): **the frame count matches the engine** on the same scripts. A token share is not reachable: measured on 2026-11-25, the product mode differs from the deterministic mode on 1689 of 1712 positions, because the Gumbel noise on the position score changes the unmask order. The quality of the product mode is judged after Phase 3, by ear.
3. **Speed**: the token stage of the 8.4 s script s1 in **2.3 s or less**, which is half of the engine's measured 4.6 s for the same script (the engine's whole render: 6.1 s), measured with `bench::time_on_stream` plus the loop's host time.
4. **VRAM**: 2.0 GB or less above idle for the weights and activations (the engine measured 2034 MiB), on a 4 GB card.
5. **Load**: model plus LoRA in **25 s or less**, LoRA merge included.
6. The existing tests of the workspace still pass, and the app still renders through the Python engine.

## Review Focus

1. The **masked-decoding bookkeeping**: exactly `ceil(masked · Δt)` positions per step, no position twice, no position left masked at the end, and the mask logit set to `-inf` before the softmax.
2. The **guidance** formula. `log_softmax` twice, the scale on the difference, and the mask applied after the combination. A missing normalisation changes every token.
3. The **attention mask**: bidirectional over the real positions, diagonal-only on padding, and the unconditioned row built from the *last* `target_len` tokens only.
4. The **8-codebook embedding**: the offset `1025 · c` and the sum, not the mean, and the prompt frames' codebooks in the `[8, T]` order of the file.
5. **GQA**: 16 q heads against 8 kv heads, the pair `h / 2`, and qk-norm before RoPE, not after.
6. **RoPE**: the pair layout (interleaved `(2i, 2i+1)`, not the split-half form), `theta = 1e6`, and the position offsets of the prompt and the target.
7. **f16 against the reference**: where f16 weights and activations lose enough to flip a token. Measure the logit difference per layer, do not guess.
8. The **LoRA merge**: scale `alpha / r = 2.0`, `W + scale · B · A`, and the replaced `audio_embeddings` and `audio_heads` overriding the base tensors.

## Tasks

### Task 1: Baseline and reference fixtures

**Files:** `tools/reference/omnivoice_reference.py` (new), `tools/reference/README.md`, `scripts/tts-baseline.ps1` (new).

**Interfaces:**
- The tool runs the engine's model with `position_temperature = 0` and `variety = 0` over the voice prompt (`D:\MLProjelerim\tts_lab\data\ref\my_voice.ogg`) and 3 texts of about 5 s, and writes `<clip>.tokens.npz` (the final `[8, T]` tokens) and `<clip>.taps.safetensors` (per-layer hidden states and the per-step logits of one script).
- A second record holds the product-mode result (`position_temperature` 5.0, `variety` 0, `guidance` 2.5, `num_step` 32) for criterion 2.
- `scripts/tts-baseline.ps1` prints the engine's wall time per render and the peak VRAM of the card, in the shape of `scripts/compare-whisper.ps1`.

**Steps:**
- [ ] **Step 1:** Write the tool. It sets the model config to the deterministic schedule and records every step's chosen positions and logits.
- [ ] **Step 2:** Make the fixtures for the prompt and the 3 texts, and the product-mode record. Write the numbers into `docs/development-log.md`.
- [ ] **Step 3:** Commit `feat(tts): add the OmniVoice reference tool and the baseline`.

### Task 2: The crate, the config and the weight index

**Files:** `crates/jeuvoice-omnivoice/{Cargo.toml,src/lib.rs,src/config.rs,src/weights.rs}`, `crates/jeuvoice-omnivoice/tests/config.rs`, workspace `Cargo.toml`.

**Interfaces:**
- `OmniVoiceConfig` from `config.json`: the 11 LLM fields of "The OmniVoice facts", `num_audio_codebook = 8`, `audio_mask_id = 1024`, `audio_vocab_size = 8200`, the special ids, `guidance_scale`, `num_step`, `t_shift`, `position_temperature`.
- `tensor_index(path) -> Vec<(name, dtype, shape, offset)>` over `model.safetensors` and the adapter file.
- A check for the model variant, as `check_large_v3_turbo` does for Whisper: 28 layers, hidden 1024, 8 codebooks, vocab 151676, or a plain `ModelLoad` error.

**Steps:**
- [ ] **Step 1: Failing tests** `config_from_the_real_file`, `a_wrong_model_is_rejected`, `the_index_finds_every_tensor`.
- [ ] **Step 2:** Implement, run the tests with `-- --ignored` (the files are on this machine only).
- [ ] **Step 3:** Commit `feat(tts): read the OmniVoice config and tensor index`.

### Task 3: RoPE and per-head RMSNorm (lesson 10)

**Files:** `crates/jeuvoice-gpu/src/kernels/mod.rs`, `ops/rope.rs`, `ops/rmsnorm.rs`, `tests/rope.rs`, `bin/gpu-bench.rs`, `docs/cuda-lessons/10-rope-qk-norm.md`.

**Interfaces:**
- `ops::rope_f16(gpu, arena, x: &Tensor /*[rows, heads, 128]*/, positions: &Tensor /*I32 [rows]*/, theta: f32)` in place: the interleaved pair `(2i, 2i+1)` with `cos/sin(pos · theta^(-2i/128))`.
- `ops::rmsnorm_rows_f16(gpu, arena, x /*[rows, 128]*/, w /*[128] F32*/, eps)` in place, for q-norm and k-norm per head.

**Steps:**
- [ ] **Step 1: Failing tests** against a CPU f64 reference (positions 0, 1, 7, 200; a non-zero row offset).
- [ ] **Step 2:** Implement; measure with `gpu-bench` rows `rope_f16 250x16x128` and `rmsnorm_rows_f16 4000x128`; write lesson 10.
- [ ] **Step 3:** Commit `feat(gpu): add RoPE and per-head RMSNorm with lesson 10`.

### Task 4: SwiGLU (lesson 11)

**Files:** the same files as Task 3, `ops/swiglu.rs`, `tests/swiglu.rs`, `docs/cuda-lessons/11-swiglu.md`.

**Interfaces:** `ops::swiglu_f16(gpu, arena, gate: &Tensor /*[rows, 3072]*/, up: &Tensor, out: &Tensor)` with `out = silu(gate) · up`, `silu(x) = x / (1 + exp(-x))`, f32 math. In place on `gate` when the caller allows it.

**Steps:**
- [ ] **Step 1: Failing test** against a CPU f64 reference, and the values `±20`, `0` (no overflow in `exp`).
- [ ] **Step 2:** Implement, bench `swiglu_f16 250x3072` against the memory limit, write lesson 11.
- [ ] **Step 3:** Commit `feat(gpu): add SwiGLU with lesson 11`.

### Task 5: The audio embedding gather

**Files:** `kernels/mod.rs`, `ops/embed.rs`, `tests/embed.rs`, `bin/gpu-bench.rs`.

**Interfaces:** `ops::embed_audio_sum_f16(gpu, arena, ids: &Tensor /*I32 [8, T]*/, table: &Tensor /*[8200, 1024] F16*/, out: &Tensor /*[T, 1024] F16*/)`: `out[t] = Σ_c table[ids[c][t] + 1025·c]`.

**Steps:**
- [ ] **Step 1: Failing test** against a CPU f64 reference, including the mask id and the codebook offset.
- [ ] **Step 2:** Implement, bench, commit `feat(gpu): add the 8-codebook audio embedding`.

### Task 6: Weights, the LoRA merge and the arenas

**Files:** `crates/jeuvoice-omnivoice/src/weights.rs`, `tests/weights.rs`.

**Interfaces:**
- `LlamaWeights::load(gpu, index, adapter: Option<&Path>) -> Result<LlamaWeights, WhisperError-like error>`: matrices as **F16**, norms and q/k-norm as **F32**, the embedding table F16, the audio head F16.
- The LoRA merge on the host in f32 (`W + 2.0 · B · A`), then a single f16 upload; `audio_embeddings` and `audio_heads` replaced by the adapter's `modules_to_save` tensors.
- One arena: weights, then activations for one pass (the plan sizes them from `T_max = 1024`).

**Steps:**
- [ ] **Step 1: Failing test** `merge_lora_on_a_small_matrix` (a hand-computed 2×2 case) and `a_missing_adapter_tensor_is_an_error`.
- [ ] **Step 2:** Implement, measure the load time and the arena bytes; write both into the development log.
- [ ] **Step 3:** Commit `feat(tts): load the OmniVoice weights and merge the LoRA adapter`.

### Task 7: The forward pass, layer by layer

**Files:** `crates/jeuvoice-omnivoice/src/forward.rs`, `tests/forward.rs`.

**Interfaces:** `Forward::run(&mut self, ids: &Tensor /*[8, T]*/, positions: &Tensor, out_hidden: &Tensor /*[T, 1024]*/) -> Result<(), Error>`, and `Forward::logits(&mut self, hidden: &Tensor /*[T, 1024]*/, out: &Tensor /*[T, 8200] F32*/)`.

**Steps:**
- [ ] **Step 1: Failing test** `hidden_matches_the_reference_layer_by_layer`: the taps of Task 1 against our hidden states after the embedding, after layer 0, 13 and 27, with `Tol { abs: 2e-2, rel: 2e-2 }` to start; print each difference.
- [ ] **Step 2:** Implement layer by layer, run the test after each layer, fix the first layer that is out before going on.
- [ ] **Step 3:** Commit `feat(tts): run the OmniVoice forward pass on the GPU`.

### Task 8: Bidirectional GQA attention at head width 128

**Files:** `ops/attention.rs` (a GQA kind), `crates/jeuvoice-omnivoice/src/forward.rs`, `tests/attention.rs`, `bin/gpu-bench.rs`.

**Interfaces:** `ops::attention_gqa(gpu, arena, blas, q /*[16, T, 128]*/, k /*[8, T, 128]*/, v, out, scratch, kind)`: q head `h` reads kv head `h / 2`, no causal mask, an optional padding mask that lets a pad row see only itself.

**Steps:**
- [ ] **Step 1: Failing test** against a CPU f64 reference at `T = 37` and `T = 300`; and the existing encoder attention as a cross-check for `heads_q = heads_kv`.
- [ ] **Step 2:** Implement (naive cuBLAS path first, then the tensor-core path if the shapes fit `sm_75`); bench at `16 x 500 x 128`.
- [ ] **Step 3:** Commit `feat(gpu): add the bidirectional grouped-query attention`.

### Task 9: The masked iterative decoding loop

**Files:** `crates/jeuvoice-omnivoice/src/decode.rs`, `tests/decode.rs`.

**Interfaces:** `decode(model, prompt: &Prompt, params: DecodeParams) -> Result<Tokens /*[8, T]*/, Error>` with `mode: Deterministic | Product`, `guidance`, `num_step`, `t_shift`, `variety`, `seed`.

**Steps:**
- [ ] **Step 1: Failing test** `tokens_match_the_deterministic_reference` on 3 scripts (criterion 1), and unit tests for the schedule (`ceil` and the remainder) and the priority score.
- [ ] **Step 2:** Implement, run the tests.
- [ ] **Step 3:** Commit `feat(tts): decode with masked iterative sampling`.

### Task 10: The product path and the seeds

**Files:** `decode.rs`, `sample.rs`, `tests/decode.rs`.

**Interfaces:** top-k 103 + Gumbel at `variety > 0`, Gumbel noise on the position score at `position_temperature`, the seed rule of the engine (`base_seed + part_index`), and `position_temperature = 0` as a test-only flag.

**Steps:**
- [ ] **Step 1: Failing test** `product_mode_matches_the_frame_count` and `at_least_99_percent_of_the_tokens` (criterion 2).
- [ ] **Step 2:** Implement; record the measured share and the frame counts.
- [ ] **Step 3:** Commit `feat(tts): sample like the engine in product mode`.

### Task 11: Text layout and the frame estimate

**Files:** `crates/jeuvoice-omnivoice/src/text.rs`, `tests/text.rs`.

**Interfaces:** the packed sequence of "The prompt layout", the 8-row repetition, `frames = estimate(text, speed)` as `RuleDurationEstimator` (`utils/duration.py:208-248`), the language code mapping, and the 30 s chunking rule (tokens only; the cross-fade waits for Phase 3).

**Steps:**
- [ ] **Step 1: Failing tests** `layout_of_a_prompt_with_a_reference`, `frame_estimate_matches_the_engine` (against the engine's `target_lens` for the 3 scripts), `an_unknown_language_is_an_error`.
- [ ] **Step 2:** Implement, run the tests.
- [ ] **Step 3:** Commit `feat(tts): build the OmniVoice prompt and estimate the frames`.

### Task 12: The render tool, the measurements, lesson 12 and the review

**Files:** `crates/jeuvoice-omnivoice/examples/render_tokens.rs`, `docs/cuda-lessons/12-masked-decoding.md`, `docs/development-log.md`, the handoff, `scripts/tts-baseline.ps1`.

**Steps:**
- [ ] **Step 1:** The tool renders tokens for one script and writes `[8, T]` int32, with progress and the per-step time.
- [ ] **Step 2:** Measure the 4 criteria (tokens, speed, VRAM, load) and write the table next to the engine's numbers.
- [ ] **Step 3:** Lesson 12 (Turkish): the masked iterative loop. What it does, why the model unmasks by confidence, the schedule, the guidance formula, why the attention is bidirectional, why there is no KV cache, the measured cost per step, the exercises (a KV-cache variant with a fixed prefix, a cosine schedule, a different guidance).
- [ ] **Step 4:** A fresh reviewer reads the branch; fix the Critical and Important findings with tests; list the minors as deferred.
- [ ] **Step 5:** Commit `feat(tts): render audio tokens on the Rust kernels, lesson 12`.

## Watch list

- **W1 The RNG.** The engine decides *which* positions to unmask with Gumbel noise, so token equality is only possible in the deterministic mode. The product path is compared by frame count and token share.
- **W2 The VRAM.** f16 weights are 1.23 GB. Activations for `T = 1024` and 2 rows: hidden 2 × 1024 × 1024 f16 = 4 MB, the MLP buffer 2 × 1024 × 3072 = 12 MB, attention scores 2 × 16 × 1024² f16 = 67 MB. A tensor-core flash attention would drop the last one. Measure after Task 8.
- **W3 The audio tokenizer.** Phase 2 stops at tokens. A listenable WAV needs Phase 3, so the ear check waits.
- **W4 The prompt audio.** Phase 2 uses prompt frames from the engine's tokenizer (a fixture). Encoding a new voice is Phase 4.

## Self-check of this plan

- Every Phase 2 item of the spec has a task: the Qwen3 backbone (2, 3, 6, 7, 8), masked iterative decoding (9, 10, 12), classifier-free guidance (9), the LoRA merge (6), the audio head and tokens (5, 7), text to frames (11), measurements and the lesson (12).
- The kernels that are new have tests, bench rows and lessons: RoPE and qk-norm (10), SwiGLU (11), the audio gather (no lesson, it is a gather), GQA attention (folded into lesson 5 as an extension).
- Facts that a task must settle, not guess: the attention mask (settled here, bidirectional), the deterministic mode of the reference (Task 1), the f16 tolerance per layer (Task 7), the VRAM of the activations (Task 8).
