# Rust CUDA Phase 2: audit implementation handoff (2026-11-25)

## 1. State

Branch `rust-cuda-phase2`, base commit `651e2bb`. The audit revisions of Work packages A and B are **in the working tree, not committed** (the audit rules require explicit user approval before a commit). After the user selected P2: C (the baseline and the production-shape bench), D3, D4 and E2 are done; D1 is fixed (the mask grid used the key-head batch); D2 was measured and is not used; the E1 experiment is open. Task 10 (the product mode) and Task 12 (the tool, the lesson, the criteria table) are done, and this file now carries the reviewer's findings.

One kernel was added: `guide_peak_f32`, with a bench row (`161.1 us`, 91% of the memory limit) and lesson 12. The four tail guards stand.

## 2. Work package A: safety and validation

| Finding | Fix | Test that proves it |
|---|---|---|
| A1, four kernels write past the tensor because the launch grid is rounded up to 256-thread blocks | `rope_half_f16`, `rope_f16`, `to_heads_f16`, `from_heads_f16` take the exact work-item count `n` and return when `i >= n` | `rope_does_not_write_past_the_tensor` (a 1-pair shape, canary), the heads canary in `the_head_layout_matches_the_row_layout` |
| A2, ids and masks reach the embedding tables unchecked; a bad `valid` slice panics | `src/inputs.rs`: every text id in `0..vocab`, every audio id in `0..audio_vocab`, mask 0 or 1, `valid.len() == rows`, each length in `0..=t`; errors carry the row, position and value | `tests/inputs.rs`, 9 pure cases; the failure was proved by muting the checks (6 failed) |
| A3, the weight test expects 3-D activations that the plan does not have | test-only change to `[BATCH * T_MAX, d]` and `[BATCH * T_MAX, audio_head_rows]` | `the_weights_load_and_fit_the_plan` (real load, 1469 MB) |
| A4, an incomplete adapter silently keeps base weights; `r = 0` asserts | `lora::check_adapter`: the two LoRA halves of all 196 `llm.layers.*` linear layers, both replacement tables with the base shape, no orphan adapter key, rank above zero; runs before `Arena::new` | `tests/lora.rs`, 6 new cases; the real adapter still loads |

Notes that a next agent needs:

- The canary must sit **directly after** the destination in the arena plan. The arena aligns every tensor to 256 bytes, so a small overflow can hide in the padding. A shape whose last block is almost empty (one pair) makes the overflow large and reliable.
- `check_adapter` derives the required set from the base tensors, so it stays correct if the layer count changes.
- `valid` is checked before the `copy_from_slice` that used to panic.

## 3. Work package B: behaviour and test quality

| Finding | Fix |
|---|---|
| B1, the nonverbal split accepted any `[a-z0-9-]` tag | the engine's exact 13-alternative `_NONVERBAL_PATTERN` |
| B2, a pre-split error or an unknown byte was dropped silently | `encode`, `encode_plain`, `bpe_word`, `style_ids` and `tokenize_with_nonverbal` return `Result`; the error names the segment or the byte |
| B3, a literal `\n` in a comment | comments and `assert_eq!` on real lines |
| B4, `guided_peak` had no direct test | a unit test against an f64 implementation of the engine's three log_softmaxes, with `guidance` 0 and 2.5, a mask logit that is the maximum, and logits around 1e4 |
| B5, the real-model assertions were weak | the decode test bounds the share against the f32 run (`ours_f32 + 0.05 >= engine`), the forward test asserts the exact tap-name set and the logit maximum error (0.059 conditioned, 0.041 unconditioned) |

The engine vectors for all 13 tags, for `[foo]` in context, and the whole s1 layout still match after B1 and B2 (`--test text -- --ignored`).

## 4. Numbers after the revisions

| Item | Value |
|---|---|
| plain tests | 455 |
| GPU tests | 114 |
| whisper GPU tests | 70 |
| real forward test | cond argmax 1694/1712 (98.9%), uncond 1712/1712, max logit error 0.059 / 0.041 |
| real decode, s1 | 4.8 s, 19.8% of the fp16 tokens, 23.7% of the f32 tokens (the engine keeps 20.0% of its own f32 tokens) |
| real decode, s2 / s3 | 4.1 s / 4.1 s |
| arena | 1469 MB |

The decode time is still above the 2.3 s criterion; that is Work package C/D/E work. The measurements above come from the existing wall-clock paths.

## 5. Documentation that changed

- The plan's time-shift formula is now `0.1 t / (1 - 0.9 t)` (the implementation always matched the engine).
- Lesson 10 ties the shared-RoPE 98.9 µs to **token-major** input; on head-major input the kernel sees `heads = 1` and there is no sharing.
- Lesson 11 says the 53.8 µs is one row and that production is `[972, 3072]`.
- The Task 7 handoff now names `651e2bb` and points here.

## 6. Open items for the supervisor

1. **Commit approval.** The tree is ahead of `651e2bb` and not committed.
2. **Criterion 1** is unchanged and still needs a decision: the engine's own fp16 run keeps 20% of its f32 tokens, we keep 19.8%, and the first step's six best scores span 0.05 with a 0.002 gap.
3. **Performance.** C (reliable CUDA-event timing and the production-shape bench), D (grouped attention GEMMs, head on used rows, direct f32 residual accumulation, redundant-work removal, token-major RoPE) and E (ragged unconditioned row, GPU guidance) need the user's selection before implementation. The order in the audit is C, then D, then E.

## 7. Commands

```
scripts\cargo.cmd test
scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test decode_gpu -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test weights -- --ignored --nocapture --test-threads 1 the_weights_load_and_fit_the_plan
```
