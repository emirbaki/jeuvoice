# E3: a d128 bidirectional flash attention — plan for a supervisor's review (2026-11-25)

Status: **not started**. This note is the brief for a revision before any code is written.

## 1. Where the tree is

Branch `rust-cuda-phase2`, `main` at `e84f780` (the Phase 2 merge). The working tree holds the
two-pass decode and its documentation. Measured state:

| Item | Value |
|---|---|
| warm decode, s1 / s2 / s3 | 2.55 / 2.7 / 2.5 s (the engine 4.6 s) |
| step | 77.4 ms (one conditioned pass of 486 rows and one unconditioned of 214) |
| attention, no padding, `[2,16,486,128]` | 881.7 us per layer (28 layers = 24.7 ms per step) |
| tests | 455 plain, 114 GPU, 70 whisper GPU, all green |
| criterion 3 | 2.55 s against 2.3 s, so about 0.25 s short |

The decode now calls the attention with `valid = None` (no padding), so the mask kernel is not
in its path. The forward tap test still uses `valid`, so the padded path must keep working.

## 2. What E3 would do

Today `attention_bidir` writes the whole score plane `[b·hq, t, t]` f16 (15 MB at the s1 shape),
then a second pass applies the padding mask, a third the softmax, and then the `probs · V`
GEMM. A flash kernel keeps a running maximum and a running sum per query row, walks the keys in
tiles, and never writes the score plane. The softmax rescaling (`exp(old_max - new_max)`) makes
the result algebraically identical.

Why it should pay: the score plane is written and read about three times per layer, 45 MB at
192 GB/s is about 0.23 ms of the measured 0.88 ms. The rest is the two GEMMs and the shared
memory work.

## 3. Scope decision that needs the supervisor

- **Option 1 (recommended): flash only when `valid = None`.** The decode uses that path; the tap
  test and any padded caller keep the current kernel. Smallest risk, no padding logic in the new
  kernel.
- **Option 2: flash with the diagonal padding rule.** Needed if any caller with `valid` must
  flash. The rule (a row past the valid length keeps only its diagonal) is awkward in a tile
  loop and is not needed by production today.

## 4. Design sketch

- One block per `(batch, query head, query tile)`; query tile 64 rows (486 is not a multiple, so
  a tail guard is mandatory).
- Key/value tiles of 64 rows of 128 f16 (16 KB each in shared memory). Scores live in registers
  as f32 running sums; no global score buffer.
- GQA: query head `h` reads key head `h / 2`; the tile loop indexes K/V by that head, which is
  the same mapping the current kernel uses.
- Both directions: every tile of keys is visited; there is no causal cutoff.
- `sm_75`: the existing d64 flash tensor-core kernel (`FLASH_*`, `FTC_*` constants) is the
  template. No async copy, 64 KB of shared memory, `m16n8k16` MMA. A d128 tile needs 4 MMA
  steps per 64 keys; the register budget for the 64x128 output tile is the first thing to check.
- The op keeps its signature and gains a `kind: Scores | Flash`; `Flash` ignores the scores
  scratch and skips the mask and softmax.

## 5. Numerical quality

- The online softmax is exact; the running sums are f32, which is **more** accurate than the
  current f16 score plane. Expect the logs to move by the same order as D3 (about 1e-3), not
  worse.
- The risk is not accuracy but a wrong tile/index: the same failure class as D1 (one row exact,
  another silently wrong). Acceptance must therefore compare every head and both the first and
  the last tile.

## 6. Acceptance tests

1. A GPU test against the CPU f64 reference at `t = 1`, `37`, `486`, `d = 128`, `hq = 16`,
   `h_kv = 8`, every head with different values, at both `valid = None` and (if Option 2) `valid < t`.
2. The real forward taps and both logit rows within the current thresholds (98.8% / 100%,
   max 0.0545 / 0.0389).
3. The real decode: determinism, the frame count, and the token shares inside the existing
   `+5 pp` bound of the engine's own dtype gap.
4. A `gpu-bench` row at the production shape, before and after, in the same process policy.
5. A Turkish lesson (the 7 parts) and the README index entry.

## 7. Expected result and the stop rule

Attention 881.7 us per layer; a flash path that reaches 400-500 us saves 10-13 ms per step,
0.3-0.4 s per decode: **2.15-2.25 s**, which meets criterion 3. If the first correct kernel is
not faster than the current one at the production shape, stop and keep the current path (D2's
lesson: the FLOP argument did not survive the clock).

## 8. Open questions for the supervisor

1. Option 1 or Option 2 (scope of the padding)?
2. Is a 64-row query tile the right first attempt for 128-wide keys on 20 SMs, or 32 rows to keep
   the register pressure down?
3. Should the `scores` scratch stay in the plan for the padded path only, or be removed once the
   flash path is the production one?
4. Does the lesson go as a new lesson 13, or as an extension of lesson 5 (the d64 flash)?
