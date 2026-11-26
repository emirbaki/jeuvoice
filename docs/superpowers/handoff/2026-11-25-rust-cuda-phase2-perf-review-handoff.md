# Rust CUDA Phase 2: performance review implementation handoff (2026-11-25)

## 1. Assignment and state

This handoff turns the E3 supervisor review into work packages for sub-agents. The review found a larger and cheaper win than the proposed d128 flash attention, and it found errors in the E3 plan. Read the review context in section 3 before any code.

Read these files first:

- `AGENTS.md`
- `docs/superpowers/handoff/2026-11-25-e3-flash-attention-plan.md` (the plan under review)
- `docs/superpowers/plans/2026-11-25-rust-cuda-phase2-omnivoice.md`
- `.superpowers/sdd/phase2/progress.md`

State at the time of this handoff:

- Branch `rust-cuda-phase2`, HEAD `4645a26` (`feat(tts): run the decode in two passes`). Tree clean.
- `main` is at `e84f780` (the Phase 2 merge).
- The decode runs two passes per step: conditioned `rows=1, t=seq` and unconditioned `rows=1, t=target_len`.
- Recorded warm decode: s1 2.55 s, s2 2.7 s, s3 2.5 s (deterministic mode). Step 77.4 ms.
- Tests: 455 plain, 114 GPU, 70 whisper GPU.

Check the current branch, commit and tree before any edit. If the code moved, map each finding to the new code. Do not apply old line numbers blindly.

## 2. How to delegate

- One writer per tree. This repo had concurrent-writer corruption twice. Do not run two sub-agents on the same checkout at once.
- Prefer sequential packages on this branch. If you parallelize, give each sub-agent its own worktree and merge in the order below.
- Packages P3 and P4 both touch `weights.rs` and `forward.rs`. Sequence them, do not parallelize them.
- P1 touches `decode.rs`. P5 touches `kernels/mod.rs` and `ops/attention.rs`. These are independent of P3/P4.
- Each sub-agent gets: this file, the one package it owns, and the rules in section 4. It reports using the template in section 9.

Recommended order: agree the P2 protocol, then P1, then P3 and P4, then P5 only if still needed.

## 3. Review context that every sub-agent must know

### 3.1 The criterion-3 scope problem

The engine baseline of 4.6 s for the token stage is PRODUCT mode. `scripts/tts-baseline.ps1` calls `--bench`, and `bench()` in `tools/reference/omnivoice_reference.py` runs with the product config (`position_temperature=5`). The recorded 2.55 s is DETERMINISTIC mode. These are not the same measurement.

The supervisor or user must decide which mode criterion 3 targets. Do not silently change the criterion. P1 makes the product path fast so a product-vs-product comparison is possible.

### 3.2 Product mode reintroduces the CPU bottleneck

- Deterministic branch: `crates/jeuvoice-omnivoice/src/decode.rs:315-348` uses the GPU `guide_peak` and downloads about 13.4 KiB per step.
- Product branch: `decode.rs:284-313` downloads the full conditioned and unconditioned logit planes (about 13.4 MiB per step) and runs `sample_row` on the CPU.

Measured in one session, same context: deterministic steady step about 116 ms, product steady step about 150 ms. Product cold 5.82 s, deterministic cold 4.92 s. The extra cost is the full-plane download plus CPU sampling.

For the app and engine config, `class_temperature = 0` (variety 0). In that case the product token is the argmax and the confidence is the unfiltered log-softmax peak. `guide_peak` already returns exactly these two values. So variety-0 product can reuse the deterministic GPU path and only add cheap position Gumbel noise on the host.

### 3.3 The measurement is clock-sensitive

The 2.55 s warm number reproduces only under sustained warm load (the `decode_gpu` test runs s1 cold then warm, then s2 and s3). A single cold render gave about 4.9 s even near boost clock. The GPU is a laptop part and its clock varies with load and power state. Any before/after comparison must fix the warmup and record the clock. See P2.

### 3.4 The E3 plan budget is wrong

The plan cites attention at `[2,16,486,128]` = 881.7 us per layer = 24.7 ms per step. Production now runs two passes: conditioned `[1,16,486,128]` and unconditioned `[1,16,214,128]`. Attention work scales with `b * t^2`. The conditioned pass is 0.5 of the benchmark, the unconditioned pass is `0.5 * (214/486)^2 = 0.097`. Total about 0.60. Real attention is about 15 ms per step, and the score plane is about 9 MB, not 15 MB. The projected 0.3-0.4 s saving is really about 0.15-0.2 s (estimate). P5 must use the real shapes.

## 4. Rules for every sub-agent

1. Write the failing test first. Run it and watch it fail. Then write the code.
2. Make the smallest change that fixes the finding. Keep the surrounding style.
3. Kernels live only in `jeuvoice-gpu`, in Rust (`#[cuda_module]`, `#[kernel]`). No CUDA C++. Target `sm_75`.
4. fp16 data, fp32 accumulate. Norm and softmax math in fp32. Keep the f32 residual stream.
5. Every new or changed `unsafe` block gets a `SAFETY` comment that says why it is safe.
6. After a change in `jeuvoice-gpu`, run the tests of `jeuvoice-gpu` and of `jeuvoice-whisper --features cuda-kernels`.
7. Run one real-model test file per process with `--test-threads 1`. The card has 4 GB.
8. Measure, do not guess. Use the P2 protocol for every timing. Record the clock.
9. Comments and docs in English, ASD-STE100 style. CUDA lessons in Turkish. Save UTF-8 without a BOM.
10. Every new kernel gets a Turkish lesson with the 7 parts and a `gpu-bench` row.
11. Do not commit, merge, rebase or push without explicit user approval.
12. Do not change the app UI.

## 5. Rulings to preserve

- Keep the Qwen3 half-split RoPE pair `(i, i + d/2)`. Do not use the interleaved pair.
- Keep the audio tokenizer on the CPU.
- Do not add a conventional KV cache. Bidirectional attention changes the static prefix when target tokens change.
- Do not switch all weights to f32.
- Keep the two-pass decode. It is measured faster than the packed pass.
- Do not restate criterion 1 or criterion 3 without user approval.
- The grouped attention uses `kv_batch = b * h_kv` for the GEMMs and `score_batch = b * hq` for the padding mask. Do not merge these two counts again (the D1 bug).

## 6. Work package P1: GPU product sampling for variety 0

**Goal.** Variety-0 product mode uses the GPU `guide_peak` path, not the full-plane CPU path. This removes the about 13.4 MiB per step download and the CPU `sample_row` cost.

**Evidence.** `decode.rs:284-313` (product, CPU) versus `decode.rs:315-348` (deterministic, GPU). `guide_peak` signature in `crates/jeuvoice-gpu/src/ops/guide.rs:14-27`. It returns `conf` (unfiltered log-softmax peak) and `pick` (argmax token) per pair.

**Key fact.** When `class_temperature <= 0`, the product token equals the argmax and the confidence equals the unfiltered peak. These are exactly `guide_peak`'s outputs. The only product addition is position Gumbel noise on the score.

**Tests to write first.**

- Pure: with `position_temperature = 0` and `class_temperature = 0`, product mode gives exactly the deterministic tokens.
- Pure: the position-noise score transform is `score / position_temperature + gumbel`, and it is repeatable for one seed.
- GPU: variety-0 product keeps the frame count (214 for s1), is repeatable for one seed, and every token is inside the vocabulary.
- GPU: variety-0 product no longer downloads the full planes. Assert the steady step time is close to the deterministic step time under the P2 protocol.

**Implementation.**

- Factor the deterministic `guide_peak` + conf/pick download + score/pick fill into a helper.
- Route variety-0 product (`class_temperature <= 0`) through that helper.
- Apply position Gumbel noise on the host to the score, in a fixed order, only when `position_temperature > 0`. Keep it seed-repeatable. Document the draw order.
- Keep the `class_temperature > 0` branch on the existing CPU path. GPU top-k token sampling is out of scope here.
- Do not change the deterministic path's results.

**Acceptance.**

- Variety-0 product warm time is close to deterministic warm time (measure with P2).
- `position_temperature = 0` product equals deterministic tokens exactly.
- Repeatability, frame count and vocab-range tests pass.
- All existing deterministic tests still pass.

## 7. Work package P2: a fixed measurement protocol

**Goal.** Make every timing reproducible and comparable, and record the clock.

**Evidence.** Warm s1 2.57 s in the `decode_gpu` test, but about 4.9 s in a single cold render, same session. The GPU clock ranged from 570 MHz idle to about 1965 MHz under load.

**Implementation.**

- Define the authoritative criterion-3 measurement: which mode, warm, how many runs, and median or best. Put it in `docs/development-log.md`. The mode choice needs the user or supervisor (section 3.1).
- Add warm timing to `examples/render_tokens.rs`: run one or more warmup renders, then report the warm render time. Keep the cold time separate.
- Record `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu` next to each reported timing.
- Use `bench::time_on_stream` for kernel timings and an explicit stream synchronize for wall-clock pass timings.

**Acceptance.**

- The same protocol reproduces the warm decode within a stated tolerance across two runs.
- Every reported number carries its mode, warm/cold label and clock.

## 8. Work packages P3, P4, P5

### P3: fused QKV and gate/up GEMM (D6)

**Goal.** Fewer, larger GEMMs. Preconcatenate Q/K/V weights into one `N=4096` matrix and gate/up into one `N=6144` matrix at load time. This removes 84 GEMM calls per pass and helps the small-M unconditioned pass (214 rows).

**Evidence.** `forward.rs:121-123` (three separate Q/K/V GEMMs) and `forward.rs:148-149` (gate and up). The cuBLAS wrapper already handles row-major via operation flags (`blas.rs:6-10`).

**Implementation notes.**

- Concatenate the merged weights once in `OmniVoiceWeights::load`. No runtime transpose.
- The combined QKV output needs a stride-aware head-layout conversion. The combined gate/up output needs a stride-aware SwiGLU input. Write these with tests before removing the separate tensors.
- Benchmark at the real shapes: `M=486` and `M=214`, not only `M=972`.

**Tests and acceptance.** Numerical equivalence against the separate-GEMM path within the existing tap thresholds. A measured speedup at the real shapes under P2, or revert. Keep the change only if the complete step does not lose speed.

### P4: metadata cache and unused scratch cleanup

**Goal.** Remove repeated host work and dead memory.

**Evidence.**

- `forward.rs:90-103` rebuilds and uploads the position vectors every pass, with a synchronize each.
- `weights.rs:105-106` still allocates `kx` and `vx`. The grouped attention has not read them since D1. About 16 MB.

**Implementation.**

- Cache the position metadata per sequence shape, or derive positions inside the RoPE kernel. Upload once per shape, not per pass.
- Remove `kx` and `vx` from `Activations` and from `BidirScratch`. Update `attention_bidir`, its tests and the bench. Keep `scores` for the padded path.

**Tests and acceptance.** A canary or shape test proves the position cache is correct for both `t=486` and `t=214`. The arena drops by about 16 MB. All attention and forward tests pass. No speed regression.

### P5: d128 bidirectional flash attention (E3), conditional

**Start only after P1 and P2, and only if criterion 3 still needs it.** Use the corrected facts below, not the original plan's numbers.

**Corrected design facts.**

- Production shapes are `b=1, t=486` and `b=1, t=214`, `hq=16`, `h_kv=8`, `d=128`. Both have a tail: `486 = 7*64 + 38`, `214 = 3*64 + 22`. A tail guard is mandatory.
- sm_75 f16 MMA is `m16n8k8`, not `m16n8k16`. The existing kernel uses `mma_m16n8k8_f32_f16` (`kernels/mod.rs:2716`). A `d=128` contraction needs 16 k-steps, not 4.
- Register budget: the d64 kernel keeps about 84 registers per thread (qa 16, o 32, s 32). d128 doubles the Q fragments and the O accumulators, about 130+ registers per thread. K+V shared is `64*128` f16 = 32 KB per block. On sm_75 (64 KB shared, 64K registers per SM) this is about 2 blocks per SM, about 25% occupancy. The d64 kernel already runs at 15-20% occupancy. The 400-500 us target is optimistic.
- Online softmax is algebraically equivalent, not bit-exact. P still rounds to f16 for the PV product.

**Scope.** Option 1: flash only when `valid = None`. Keep the grouped path for padded callers. Make the scratch optional for the flash kind.

**Method.**

- Prototype one query-tile size first. The lever is rows per warp and register pressure, not only block size. Try 16 rows per warp (current) and 8. If registers spill, consider splitting d128 into two d64 halves.
- Check the PTX for `ld.local` and `st.local`. They must be zero. Lesson 5 records a 7.5x slowdown from local-memory spill; use the `unroll!` macro for all fragment and accumulator arrays.
- Stop rule: if the first correct kernel is not faster than the grouped path at the real `b=1` shapes, stop and keep the grouped path.

**Tests and acceptance.**

- CPU f64 reference at `t=1, 37, 214, 486`, `d=128`, `hq=16`, `h_kv=8`, every head different, first and last tile compared.
- Real forward taps and both logit rows within the current thresholds.
- Real decode determinism, frame count and token shares within the existing bound.
- A `gpu-bench` row at the real shapes, before and after, under P2.
- A new Turkish lesson 13 with the 7 parts, cross-linked to lesson 5, and a README index entry.

## 9. Verification commands

Plain tests after every package:

```powershell
scripts\cargo.cmd test
```

GPU suites after a kernel change:

```powershell
scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels
```

Real OmniVoice tests, one file per process:

```powershell
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test decode_gpu -- --ignored --nocapture --test-threads 1
```

Build and run the benchmark:

```powershell
scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
```

Put `%CUDA_PATH%\bin\x64` on `PATH` before running `target\release\gpu-bench.exe`. Do not run two real-model GPU test processes at once.

## 10. Sub-agent report template

Each sub-agent reports:

- Package id and the commit or diff it produced (no commit without approval).
- The failing test it wrote first, and the failure output.
- The change, with file and function references.
- Before/after numbers under the P2 protocol, with mode, warm/cold and clock.
- Test counts: plain, GPU, whisper GPU, and the real-model results.
- Any ruling it had to apply, and anything it deferred.

## 11. Definition of done

- Variety-0 product mode runs on the GPU path and is close to deterministic speed.
- The criterion-3 mode is decided by the user, and the measurement protocol is fixed and documented.
- Every kept performance change has a same-shape before/after measurement under the protocol.
- Real-model numerical quality stays within the recorded thresholds.
- All required plain, GPU and real-model tests pass.
- Dead `kx`/`vx` scratch is gone; position metadata is not re-uploaded every pass.
- If P5 ran, its lesson, bench row and stop-rule decision are recorded.
- Benchmarks, lessons, the development log, the ledger and a new handoff match the code.
