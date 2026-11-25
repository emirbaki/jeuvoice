# Rust CUDA Phase 2: audit implementation handoff (2026-11-25)

## 1. Assignment and state

Implement the findings of the read-only audit. Revise the completed work only. Do not implement Task 10 product sampling or Task 12's `render_tokens` tool and final phase review. The audit's measurement and optimization work packages need explicit user selection before implementation.

Read these files first:

- `AGENTS.md`
- `docs/superpowers/plans/2026-11-25-rust-cuda-phase2-omnivoice.md`
- `docs/superpowers/handoff/2026-11-25-rust-cuda-phase2-task7-handoff.md`
- `.superpowers/sdd/phase2/progress.md`

The audit used branch `rust-cuda-phase2` at commit `651e2bb`. The tree was clean. Check the current branch, commit and tree before any edit. If the code changed, map each finding to the new code instead of applying the old text blindly.

The audit changed no project files and ran no GPU tests. Its runtime savings are estimates unless a measured value is stated.

### Rules for the implementation agent

1. Write the failing test before each code fix. Run it and watch it fail.
2. Make the smallest change that fixes the finding. Keep the existing coding style.
3. Keep kernels in `jeuvoice-gpu`. Use Rust kernels only. Keep the `sm_75` target.
4. Keep fp16 data, fp32 accumulation, the f32 residual stream and f32 norms.
5. Add a `SAFETY` comment to every new or changed `unsafe` block.
6. Run `jeuvoice-gpu` and `jeuvoice-whisper --features cuda-kernels` tests after a kernel change.
7. Run one real-model test file per process with `--test-threads 1`.
8. Measure performance before and after each performance change. Do not estimate a speedup and record it as a measurement.
9. Keep comments and docs in English. Keep CUDA lessons in Turkish. Save UTF-8 without a BOM.
10. Do not commit, merge, rebase or push without explicit user approval.

### Rulings to preserve

- Keep the Qwen3 half-split RoPE pair `(i, i + d/2)`. Do not use the interleaved pair.
- Keep the audio tokenizer on the CPU. The measured GPU attempt took 202 seconds.
- Do not add a conventional KV cache. Bidirectional attention changes the hidden states of the static prefix when target tokens change.
- Do not switch all weights to f32. This breaks the VRAM goal and does not guarantee the engine's fp16 token trajectory.
- Do not restate criterion 1 without user approval. The deterministic trajectory is chaotic under arithmetic changes.
- Do not change the app UI.

## 2. Work package A: memory safety and input validation

### A1. Add tail guards to four kernels

**Evidence**

- `crates/jeuvoice-gpu/src/ops/rope.rs:65-74` launches rounded-up element grids for `rope_half_f16` and `rope_f16`.
- `crates/jeuvoice-gpu/src/kernels/mod.rs:1579-1621` has no tail guard in either kernel.
- `crates/jeuvoice-gpu/src/ops/heads.rs:20-36` launches rounded-up grids for `to_heads_f16` and `from_heads_f16`.
- `crates/jeuvoice-gpu/src/kernels/mod.rs:1914-1947` has no tail guard in either head-layout kernel.
- `LaunchConfig::for_num_elems` rounds up to 256-thread blocks. The existing tests do not detect accesses outside the logical tensors.
- The RoPE test uses 960 work items, so 64 of 1024 threads are invalid.
- The head-layout test uses 120 work items, so 136 of 256 threads are invalid.

The fixed OmniVoice production shapes are multiples of 256. The public operations and their small test shapes are not safe.

**Tests to write first**

- Use the existing non-multiple-of-256 shapes.
- Put a canary tensor directly after each destination tensor in the arena plan.
- Fill the canary with a known value and assert that every value stays unchanged.
- Cover both RoPE pair layouts and both head-layout directions.
- Keep the existing numerical assertions.

**Implementation**

- Pass the exact work-item count to each kernel.
- For RoPE, use `n = rows * heads * (d / 2)`.
- For the head-layout kernels, use `n = src.numel()`.
- Add `if i >= n { return; }` before any index calculation, read or write.
- Update the kernel safety text and the host `SAFETY` comments.

**Acceptance**

- The new canary tests fail before the fix and pass after it.
- All existing GPU tests pass.
- No neighboring tensor changes after a launch.

### A2. Validate forward IDs, masks and valid lengths

**Evidence**

- `crates/jeuvoice-omnivoice/src/forward.rs:54-63` checks only the ID and mask lengths.
- `embed_mixed_f16` in `crates/jeuvoice-gpu/src/kernels/mod.rs:1789-1820` uses IDs as table indices without a range check.
- A negative ID becomes a large `usize`. An oversized ID reads outside the table.
- `forward.rs:81-85` panics when `valid` has the wrong slice length.
- Negative valid lengths and lengths above `t` can disable the intended padding behavior.

**Tests to write first**

Factor the host validation into a pure helper. Test these cases without a GPU:

- A negative text ID.
- A text ID equal to or above `cfg.vocab_size`.
- A negative audio ID.
- An audio ID above the mask ID.
- A mask value other than 0 or 1.
- A `valid` slice with the wrong row count.
- A negative valid length.
- A valid length above `t`.
- Valid boundary values, including 0 and `t`.

**Implementation**

- Validate before every device upload.
- At text positions, validate the text IDs against `0..cfg.vocab_size`.
- At audio positions, validate every codebook ID against `0..cfg.audio_vocab_size`.
- Require `valid.len() == rows` and every valid length to be in `0..=t`.
- Return `TtsError::Render` with the row, position and bad value. Do not panic on user input.

**Acceptance**

- Invalid input returns an actionable error before any kernel launch.
- The current fixtures and decode tests keep their previous results.

### A3. Fix the impossible activation-shape assertions

**Evidence**

- `crates/jeuvoice-omnivoice/src/weights.rs:81-95` plans `x` and `logits` as 2-D tensors.
- `crates/jeuvoice-omnivoice/tests/weights.rs:60-63` expects 3-D tensors.

**Implementation**

Change only the test expectations:

- `x`: `[BATCH * T_MAX, d]`
- `logits`: `[BATCH * T_MAX, cfg.audio_head_rows()]`

Do not make the production tensors 3-D. The GEMM path uses the 2-D layout.

**Acceptance**

Run `the_weights_load_and_fit_the_plan` alone. It must pass after the real load.

### A4. Reject incomplete or malformed adapters

**Evidence**

In `crates/jeuvoice-omnivoice/src/weights.rs:163-180`:

- A missing replacement table silently keeps the base table.
- A missing LoRA A tensor silently skips the merge, including when B exists.
- Replacement validation compares only element counts, not shapes.

In `weights.rs:244-255`, `r = 0` reaches an assertion instead of returning `ModelLoad`.

**Tests to write first**

Use small synthetic indexes where possible. Cover:

- A missing LoRA A tensor.
- A missing LoRA B tensor.
- A missing `audio_embeddings` replacement.
- A missing `audio_heads` replacement.
- A replacement with the same element count but the wrong shape.
- `r = 0`.
- A valid minimal adapter.

**Implementation**

- Preflight required base and adapter names and shapes before arena allocation.
- Require the 196 LoRA pairs and both replacement tables when an adapter is supplied.
- Require `r > 0`.
- Return `TtsError::ModelLoad` with the tensor or config key.
- Keep `adapter = None` as the valid base-only mode.

**Acceptance**

- Incomplete adapters fail before allocation and before any silent base fallback.
- The real adapter still loads and produces the same weights.

## 3. Work package B: behavior and test quality

### B1. Match the engine's nonverbal-tag pattern

**Evidence**

- `crates/jeuvoice-omnivoice/src/text.rs:214-229` splits every `[a-z0-9-]` tag.
- The engine splits only the explicit `_NONVERBAL_PATTERN` alternatives at `engine/.venv/Lib/site-packages/omnivoice/models/omnivoice.py:1651-1655`.

Unknown text such as `[foo]` can therefore produce different tokens.

**Implementation and tests**

- Use the engine's exact tag alternation.
- Add engine-generated vectors for every supported tag.
- Add an unknown tag with surrounding spaces and punctuation.
- Assert that the unknown tag follows the engine's ordinary tokenization path.

### B2. Stop silent text loss in BPE encoding

**Evidence**

- `text.rs:137-140` skips regex iterator errors.
- `text.rs:161-164` drops a BPE part that has no vocab ID.

**Implementation and tests**

- Make the encoding path return a `Result`.
- Propagate regex and vocab failures with the offending segment.
- Add a synthetic incomplete-tokenizer test.
- Keep the real tokenizer output unchanged on all existing vectors.

### B3. Restore the commented-out duration assertion

`crates/jeuvoice-omnivoice/src/duration.rs:107` contains literal `\n` text inside a `//` comment. Split the comments and the `assert_eq!` onto real separate lines. Run the plain duration tests.

### B4. Test guidance directly

**Evidence**

- `guided_peak` at `crates/jeuvoice-omnivoice/src/decode.rs:94-123` has no direct unit test.
- `tests/decode.rs` tests only the schedule.

**Tests to write first**

Use an independent f64 implementation of the engine's formula:

1. `log_softmax(c)`
2. `log_softmax(u)`
3. `log_softmax(lp_c + guidance * (lp_c - lp_u))`
4. Set the mask logit to negative infinity after the outer normalization.

Cover:

- `guidance = 0` and `guidance = 2.5`.
- A case where the mask logit is the maximum before masking.
- A case that proves the mask stays in the outer normalizer.
- Large logits that require stable log-sum-exp.
- The selected token and the confidence value.

Keep the simplified production formula unless a test proves a numerical problem.

### B5. Strengthen the real-model assertions

- In `tests/decode_gpu.rs:86-103`, keep the current fp16 bound and add `ours_f32 + 0.05 >= engine` as a regression bound. Do not change the success criterion without user approval.
- In `tests/forward.rs:127-131`, assert the exact expected tap-name set. Do not skip a missing reference silently.
- In `tests/forward.rs:184-205`, assert the unconditioned maximum logit error, not only argmax agreement.
- Keep deterministic repeatability on s1.

## 4. Work package C: establish a reliable performance baseline

Complete this package before ranking optimization work by measured time.

### C1. Fix pass and decode timing

**Evidence**

- `tests/forward.rs:65-69` uses a wall clock around asynchronous launches.
- The next pass starts with synchronizing uploads, so its timer includes previous GPU work and current enqueue work.
- `tests/decode_gpu.rs:104-108` runs s1 twice but reports no warm decode time.
- The first pass includes about 656 ms of module and cuBLAS setup.

**Implementation**

- Use CUDA events for GPU work.
- Use an explicit stream synchronization before and after wall-clock measurements.
- Report cold and warm decode times separately.
- Keep host time and GPU time separate.
- Record the first-pass setup cost separately.

The 4.6 second decode measurement remains useful because each logit download synchronizes the stream. Treat the isolated 112 ms pass value as approximate until the new measurement exists.

### C2. Benchmark production shapes

The current attention benchmark at `crates/jeuvoice-gpu/src/bin/gpu-bench.rs:514-540` uses `valid = None`. The decode uses valid lengths `[486, 214]`.

Add or record these shapes:

- Bidirectional attention without padding, as the existing baseline.
- Bidirectional attention with `valid = [486, 214]`.
- SwiGLU with `[972, 3072]`, not only `[486, 3072]`.
- Q projection: `M=972, N=2048, K=1024`.
- K and V projections: `M=972, N=1024, K=1024`.
- O projection: `M=972, N=1024, K=2048`.
- Gate and up projections: `M=972, N=3072, K=1024`.
- Down projection: `M=972, N=1024, K=3072`.
- Audio head with f32 output: `M=972, N=8200, K=1024`.

Run one variant per process when cache or clock drift affects the result. Use the same warmup and iteration policy for before-and-after comparisons.

### C3. Record the baseline

Use a table with these rows:

- Cold forward pass and warm forward pass.
- Cold s1 decode and warm s1 decode.
- Attention with and without the production padding mask.
- Every production GEMM shape.
- Load time and arena bytes.

## 5. Work package D: low-risk performance revisions

Implement one revision at a time. Run numerical tests and benchmarks after each revision. Revert a revision that loses speed or violates a quality threshold.

### D1. Remove K/V repetition with grouped attention GEMMs

**Evidence**

`attention_bidir` at `crates/jeuvoice-gpu/src/ops/attention.rs:165-190` expands 8 K/V heads to 16 before each batched GEMM.

Two adjacent query heads share one K/V head. Their query data is already contiguous as `[2*T, 128]`.

**Exact transformation**

For `Q * K^T`:

- Use `batch = B * 8` instead of `B * 16`.
- Use `M = 2*T`, `N = T`, `K = 128`.
- Use Q stride `2*T*128`.
- Use K stride `T*128`.
- Use score stride `2*T*T`.

For `probabilities * V`:

- Use `batch = B * 8`.
- Use `M = 2*T`, `N = 128`, `K = T`.
- Use score stride `2*T*T`.
- Use V stride `T*128`.
- Use output stride `2*T*128`.

The score and output layouts remain `[B, 16, T, ...]`. The existing softmax and padding logic can stay unchanged.

**Expected benefit, not a measured speedup**

- No `repeat_heads` calls.
- No `kx` or `vx` scratch traffic.
- About 11.4 MiB less raw repeat read/write traffic per layer at the s1 shape.
- Half as many batched GEMM instances, with larger M.

**Tests and acceptance**

- Keep the CPU f64 attention references, GQA mapping and padding tests.
- Add a two-group case with different values in every head.
- Keep real-model hidden and logit comparisons within their current thresholds.
- Remove `kx` and `vx` from the activation plan only after the new path passes.
- Benchmark the padded and unpadded production shapes.

### D2. Compute the final norm and audio head only for used rows

**Evidence**

- `forward.rs:154-159` processes all 972 positions.
- The decode uses 214 conditioned target positions and 214 unconditioned target positions.
- In the current padded layout, these rows form one contiguous range: `target .. target + 2*target_len`.

**Implementation**

- Add a decode-specific head-range path. Do not change the general forward tap behavior.
- Apply the final RMSNorm only to the selected rows.
- Write compact logits starting at row zero. This preserves alignment and simplifies downloads.
- Keep conditioned rows first and unconditioned rows second in the compact output.
- Update the decode indexing in the same change.

**Expected benefit, not a measured speedup**

At s1, the head processes 428 rows instead of 972. The audio-head work falls from 16.32 to 7.19 GFLOPs per pass. The head weight still needs to be read, so the end-to-end head speedup can be smaller.

**Acceptance**

- Conditioned and unconditioned logits match the full-head path within the existing numerical thresholds.
- Deterministic tokens do not regress on s1, s2 or s3.

### D3. Accumulate residual GEMMs directly into f32

**Evidence**

- `forward.rs:140-149` writes `o_proj` and `down_proj` to f16 `proj`, then calls `add_into_f32`.
- `Blas::gemm_f16_f32out` already supports `beta = 1`.

**Implementation**

- Accumulate `o_proj` directly into `x` with f32 output and `beta = 1`.
- Do the same for `down_proj`.
- Remove `proj` only if no other path uses it.

**Expected benefit, not a measured speedup**

- Remove 56 residual-add launches per pass.
- Remove the f16 rounding of each projection before the residual addition.
- Reduce temporary memory traffic.

**Acceptance**

- Compare every hidden tap and both logit rows against the f32 fixtures.
- Benchmark the two GEMM shapes with f16 output plus addition against direct f32 accumulation.
- Keep the change only if the complete pass does not lose speed.

### D4. Remove exact redundant decode work

These changes must not alter the selected tokens:

- Skip `guided_peak` when `tokens[index] != mask_id`. The s1 schedule permits 16.27% fewer calls.
- Skip the forward pass when `counts[step] == 0`. This matters most for short targets.
- Before GPU guidance exists, download the adjacent conditioned and unconditioned ranges in one call. The byte count stays the same, but one synchronization goes away.

Add tests for a short target whose schedule has zero-count steps. Compare all three deterministic token fixtures after the change.

### D5. Use shared RoPE only with the correct layout

**Evidence**

- Lesson 10 measures 117.9 microseconds for naive half-split RoPE and 98.9 microseconds for the shared version.
- `forward.rs:123-133` currently transposes to head-major layout and then flattens Q/K to 2-D.
- A direct switch to `HalfSplitShared` on that 2-D input gives the kernel `heads = 1`. It does not provide the measured cross-head sharing.

**Implementation**

- Apply per-head norm and RoPE to token-major `q_row` and `k_row` before `to_heads`.
- Pass 3-D `[positions, heads, head_dim]` views and one position per sequence position.
- Keep the V transpose unchanged.
- Cache constant position metadata if this removes repeated uploads.

**Acceptance**

- Prove that the shared kernel sees 16 Q heads and 8 K heads per position.
- Compare hidden taps and logits against the current path.
- Benchmark the actual Q and K shapes. Keep the faster correct path.

### D6. Consider fused projection weights only after measurement

Preconcatenating Q/K/V and gate/up weights can remove 84 GEMM calls per pass. FLOPs stay unchanged. Larger N dimensions can help, but this is not guaranteed.

A combined QKV output needs stride-aware head-layout conversion. A combined gate/up output needs stride-aware SwiGLU input. Implement these layout changes with tests before removing the separate weight tensors.

## 6. Work package E: structural performance revisions

These changes are larger. Do not start them before Work packages A to C are complete.

### E1. Remove padded unconditioned computation

At s1, the conditioned row has 486 real positions. The unconditioned row has 214 real positions and 272 padded positions. Padding changes attention values but does not reduce GEMM or softmax work.

A ragged pass needs only `486 + 214 = 700` positions instead of 972:

- 28.0% fewer linear-projection FLOPs.
- 40.3% less attention QK/PV work.

These are arithmetic reductions, not measured time savings.

**Practical experiment**

The existing API can run one conditioned pass of length 486 and one unconditioned pass of length 214. Compare the real rows with the packed path. This experiment requires no new kernel.

Two passes reread projection weights and double many launches. Do not assume that the experiment wins. The preferred final design uses 700-row linear GEMMs and two attention segments without repeating weights.

**Acceptance**

- Add a packed-versus-split numerical comparison for both rows.
- Preserve RoPE positions, including the unconditioned row starting at position zero.
- Measure the complete decode, not only FLOPs.

### E2. Move guided scoring to the GPU

**Evidence**

- The CPU guided-peak path costs about 1 second on s1.
- It evaluates 54,784 rows and about 56.15 million exponentials per decode.
- The current downloads move about 13.39 MiB per step.
- Scores and picks need only about 13.4 KiB per step.

**Implementation**

- Compute the guided logits, log-sum-exp, masked argmax and confidence on the GPU.
- Return `scores: [codebooks * target_len] f32` and `picks: [codebooks * target_len] i32`.
- Keep the schedule and top-k bookkeeping on the CPU initially.
- Reuse the pure guidance tests from Work package B.
- Add a GPU-versus-CPU f32 reference test with the mask as the maximum logit.
- Add the required Turkish lesson and `gpu-bench` row for the new kernel.

**Acceptance**

- Scores and picks match the CPU reference within a stated tolerance.
- Deterministic token shares stay within the agreed bounds.
- Measure host transfer bytes, kernel time and complete decode time.

### E3. Add d128 bidirectional GQA FlashTC only if profiling justifies it

The current attention materializes f16 scores, applies a separate mask and softmax, then performs a second GEMM. The bench projection is about 1.0 to 1.14 seconds of attention per 32-step render. This projection omits the production padding mask.

A d128 tensor-core flash path can remove score traffic and K/V repetition. The existing d64 speedup does not guarantee a d128 speedup.

Required work:

- Support bidirectional attention, GQA and the diagonal-only padding rule.
- Test against the CPU f64 reference at non-multiple tile sizes.
- Include large-score tests and real-model tap comparisons.
- Add the required Turkish lesson and benchmark rows.
- Measure the complete decode against the repeat-free grouped GEMM path.

### E4. Harden attention-score scaling

**Evidence**

- `attention_bidir` stores unscaled QK results in f16 and applies the scale during softmax.
- The older naive attention path applies the scale inside the GEMM to avoid f16 overflow.

**Implementation and tests**

- Add a large-score test before the change.
- Apply `1/sqrt(d)` as the QK GEMM alpha.
- Use a softmax scale of 1.0.
- Compare against the f64 reference and the real-model taps.

This is primarily numerical hardening. Treat any speed change as a benchmark result.

## 7. Documentation fixes to include with the code work

- Update the previous handoff's last commit when the branch advances. It names `1e7dcc5`, while the audit started at `651e2bb`.
- Correct the plan's time-shift formula to `t_shift*t/(1+(t_shift-1)*t)`. The implementation in `decode.rs:53-59` already matches the engine.
- Update lesson 11 and the SwiGLU benchmark text to distinguish the one-row benchmark from the production `[972,3072]` shape.
- Keep lesson 10's measured shared-RoPE claim tied to token-major input.
- Record every before-and-after measurement in `docs/development-log.md` and `.superpowers/sdd/phase2/progress.md`.
- Write a new handoff with the final commit, test counts, numerical results, timings and remaining findings.
- Update `AGENTS.md` if the work changes any command, structure or rule recorded there.

## 8. Verification commands

Run plain tests after every work package:

```powershell
scripts\cargo.cmd test
```

Run the GPU suites after a kernel change:

```powershell
scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels
```

Run the real OmniVoice tests one file per process:

```powershell
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test decode_gpu -- --ignored --nocapture --test-threads 1
```

Run each real weight test alone when both weight arenas would exceed the card:

```powershell
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test weights -- --ignored --nocapture --test-threads 1 the_weights_load_and_fit_the_plan
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test weights -- --ignored --nocapture --test-threads 1 the_adapter_changes_the_base_weights
```

Build the benchmark after a kernel or operation change:

```powershell
scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
```

Put `%CUDA_PATH%\bin\x64` on `PATH` before running `target\release\gpu-bench.exe`. Do not run two real-model GPU test processes at the same time.

## 9. Definition of done

- All memory-safety findings have regression tests.
- Invalid user input returns actionable errors instead of panics or device faults.
- The real weight tests use the actual tensor shapes.
- Adapter and tokenizer failures cannot silently select base weights or drop text.
- Guidance, taps and both logit rows have direct assertions.
- Cold and warm performance numbers use reliable synchronization.
- Every performance revision has same-shape before-and-after measurements.
- Real-model numerical quality stays within the recorded thresholds.
- All required plain, GPU and real-model tests pass.
- Benchmarks, lessons, the development log, the ledger and the handoff match the code.
- Task 10 product sampling and Task 12's `render_tokens` tool and final phase review remain untouched unless the user explicitly expands the scope.
