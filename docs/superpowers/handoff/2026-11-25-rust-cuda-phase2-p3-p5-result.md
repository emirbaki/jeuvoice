# Rust CUDA Phase 2: P3, P4 and P5 result handoff (2026-11-25)

This file is the result of the P3, P4 and P5 work packages of
`docs/superpowers/handoff/2026-11-25-rust-cuda-phase2-perf-review-handoff.md`. Read that file
first for the findings and the rules; this one records what was done, what was measured and what
is still open.

## 1. State

- Branch `rust-cuda-phase2`. Last commit `a885f9b` (P1 and the P2 protocol).
- **The P3, P4 and P5 work is uncommitted** in this checkout. Nothing was committed, merged or
  pushed: rule 11 of the review handoff needs the user's approval.
- Plain suite 456 pass. GPU suite 121 pass. Whisper `cuda-kernels` 70 pass. The real-model files
  `forward` (3) and `decode_gpu` (2) pass, one process at a time.
- Arena 1460 MB (weights 1225 MB, activations 235 MB). The card has 4 GB.

## 2. What changed

### P3: fused QKV and gate/up (files: `weights.rs`, `forward.rs`, `ops/heads.rs`, `ops/swiglu.rs`, `kernels/mod.rs`)

- The load builds `llm.layers.N.self_attn.qkv_proj.weight` `[4096, 1024]` (q | k | v) and
  `llm.layers.N.mlp.gate_up_proj.weight` `[6144, 1024]` (gate | up). Each part goes through its
  own LoRA merge in f32 first (`read_merged`), then the parts are concatenated. The parts are not
  uploaded a second time: `fused_group`, `fused_part`, `fused_shape`, `OmniVoiceWeights::has`.
- `to_heads_f16` takes a source row stride; the op `to_heads_at(col)` reads one block of the wide
  row. `to_heads` is `to_heads_at(0)` plus the old size check.
- `swiglu_strided_f16` reads the two halves of one wide row and writes a contiguous `[rows, d]`.
- Activations: `q_row`/`k_row`/`v_row` became `qkv`; `up` became `gate_up` and `gate` stays the
  SwiGLU output (the `blas.rs` wrapper has no `ld` parameter, so the down GEMM needs a contiguous
  input). The arena grew by 13 MB.

Measured (isolated GEMMs, 1485 MHz, 3 runs): 486 rows: qkv 341.8 us against 393.9 (q + 2·kv);
gate_up 477.9 against 468.4. 214 rows: qkv 191.4 against 212.8; gate_up 242.7 against 324.4.
The strided SwiGLU is 11% slower than the contiguous one (59.2 against 53.4 us).

### P4: position metadata and dead scratch (files: `kernels/mod.rs`, `ops/rope.rs`, `weights.rs`, `forward.rs`, `ops/attention.rs`)

- New kernel `rope_half_f16_rows` and op `rope_rows(x, inv_freq, t)`: the position of a flat head
  row is `row % t`. The two position tensors and their per-pass uploads are gone.
- `kx` and `vx` are gone from `Activations` and `BidirScratch` (dead since D1). The arena fell by
  16.8 MB.

Measured: `rope_half_f16_rows` 111.2 us against 117.6 us with the table on the same 8 MB tensor.
The packed 972-row pass fell from 105.5 to 100.0 ms warm.

### P5: d128 flash attention (files: `kernels/mod.rs`, `ops/attention.rs`, `tests/attention_flash128.rs`)

- New kernel `flash_attn_tc128_f16`: the d64 tensor-core kernel of lesson 5 at a head width of
  128 (16 k steps of 8, four B-fragment groups at columns 0/32/64/96, 16 O accumulators), 64
  query rows per block (4 warps of 16), 32 keys per tile, GQA by `head / (hq / hkv)`, grid
  `(ceil(t/64), hq, b)`.
- Op `attention_flash128`. `attention_bidir` dispatches to it when `valid == None` and the head
  width is 128; the cuBLAS implementation is now `attention_bidir_grouped` and the padded callers
  keep using it.
- Measured at the production shapes: 486 positions 468 us → 333 us (−29%, 4.1 → 5.8 TF/s); 214
  positions 164 us → 104 us (−37%). Per step 17.7 ms → 12.2 ms of attention.
- The 64-key tile was tried and is not better (331/103 us): the shared memory doubles and the
  block count per SM falls to one. 32 keys stays.
- PTX: no `ld.local`, no `st.local`; about 150 registers per thread, 3 blocks per SM, 12 of 64
  warps.
- Bug found by the new f64 test: the op passed `ATTN_SCALE` (`1/√64`); a 128-wide head needs
  `1/√128`. The symptom was quiet (mean 0.011, max 0.083).

## 3. Criterion 3

Protocol (P2): product mode, two warmup renders, median of the timed runs, `nvidia-smi` clock
recorded per run.

| Build | Runs (s) | Median | Clock / power / temp |
|---|---|---|---|
| before P3 | 2.702, 2.622 | 2.66 | 1875-1927 MHz, 73-75 W, 62-65 C |
| after P3 | 2.586, 2.602 | 2.59 | 1875-1942 MHz |
| after P4 | 2.456, 2.468 | 2.46 | 1867-1905 MHz |
| after P5 | 2.269, 2.266, 2.390, 2.260, 2.336, 2.326, 2.234 | 2.30 | 1867-1942 MHz, 73.6-75.0 W, 64.8-69.8 C |

**Criterion 3 (2.3 s) is met at the median and missed on the hotter runs.** The before numbers
were taken at 62-65 C and the after numbers at 65-70 C, so the comparison is biased against the
new code; the last run of the final build, at 64.8 C, is 2.234 s. The trustworthy part is the
isolated measurement: attention alone accounts for about 0.18 s of the 0.37 s gap, and the GEMM
fusion another 0.06-0.13 s.

The user must decide whether 2.30 s (median) counts as met. Do not restate the criterion without
approval (ruling of the review handoff).

## 4. Rulings applied

- Qwen3 keeps the half-split RoPE pair `(i, i + d/2)`. The derived position uses the same kernel
  arithmetic, so no layout changed.
- The two-pass decode, the audio tokenizer on the CPU and the f32 residual stream are unchanged.
- `kv_batch = b·h_kv` for the grouped GEMMs and `score_batch = b·hq` for the padding mask stay as
  they are (the D1 bug).
- No conventional KV cache: the flash kernel is bidirectional and has no mask, which is only
  valid because the pass sees the whole sequence.
- `class_temperature > 0` (the branch the app never uses) still samples on the host.

## 5. Open work, in the order I would take it

1. **Commit the three packages** (needs approval). Suggested split: P3, then P4, then P5, each
   with its tests and docs.
2. **Criterion 3 decision** by the user (section 3).
3. **Flash headroom**, if more speed is wanted. The kernel runs at 38% of the tensor peak with 12
   of 64 warps. The measured options from the lesson: 8 warps of 16 rows (128 rows per block,
   fewer barriers, fewer blocks), 32 rows per warp (needs about 210 registers), 16 keys per tile,
   or streaming V from global instead of shared. Each needs its own before/after at `t = 486` and
   `t = 214`; the micro-bench at 214 is noisy (±10%), so measure `t = 486` and the whole step.
4. **The `scores` tensor is now dead in production** (67 MB): the flash path covers both decode
   passes and the grouped path is only used by the padded tap test. Removing it from the plan
   would need the padded test to allocate it outside the model arena; that is a small design
   question, and the arena is not the constraint at 1460 MB.
5. The 486-row GEMM of gate/up (N=6144) is 2% slower than two N=3072 GEMMs. Splitting the two
   shapes (fused for M=214, separate for M=486) would need both weight layouts: not worth it at
   the measured size.

## 6. How to reproduce

```powershell
scripts\cargo.cmd test
scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test forward -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test decode_gpu -- --ignored --nocapture --test-threads 1
scripts\cargo.cmd oxide test -- -p jeuvoice-omnivoice --features cuda-kernels --release --test weights -- --ignored --nocapture --test-threads 1 <one test name>
scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release
target\release\gpu-bench.exe attn_p5
scripts\cargo.cmd oxide build -- --features cuda-kernels -p jeuvoice-omnivoice --release --example render_tokens
target\release\examples\render_tokens.exe s1 out\s1.bin product 2
```

The weights tests must run one at a time: `the_adapter_changes_the_base_weights` holds 2.7 GB.

## 7. Traps found in this round

- `ATTN_SCALE` is 1/√64. Any 128-wide head needs its own scale. The tests are what caught it.
- The micro-bench of `attn_p5` at `t = 214` swings between 95 and 126 us depending on the clock
  state; the grouped row swings with it. Compare ratios at `t = 486` and the whole step, not the
  small shape.
- A tap maximum can move between processes of the same binary (`hidden_27` 13.82 against 16.61
  with the same mean to 3e-5): cuBLAS picks its kernel per process and the trajectory amplifies
  one element. Do not record a single max as a regression or an improvement.
- `rope_rows` requires a row count that is a whole multiple of `t`; a partial run panics on
  purpose.
