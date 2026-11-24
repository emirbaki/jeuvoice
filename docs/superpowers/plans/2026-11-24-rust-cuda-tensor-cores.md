# Tensor-core kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tensor-core kernels for lessons 4 and 5: a GEMM and a flash attention on `mma.sync.m16n8k8`, measured against cuBLAS and the naive attention.

**Design (approved by the user on 2026-11-24):**

- `cuda_device::wmma::mma_m16n8k8_f32_f16` (`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`, "PTX 6.5 and sm_75+") and `cuda_device::wmma::ldmatrix_x1/x2/x4` (+ `_trans`) give the tensor cores on `sm_75`. The `ptx_asm!` test-profile bug does not block this work, so there is no wrapper macro and no fork.
- `m16n8k16` (sm_80) and `cp.async` (sm_80) are not allowed (PTX target `sm_75`).
- The product keeps cuBLAS for GEMM. The encoder switches its attention to the tensor-core flash kernel only if it measures faster than `AttnKind::Naive` (2.48 ms at `20 x 1500 x 64`) and passes the encoder reference test.

**Spec:** `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md` (lessons 4 and 5). The handoff, section 4, step 2, describes the task.

## Global Constraints

All rules of `AGENTS.md` apply: Rust kernels only, device code only in `jeuvoice-gpu`, raw device pointers, a `SAFETY` comment on each `unsafe` block, PTX `sm_75`, fp16 data with f32 accumulate, tests first, a Turkish lesson section for each kernel, measure with `gpu-bench`, work on the branch `rust-cuda-tensor-cores`.

## The m16n8k8 fragment layout (PTX ISA, "Matrix Fragments for mma.m16n8k8")

For lane `l`: `g = l / 4` (the group, 0..7) and `t = l % 4` (the thread in the group, 0..3). A pair of f16 values packs into one `u32`, with the lower column in the low 16 bits.

| Fragment | Registers | Elements |
|---|---|---|
| A (16 x 8, row-major) | `a[0]`, `a[1]` | `a[0]` = A[g][2t, 2t+1], `a[1]` = A[g+8][2t, 2t+1] |
| B (8 x 8, column-major) | `b` | B[2t, 2t+1][g] (two k rows of column g) |
| C, D (16 x 8, f32) | `c[0..4]` | `c[0], c[1]` = C[g][2t, 2t+1], `c[2], c[3]` = C[g+8][2t, 2t+1] |

---

### Task 1: The fragment probe

**Files:** `crates/jeuvoice-gpu/src/kernels/mod.rs` (kernel `mma_probe_f16`), `crates/jeuvoice-gpu/src/ops/gemm_lesson.rs` (host `mma_probe`), `crates/jeuvoice-gpu/tests/tensor_cores.rs`.

- [ ] **Step 1: Failing test** `mma_probe_matches_cpu`: one warp computes `D [16, 8] = A [16, 8] · B [8, 8]` (A and B f16 row-major in global memory, D f32). Random A and B, compared with a CPU f64 product, `Tol { abs: 1e-3, rel: 1e-3 }`. A second case with A = identity rows checks that D[r] = B row by row (a wrong fragment layout shows as a permutation).
- [ ] **Step 2: Implement** `mma_probe_f16`: each lane builds its fragments from global memory by the table above, calls `mma_m16n8k8_f32_f16`, and writes its four outputs by the table.
- [ ] **Step 3: Commit** `feat(gpu): add the mma fragment probe`.

### Task 2: `gemm_tc_f16` and the lesson 4 section

**Files:** `kernels/mod.rs`, `ops/gemm_lesson.rs` (`GemmKind::TensorCore`), `tests/gemm_lesson.rs`, `src/bin/gpu-bench.rs`, `docs/cuda-lessons/04-gemm.md`.

- [ ] **Step 1: Failing test:** `GemmKind::TensorCore` against cuBLAS at `m = 67, n = 129, k = 96` and `1500 x 1280 x 1280`, with the tolerance of the other lesson kinds.
- [ ] **Step 2: Implement:** block tile 64 x 64, k step 32, 4 warps (each 32 x 32 = 2 x 4 mma tiles), A and B tiles in shared memory with padding against bank conflicts, `ldmatrix_x4` for A and `ldmatrix_x2_trans` for B, edge tiles filled with zeros. Then a register prefetch of the next k tile.
- [ ] **Step 3: Bench rows** `gemm_tc_f16` (both versions) at `1500 x 1280 x 1280`, next to cuBLAS and `gemm_regtile_f16`.
- [ ] **Step 4: Lesson 4 section** in Turkish: the fragment table, `ldmatrix`, the measured steps, the gap to cuBLAS and why.
- [ ] **Step 5: Commit** `feat(gpu): add the tensor-core GEMM lesson kernel`.

### Task 3: Tensor-core flash attention and the lesson 5 section

**Files:** `kernels/mod.rs`, `ops/attention.rs` (`AttnKind::FlashTc`), `tests/attention.rs`, `gpu-bench.rs`, `docs/cuda-lessons/05-attention.md`, `crates/jeuvoice-whisper/src/cuda/encoder.rs` (only if it measures faster).

- [ ] **Step 1: Failing tests:** `AttnKind::FlashTc` against the CPU at `heads = 2, seq = 37`, against `Naive` at `20 x 1500 x 64`, and the large-score case.
- [ ] **Step 2: Implement:** 64 query rows per block (4 warps of 16 rows), K and V tiles of 64 keys in shared memory, `Q·Kᵀ` with mma, the online softmax on the f32 accumulator registers (a row's max and sum need a shuffle over the 4 lanes of a group), `P` packed to f16 fragments, `P·V` with mma. Keys past `seq` are masked.
- [ ] **Step 3: Bench rows** next to `Naive` and `Flash`.
- [ ] **Step 4:** If `FlashTc` is faster than `Naive`, switch the encoder default to it and run the encoder reference test (`--ignored`). Record the ruling either way.
- [ ] **Step 5: Lesson 5 section** in Turkish. **Step 6: Commit** `feat(gpu): add the tensor-core flash attention`.

## Review Focus

1. Matrix sizes that are not a multiple of 16, 8 or 64 (`m = 67`, `seq = 37`): the edge tiles must read zeros, never past the end.
2. A warp that diverges before an `mma` or `ldmatrix` call: every lane must reach it (the `Safety` text of the wrappers).
3. Shared-memory reuse between k tiles: a barrier before the next tile overwrites the current one.
4. Scores that are large after the scale: the online softmax must not give `inf` or NaN.
5. The fragment layout on another card generation: the probe test runs on every card the tests run on.
