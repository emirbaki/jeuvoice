# Rust CUDA Phase 0 (GPU base) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `jeuvoice-gpu` crate with Rust kernels (cuda-oxide), cuBLAS GEMM, a fixed activation buffer, test and benchmark helpers, the first real kernel (RMSNorm) and lesson 0. The app does not use it yet.

**Architecture:** The whole workspace builds with the cuda-oxide codegen backend through `scripts\cargo.cmd oxide ...` (build way 1 of the spec, checked on 2026-11-24: `jeuvoice-app` builds in release with the backend and starts). `jeuvoice-gpu` is a workspace member but not a default member, so a plain `cargo build` without the cuda-oxide tools still builds the app as today. cuBLAS loads at run time from `cublas64_13.dll`.

**Tech Stack:** Rust 1.99.1, NVIDIA cuda-rust (`cuda-core`, `cuda-device`, `cuda-host`) at commit `5e5c77b19a4b3204d6542bab2f0fde2a4eed824b`, cuBLAS 13 (`cublas64_13.dll`), `libloading`, `half`.

**Spec:** `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md` (Phase 0 parts). Phase 1 (Whisper) gets its own plan after this one is done. Three Phase 0 parts of the spec move to the Phase 1 plan, because only Whisper uses them: the ggml weight reader, the `tools/reference/` Python tool, and the reference-file loader. The packaging change also moves to Phase 1: in Phase 0 the app does not use `jeuvoice-gpu`.

## Global Constraints

- Rust **1.99.1**, pinned in `rust-toolchain.toml` for all platforms, with `rust-src`, `rustc-dev`, `llvm-tools`, `clippy`, `rustfmt`.
- cuda-rust git dependency pinned to rev `5e5c77b19a4b3204d6542bab2f0fde2a4eed824b`. If it blocks the work, patch a fork under the user's GitHub account and pin to the fork commit. Do not switch a kernel to CUDA C++.
- A build without the cuda-oxide tools must still work: `jeuvoice-gpu` is not in `default-members`.
- Kernels are Rust (`#[cuda_module]` / `#[kernel]`). Shared memory access uses `unsafe` with a `SAFETY` comment, or `#[verify]`.
- Number format: fp16 weights and activations, fp32 accumulate, norm and softmax math in fp32.
- Errors use plain words that the user can act on, like `WhisperError` in `crates/jeuvoice-whisper/src/lib.rs`.
- Code comments in English. Lessons in Turkish, in `docs/cuda-lessons/`.
- No change to the app UI.
- Work on the branch `rust-cuda-phase0`. The first commit adds the spec file.

## Review Focus

1. A machine without the NVIDIA driver or without a GPU: `Gpu::open` returns an error, with no panic and no abort (Task 2 test `open_without_gpu_is_an_error`).
2. A user machine with the NVIDIA driver but without the CUDA toolkit: the kernels must still load (Task 6, step "toolkit-free run").
3. Sizes that are not a multiple of the block or tile size (d = 1000, m = 67): correct results (Task 4 and Task 6 tests).
4. An activation buffer that is too small: an error that gives the needed and the free bytes, never a write past the end (Task 3 test `plan_overflow_is_an_error`).
5. A plain build with no cuda-oxide tools: `scripts\cargo.cmd test` passes as before (Task 1 verification).

---

### Task 1: Toolchain, build command and the `jeuvoice-gpu` crate

**Files:**
- Modify: `rust-toolchain.toml`
- Modify: `Cargo.toml` (workspace members, `default-members`, workspace dependencies)
- Modify: `scripts/cargo.cmd`
- Create: `crates/jeuvoice-gpu/Cargo.toml`, `crates/jeuvoice-gpu/src/lib.rs`, `crates/jeuvoice-gpu/src/kernels/mod.rs`
- Test: `crates/jeuvoice-gpu/tests/smoke.rs`
- Modify: `docs/development-log.md`

**Interfaces:**
- Produces: the build commands every later task uses:
  - `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu` (GPU tests)
  - `scripts\cargo.cmd test` (plain workspace tests, no cuda-oxide)
- Produces: `jeuvoice_gpu::kernels` module with `#[cuda_module] mod device` and the smoke kernel `add_f32(a: &[f32], b: &[f32], out: DisjointSlice<f32>)`.

- [ ] **Step 1: Commit the spec on a new branch**

```
git switch -c rust-cuda-phase0
git add docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md docs/superpowers/plans/2026-11-24-rust-cuda-phase0.md
git commit -m "docs: add the Rust CUDA inference design and the Phase 0 plan"
```

- [ ] **Step 2: Pin the toolchain**

`rust-toolchain.toml`: `channel = "1.99.1"`, `components = ["rust-src", "rustc-dev", "llvm-tools", "clippy", "rustfmt"]`.

- [ ] **Step 3: Add the cuda-oxide environment to `scripts/cargo.cmd`**

After the existing `vcvars64.bat` call, and only when the first argument is `oxide`:
- Remove every PATH entry that contains `mingw` (libclang reads the MinGW headers otherwise; see `D:\MLProjelerim\cuda-rust-spike\env.ps1`).
- Set `CUDA_PATH` and `CUDA_TOOLKIT_PATH` from the existing `CUDA_PATH`, else from the newest `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*`.
- Set `LIBCLANG_PATH` to `C:\Program Files\LLVM\bin` if it is not set.
- Add `%CUDA_PATH%\bin\x64`, `%CUDA_PATH%\bin`, `%LIBCLANG_PATH%`, `%VCPKG_ROOT%\installed\x64-windows\bin` to PATH. `VCPKG_ROOT` defaults to `D:\vcpkg`.
- If `cargo-oxide.exe`, `libclang.dll` or `%VCPKG_ROOT%\installed\x64-windows\lib\ffi.lib` is missing, print one line that names the missing tool and the install command from the spec, and `exit /b 1`.

The existing `BINDGEN_EXTRA_CLANG_ARGS` stays as is (cuda-bindings also uses bindgen).

- [ ] **Step 4: Create the crate and the workspace entries**

- Workspace `members` gets `crates/jeuvoice-gpu`. Add `default-members` = the five current crates (not `jeuvoice-gpu`).
- Workspace dependencies: `cuda-core`, `cuda-device`, `cuda-host` from `git = "https://github.com/NVIDIA/cuda-rust"`, `rev = "5e5c77b19a4b3204d6542bab2f0fde2a4eed824b"`. Add `half = "2"` and `libloading = "0.8"`.
- `crates/jeuvoice-gpu/Cargo.toml` uses those, plus the `[profile.*.package.jeuvoice-gpu] debug = 1` entries the other crates have.
- `src/kernels/mod.rs` holds `#[cuda_module] pub mod device { ... }` with `add_f32` (the `vecadd` example shape).

- [ ] **Step 5: Write the smoke test**

`crates/jeuvoice-gpu/tests/smoke.rs`:

```rust
#[test]
fn add_f32_runs_on_the_gpu() {
    // 1000 is not a multiple of 256: the last block is partly empty.
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
    let out = jeuvoice_gpu::smoke_add(&a, &b).expect("smoke kernel");
    assert_eq!(out.len(), n);
    for i in 0..n {
        assert_eq!(out[i], 3.0 * i as f32);
    }
}
```

`pub fn smoke_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String>` in `lib.rs` opens device 0, launches `add_f32`, and copies the result back. Task 2 replaces the `String` error.

- [ ] **Step 6: Run the GPU test**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`
Expected: `add_f32_runs_on_the_gpu ... ok`

- [ ] **Step 7: Run the plain workspace tests**

Run: `scripts\cargo.cmd test`
Expected: all current tests pass, and `jeuvoice-gpu` does not build (it is not a default member).

Run: `scripts\cargo.cmd build -p jeuvoice-app --release`
Expected: the release app builds with the 1.99.1 pin and plain cargo, as `packaging/windows/collect.ps1` needs.

- [ ] **Step 8: Write the development log entry**

Add a dated section to `docs/development-log.md`: build way 1 was picked, with the 2026-11-24 result (release app builds with the backend in 26 min and starts), the MinGW fix, the two commands above, and the tool install list.

- [ ] **Step 9: Commit**

```
git add rust-toolchain.toml Cargo.toml Cargo.lock scripts/cargo.cmd crates/jeuvoice-gpu docs/development-log.md
git commit -m "feat(gpu): add the jeuvoice-gpu crate built with cuda-oxide"
```

---

### Task 2: Device, context and errors

**Files:**
- Create: `crates/jeuvoice-gpu/src/device.rs`, `crates/jeuvoice-gpu/src/error.rs`
- Modify: `crates/jeuvoice-gpu/src/lib.rs` (`smoke_add` uses `Gpu`)
- Test: `crates/jeuvoice-gpu/tests/device.rs`, `crates/jeuvoice-gpu/tests/no_gpu.rs`

**Interfaces:**
- Produces:
  - `pub struct Gpu` with `pub fn open(ordinal: usize) -> Result<Gpu, GpuError>`, `pub fn info(&self) -> &DeviceInfo`, `pub fn stream(&self) -> &Arc<cuda_core::Stream>`, `pub fn ctx(&self) -> &Arc<cuda_core::CudaContext>`, `pub fn free_vram(&self) -> Result<u64, GpuError>`.
  - `pub struct DeviceInfo { pub name: String, pub cc: (u32, u32), pub sm_count: u32, pub total_vram: u64, pub mem_bandwidth_gbs: f64, pub fp16_tensor_tflops: f64 }`
  - `pub enum GpuError { NoDriver, NoDevice, TooOld { cc: (u32, u32) }, OutOfMemory { needed: u64, free: u64 }, Driver(String), Blas(String) }` with `Display` in plain words and `std::error::Error`.
- Consumes: Task 1 crate.

- [ ] **Step 1: Write the failing tests**

`tests/device.rs`:

```rust
#[test]
fn open_reads_the_device_info() {
    let gpu = jeuvoice_gpu::Gpu::open(0).expect("open GPU 0");
    let info = gpu.info();
    assert!(!info.name.is_empty());
    assert!(info.cc >= (7, 5), "cc {:?}", info.cc);
    assert!(info.sm_count > 0);
    // RTX 3050 Ti Laptop: about 192 GB/s. Any real card is in this range.
    assert!(info.mem_bandwidth_gbs > 50.0 && info.mem_bandwidth_gbs < 10_000.0);
    assert!(info.fp16_tensor_tflops > 1.0);
    assert!(gpu.free_vram().unwrap() <= info.total_vram);
}
```

`tests/no_gpu.rs` (its own process, one test only):

```rust
#[test]
fn open_without_gpu_is_an_error() {
    // Must be set before the first CUDA call in this process.
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "") };
    let err = jeuvoice_gpu::Gpu::open(0).err().expect("no device must be an error");
    assert!(matches!(err, jeuvoice_gpu::GpuError::NoDevice | jeuvoice_gpu::GpuError::NoDriver), "{err:?}");
    assert!(!err.to_string().is_empty());
}
```

- [ ] **Step 2: Run them to see them fail**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu --test device --test no_gpu`
Expected: compile error, `Gpu` is not defined.

- [ ] **Step 3: Implement `Gpu`, `DeviceInfo` and `GpuError`**

- `cuda_core` loads the driver at run time with `libloading`, so a missing `nvcuda.dll` maps to `GpuError::NoDriver`. Zero devices maps to `NoDevice`. `cc < (7, 5)` maps to `TooOld`.
- `mem_bandwidth_gbs = 2 × memory clock (Hz) × bus width (bits) / 8 / 1e9`, from `cuDeviceGetAttribute` (`MEMORY_CLOCK_RATE`, `GLOBAL_MEMORY_BUS_WIDTH`).
- `fp16_tensor_tflops = 2 × sm_count × clock (Hz) × fp16 FMA per clock per SM (fp32 accumulate) / 1e12`. The per-SM numbers come from a small table keyed by `cc` (7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 12.0). Cite the NVIDIA architecture whitepaper for each row in a comment.
- `Display` texts, for example `NoDriver`: "The NVIDIA driver was not found. Install the newest NVIDIA driver." `OutOfMemory`: "The graphics card needs {needed} MB of free memory, but only {free} MB is free. Close other programs that use the graphics card."

- [ ] **Step 4: Run the tests to see them pass**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`
Expected: all tests pass, including `add_f32_runs_on_the_gpu`.

- [ ] **Step 5: Commit**

`git commit -m "feat(gpu): open the device and report errors in plain words"`

---

### Task 3: Activation buffer and tensor views

**Files:**
- Create: `crates/jeuvoice-gpu/src/tensor.rs`, `crates/jeuvoice-gpu/src/arena.rs`
- Test: unit tests in `arena.rs` (CPU only), `crates/jeuvoice-gpu/tests/arena.rs` (GPU)

**Interfaces:**
- Produces:
  - `pub enum DType { F16, F32, I32 }` with `pub fn size(self) -> usize`.
  - `pub struct Tensor { pub offset: usize, pub dtype: DType, pub shape: Vec<usize>, pub strides: Vec<usize> }` (strides in elements, row-major by default), `pub fn numel(&self) -> usize`, `pub fn bytes(&self) -> usize`.
  - `pub struct ArenaPlan` (CPU only): `pub fn new() -> Self`, `pub fn add(&mut self, dtype: DType, shape: &[usize]) -> Tensor`, `pub fn total_bytes(&self) -> usize`. Every offset is a multiple of 256 bytes.
  - `pub struct Arena`: `pub fn new(gpu: &Gpu, plan: &ArenaPlan) -> Result<Arena, GpuError>`, `pub fn view<T: Element>(&self, t: &Tensor) -> DeviceBufferView<'_, T>`, `pub fn upload<T: Element>(&self, gpu: &Gpu, t: &Tensor, data: &[T]) -> Result<(), GpuError>`, `pub fn download<T: Element>(&self, gpu: &Gpu, t: &Tensor) -> Result<Vec<T>, GpuError>`.
  - `pub trait Element: DeviceCopy { const DTYPE: DType; }` for `half::f16`, `f32`, `i32`. `view`, `upload` and `download` panic if `T::DTYPE != t.dtype` (a programming error, not a user error).
- Consumes: `Gpu`, `GpuError` (Task 2).

- [ ] **Step 1: Write the failing CPU tests in `arena.rs`**

```rust
#[test]
fn offsets_are_256_byte_aligned() {
    let mut p = ArenaPlan::new();
    let a = p.add(DType::F16, &[3]);      // 6 bytes
    let b = p.add(DType::F32, &[5, 7]);   // 140 bytes
    assert_eq!(a.offset, 0);
    assert_eq!(b.offset, 256);
    assert_eq!(b.strides, vec![7, 1]);
    assert_eq!(p.total_bytes(), 512);
}
```

- [ ] **Step 2: Write the failing GPU tests in `tests/arena.rs`**

- `round_trip_f16_f32_i32`: upload and download one tensor of each dtype, values equal.
- `kernel_writes_only_its_view`: plan three `F32` tensors of 1000 elements, fill all with 7.0, run `add_f32` with the middle one as output, check the first and the last are still all 7.0 and the middle one holds `a + b`.
- `plan_overflow_is_an_error`: an `ArenaPlan` with one `F32` tensor of `free_vram / 4 + 1_000_000` elements gives `Err(GpuError::OutOfMemory { needed, free })` with `needed > free`. Check before allocation with `Gpu::free_vram`.

- [ ] **Step 3: Run them to see them fail**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu --lib --test arena`
Expected: compile error, `ArenaPlan` is not defined.

- [ ] **Step 4: Implement `tensor.rs` and `arena.rs`**

`Arena` holds one `DeviceBuffer<u8>` of `plan.total_bytes()`. Views come from `DeviceBuffer::slice` of the byte range. If a cuda-oxide kernel launch does not accept a `DeviceBufferView` as a `&[T]` or `DisjointSlice<T>` argument, find the supported way in the cuda-core and cuda-host sources (for example raw device pointer arguments) and write it in the lesson 0 notes of Task 6. The test `kernel_writes_only_its_view` decides.

- [ ] **Step 5: Run the tests to see them pass**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`
Expected: all pass.

- [ ] **Step 6: Commit**

`git commit -m "feat(gpu): add the activation buffer and tensor views"`

---

### Task 4: cuBLAS GEMM

**Files:**
- Create: `crates/jeuvoice-gpu/src/blas.rs`
- Test: `crates/jeuvoice-gpu/tests/blas.rs`

**Interfaces:**
- Produces:
  - `pub struct Blas`: `pub fn load(gpu: &Gpu) -> Result<Blas, GpuError>` (loads `cublas64_13.dll` from PATH and the exe folder), `pub fn load_from(gpu: &Gpu, path: &Path) -> Result<Blas, GpuError>`.
  - `pub enum Op { N, T }`.
  - `pub fn gemm_f16(&self, gpu: &Gpu, a: &DeviceBufferView<f16>, op_a: Op, b: &DeviceBufferView<f16>, op_b: Op, c: &DeviceBufferView<f16>, m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> Result<(), GpuError>`
    Meaning in **row-major** terms: `C[m×n] = alpha · op(A) · op(B) + beta · C`. `op(A)` is `m×k`, `op(B)` is `k×n`. `Op::T` means the stored matrix is the transpose (a PyTorch `Linear` weight `[out, in]` is `B` with `Op::T`).
  - `pub fn gemm_f16_batched(&self, gpu: &Gpu, a, op_a, stride_a: usize, b, op_b, stride_b: usize, c, stride_c: usize, m, n, k, batch: usize, alpha: f32, beta: f32) -> Result<(), GpuError>` (strides in elements).
  - Both use `cublasGemmEx` / `cublasGemmStridedBatchedEx` with `CUDA_R_16F` in and out, `CUBLAS_COMPUTE_32F`, default tensor-op algorithm, on `gpu.stream()` (`cublasSetStream_v2` with `Stream::cu_stream()`).
- Consumes: `Gpu`, `Arena`, `Tensor` (Tasks 2–3).

- [ ] **Step 1: Write the failing tests**

All compare with a CPU `f32` matrix multiply on the same `f16` inputs. Tolerance: `max_abs <= 2e-2 * max(1, |want|)` per element. Inputs are repeatable pseudo-random values in [-1, 1].

- `gemm_nn_odd_sizes`: m = 67, n = 129, k = 96, `Op::N, Op::N`.
- `gemm_linear_weight`: x `[m=67, k=96]`, W `[n=129, k=96]`, `Op::N, Op::T` gives `x · Wᵀ`.
- `gemm_beta_adds`: beta = 1.0 adds to a C that holds known values.
- `gemm_batched_heads`: batch = 16, m = n = 64, k = 128 (attention shape), each batch checked.
- `gemm_follows_kernel_on_stream`: `add_f32`-style kernel writes A, then `gemm_f16` reads it with no sync between. The result uses the new A. (Add an `f16` fill kernel `fill_f16(out: DisjointSlice<f16>, v: f32)` for this, in `kernels/mod.rs`.)
- `missing_dll_is_an_error`: `Blas::load_from(&gpu, Path::new("C:/nope/cublas64_13.dll"))` gives `Err(GpuError::Blas(_))`.

- [ ] **Step 2: Run them to see them fail**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu --test blas`
Expected: compile error, `Blas` is not defined.

- [ ] **Step 3: Implement `blas.rs`**

Hand-written `extern "C"` function pointer types, loaded with `libloading`. cuBLAS is column-major: compute the row-major `C = op(A)·op(B)` as the column-major `Cᵀ = op(B)ᵀ·op(A)ᵀ`, so pass B first, swap m and n, and keep each `Op`. Write that reason in a comment, because lesson 4 builds on it.

- [ ] **Step 4: Run the tests to see them pass**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`
Expected: all pass.

- [ ] **Step 5: Commit**

`git commit -m "feat(gpu): call cuBLAS fp16 GEMM on the kernel stream"`

---

### Task 5: Test and benchmark helpers

**Files:**
- Create: `crates/jeuvoice-gpu/src/check.rs`, `crates/jeuvoice-gpu/src/bench.rs`, `crates/jeuvoice-gpu/src/bin/gpu-bench.rs`
- Test: unit tests in `check.rs` and `bench.rs`

**Interfaces:**
- Produces:
  - `pub struct Diff { pub max_abs: f32, pub max_rel: f32, pub mean_abs: f32, pub worst_index: usize }`, `pub fn compare(got: &[f32], want: &[f32]) -> Diff` (panics on length mismatch). `max_rel` uses `|got - want| / max(|want|, 1e-6)`.
  - `pub struct Tol { pub abs: f32, pub rel: f32 }`, `pub fn assert_close(name: &str, got: &[f32], want: &[f32], tol: Tol)`. An element fails only if both its abs and its rel error are over the limits. The panic message gives the name, the `Diff`, and the worst element.
  - `pub struct Timing { pub us: f64 }`, `pub fn time_on_stream(gpu: &Gpu, warmup: usize, iters: usize, f: impl FnMut() -> Result<(), GpuError>) -> Result<Timing, GpuError>` with CUDA events around the `iters` calls.
  - `pub fn pct_of_memory_limit(bytes: u64, t: &Timing, info: &DeviceInfo) -> f64`, `pub fn pct_of_tensor_limit(flops: u64, t: &Timing, info: &DeviceInfo) -> f64`.
  - `gpu-bench` binary: prints the device info, then one table row per benchmark: name, size, µs, GB/s or TFLOP/s, % of limit. It takes an optional name filter as its first argument.
- Consumes: Tasks 2–4.

- [ ] **Step 1: Write the failing unit tests**

- `compare_finds_the_worst`: `compare(&[1.0, 2.0, 3.5], &[1.0, 2.0, 3.0])` gives `max_abs == 0.5`, `worst_index == 2`.
- `assert_close_passes_small_rel_error`: got `[1000.5]`, want `[1000.0]`, `Tol { abs: 0.1, rel: 1e-3 }` passes.
- `assert_close_names_the_worst` (`#[should_panic(expected = "rmsnorm")]`).
- `pct_math`: 192e9 bytes in 1 s on a card with 192 GB/s gives 100.0.

- [ ] **Step 2: Run them to see them fail, implement, run them to see them pass**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu --lib`
Expected first: compile errors. Expected after: all pass.

- [ ] **Step 3: Add the GEMM benchmark to `gpu-bench`**

Rows: `gemm_f16 1500x1280x1280` (Whisper encoder projection), `gemm_f16 1500x5120x1280` (Whisper MLP), `gemm_f16 4096x1024x1024` (Qwen3 projection), each with % of the tensor limit.

Run: `scripts\cargo.cmd oxide run -- -p jeuvoice-gpu --bin gpu-bench --release`
Expected: the table prints. Write the numbers in the Task 5 commit message.

- [ ] **Step 4: Commit**

`git commit -m "feat(gpu): add compare, timing and the gpu-bench table"`

---

### Task 6: RMSNorm kernels and lesson 0

**Files:**
- Create: `crates/jeuvoice-gpu/src/ops/mod.rs`, `crates/jeuvoice-gpu/src/ops/rmsnorm.rs`
- Modify: `crates/jeuvoice-gpu/src/kernels/mod.rs`, `crates/jeuvoice-gpu/src/bin/gpu-bench.rs`
- Test: `crates/jeuvoice-gpu/tests/rmsnorm.rs`
- Create: `docs/cuda-lessons/README.md`, `docs/cuda-lessons/00-gpu-temelleri.md`

**Interfaces:**
- Produces:
  - Kernels: `rmsnorm_naive_f32` (one thread per row) and `rmsnorm_f32` (one block of 256 threads per row, warp shuffle + shared memory, as in `D:\MLProjelerim\cuda-rust-spike\rmsnorm\src\main.rs`), and `rmsnorm_f16` (f16 in and out, f32 math, same block plan).
  - `pub fn rmsnorm(gpu: &Gpu, arena: &Arena, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError>`: picks `rmsnorm_f32` or `rmsnorm_f16` from `x.dtype`. Shape `[rows, d]`. `w` is `[d]` in the same dtype.
  - `pub fn rmsnorm_naive(...)` with the same signature, `F32` only (for the lesson and the benchmark).
- Consumes: Tasks 2–5.

- [ ] **Step 1: Write the failing tests**

CPU reference in the test file: `y = x / sqrt(mean(x²) + eps) · w` in `f64`. `eps = 1e-6`.

- `rmsnorm_f32_shapes`: for `(rows, d)` in `[(1, 1024), (7, 1000), (4096, 1024), (3, 64)]`, both `rmsnorm` and `rmsnorm_naive` pass `assert_close` with `Tol { abs: 1e-5, rel: 1e-4 }`.
- `rmsnorm_f16_matches`: `(4096, 1024)`, f16 in and out, `Tol { abs: 2e-3, rel: 1e-2 }` against the reference on the f16-rounded input.
- `rmsnorm_zero_row`: a row of zeros gives zeros (no NaN), since `eps > 0`.

- [ ] **Step 2: Run them to see them fail**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu --test rmsnorm`
Expected: compile error, `rmsnorm` is not defined.

- [ ] **Step 3: Implement the three kernels and the two host functions**

`d` and `eps` go in as kernel arguments. A `d` that is not a multiple of 256 must work (the strided loop handles it).

- [ ] **Step 4: Run the tests to see them pass**

Run: `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`
Expected: all pass.

- [ ] **Step 5: Add benchmark rows and measure**

Rows `rmsnorm_naive_f32 4096x1024`, `rmsnorm_f32 4096x1024`, `rmsnorm_f16 4096x1024`, with % of the memory limit.

Run: `scripts\cargo.cmd oxide run -- -p jeuvoice-gpu --bin gpu-bench --release rmsnorm`
Expected: `rmsnorm_f32` reaches 80% or more of the memory limit (the toolchain test reached 92%). The naive kernel is much slower. Keep the output for the lesson.

- [ ] **Step 6: Toolkit-free run**

Run `gpu-bench` from a `cmd` window with `PATH` reduced to `C:\Windows\System32;C:\Windows` plus the folder that holds `cublas64_13.dll` and `cublasLt64_13.dll` (copy them next to the exe), so no CUDA toolkit folder is on PATH.
Expected: the table prints. If kernel loading fails because libNVVM or nvJitLink is missing, build with `--materialize-cubin --arch sm_86` and add a matching `--arch sm_75` / `sm_89` note to the development log. Write the result in `docs/development-log.md` either way. Phase 1 packaging depends on it.

- [ ] **Step 7: Write lesson 0 in Turkish**

`docs/cuda-lessons/README.md`: the lesson list from the spec (0–9) with one line each, and how to run the tests and `gpu-bench`.

`docs/cuda-lessons/00-gpu-temelleri.md` has the seven parts from the spec:
1. What RMSNorm does in Qwen3 (and why Whisper uses LayerNorm instead), with the formula.
2. Why it exists, and how often it runs in one forward pass.
3. GPU basics with this kernel: thread, warp (32), block, grid, global / shared memory / registers, what PTX is, how `cargo oxide` turns Rust into PTX.
4. The naive kernel and its measured speed. Then the block-per-row kernel, one change at a time: coalesced reads, the warp shuffle (show the five `shfl.sync.bfly` lines from `cargo oxide inspect`), shared memory and `bar.sync`, then f16 data with f32 math. Each with the measured number from Step 5.
5. Why RMSNorm is memory bound: bytes moved vs the card's limit, and the % reached.
6. What to look at in `cargo oxide inspect` and in Nsight Compute (`ncu --set full`) for this kernel.
7. Three exercises, for example: make `rmsnorm_f32` work with 128 threads per block and compare, use `f16x2` loads, add the `#[verify]` proof in place of `unsafe`.

Also note the facts the earlier tasks found: why `unsafe` is needed on shared memory, the MinGW fix, the views answer from Task 3.

- [ ] **Step 8: Commit**

```
git add crates/jeuvoice-gpu docs/cuda-lessons docs/development-log.md
git commit -m "feat(gpu): add the RMSNorm kernels and lesson 0"
```
