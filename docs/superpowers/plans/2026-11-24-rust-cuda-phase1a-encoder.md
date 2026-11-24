# Rust CUDA Phase 1a (Whisper encoder) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Whisper large-v3-turbo encoder runs on the GPU in Rust kernels: audio samples in, the `[1500, 1280]` encoder output out, checked layer by layer against PyTorch. Lessons 1–7 are written.

**Architecture:** All kernels stay in `jeuvoice-gpu` (the only crate with device code, see the Phase 0 PTX race). `jeuvoice-whisper` reads the ggml model file on the CPU (always built) and, behind the Cargo feature `cuda-kernels`, holds the host code that runs the encoder with `jeuvoice-gpu` ops and cuBLAS. A Python tool (developer only) writes reference tensors from Hugging Face `transformers`.

**Tech Stack:** Rust 1.99.1, cuda-oxide (cuda-rust `5e5c77b19a`), cuBLAS 13, `safetensors` crate, Python 3.12 in `engine\.venv` (torch 2.14 + transformers 5.19) for references only.

**Spec:** `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md`, Phase 1. Phase 1b (decoder, decoding rules, DTW, `CudaWhisper` in the app, packaging, speed criteria) gets its own plan after this one.

## Global Constraints

- All Phase 0 constraints stay: Rust 1.99.1, cuda-rust pin, kernels in Rust only (fork cuda-rust if blocked), fp16 data with fp32 accumulate, norm and softmax math in fp32, plain-word errors, English code comments, Turkish lessons, no UI change.
- Device code only in `jeuvoice-gpu`. Arena kernels take raw device pointers and a length.
- PTX target `sm_75` (set by `scripts\cargo.cmd oxide`). A kernel that needs `sm_80` features is not allowed in this plan.
- `jeuvoice-whisper` without the `cuda-kernels` feature builds and tests with plain `scripts\cargo.cmd test`, as today.
- Commands: GPU crate tests `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`. Whisper GPU tests `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels`. Benchmarks: `scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release`, then `target\release\gpu-bench.exe [filter]`.
- Model: large-v3-turbo. `n_mels = 128`, `n_audio_ctx = 1500`, `n_audio_state = 1280`, `n_audio_head = 20` (head size 64), `n_audio_layer = 32`, MLP width 5120, sample rate 16 kHz, `n_fft = 400`, hop 160, 30 s windows (480,000 samples, 3000 mel frames).
- Test audio (as in the README speed test): `D:\MLProjelerim\tts_lab\data\ref\my_voice2.mp4`, `my_voice.ogg`, `sesim\ses_001_0007.wav`. Tests that need them or the model file are `#[ignore]` and say why.
- Each kernel task ends with its lesson in `docs/cuda-lessons/NN-*.md` (the 7 parts of lesson 0: what, why, naive first, each speed step with numbers, the card limit, PTX/Nsight, exercises) and a `gpu-bench` row.

## Review Focus

1. Audio shorter than 30 s and audio of exactly 0 samples: the mel pads with zeros as whisper.cpp does, and no kernel reads past the input (Task 9 test `mel_short_and_empty_audio`).
2. A model file that is not large-v3-turbo, is quantized, or is cut short: a plain error, no panic (Task 2 tests `rejects_quantized_tensor`, `rejects_truncated_file`).
3. fp16 overflow in the encoder: activations above 65504 turn into `inf`. LayerNorm and the residual adds keep f32 math, and the encoder test checks for no `inf`/NaN in any layer output (Task 10 test `encoder_matches_reference`).
4. Calling the GPU code from a different thread than the one that opened the GPU (the app runs Whisper on a worker thread): Task 1 test `upload_from_another_thread`.
5. Softmax and attention rows of length 1500 (not a multiple of 32 or 64): Task 5 and Task 7 tests use 1500.

---

### Task 1: Phase 0 follow-ups the encoder needs

**Files:**
- Modify: `crates/jeuvoice-gpu/src/arena.rs`, `crates/jeuvoice-gpu/src/blas.rs`, `crates/jeuvoice-gpu/src/ops/mod.rs`, `crates/jeuvoice-gpu/src/kernels/mod.rs`
- Test: `crates/jeuvoice-gpu/tests/followups.rs`

**Interfaces:**
- Produces:
  - `Arena` holds an `Arc<Stream>` and syncs it in `Drop` before the buffer is freed (finding M1).
  - `Blas` holds `Arc<CudaContext>` and `Arc<Stream>`. `gemm_f16` / `gemm_f16_batched` keep the `gpu: &Gpu` argument but assert it is the same context (finding M2). `Drop` binds the context before `cublasDestroy_v2`.
  - `Arena::upload` / `download` call `gpu.ctx().bind_to_thread()` first (finding M3).
  - `Blas::load` tries `<folder of current_exe()>\cublas64_13.dll` first, then the normal search (finding M9).
  - Every op returns `Ok(())` at once for zero elements and uses `u32::try_from` for grid sizes (finding M4).
  - `block_sum` ends with `thread::sync_threads()` so a second call in one kernel is safe, and its doc says it needs exactly `ROW_BLOCK` threads (finding M6).

- [ ] **Step 1: Write the failing tests**
  - `upload_from_another_thread`: open a `Gpu` on the test thread, move it into `std::thread::spawn` (wrap in `Arc` if `Gpu` is not `Send`, and make it `Send + Sync` if cuda-core allows), upload and download a tensor there. Values equal.
  - `blas_outlives_gpu_value`: `let blas = Blas::load(&gpu)?; let gpu2 = Gpu::open(0)?; drop(gpu);` then a GEMM with `&gpu2` works.
  - `empty_ops_are_ok`: `ops::add_f32` and `ops::fill_f16` on 0-element tensors return `Ok(())`.
  - `block_sum_twice`: a test kernel `block_sum_twice_f32` that calls `block_sum` two times per block on different values gives both sums right (add the kernel to `kernels/mod.rs`, marked test-only in its doc).
- [ ] **Step 2: Run them to see them fail** — `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu --test followups`. Expected: compile errors or the listed failures.
- [ ] **Step 3: Implement the six changes above.**
- [ ] **Step 4: Run the whole GPU suite** — `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu`. Expected: all pass (23 old + 4 new).
- [ ] **Step 5: Commit** — `fix(gpu): close the Phase 0 review minors the encoder needs`. Record M5, M7, M8, M10 as still deferred in the development log.

---

### Task 2: ggml model reader

**Files:**
- Create: `crates/jeuvoice-whisper/src/ggml.rs`
- Modify: `crates/jeuvoice-whisper/src/lib.rs` (`pub mod ggml;`)
- Test: `crates/jeuvoice-whisper/tests/ggml.rs`

**Interfaces:**
- Produces (CPU only, no feature gate):
  - `pub struct Hparams { pub n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype: i32 }`
  - `pub enum GgmlType { F32, F16 }`
  - `pub struct GgmlTensor { pub name: String, pub dtype: GgmlType, pub shape: Vec<usize> /* PyTorch order: slowest first */, pub data_offset: u64, pub data_len: u64 }`
  - `pub struct GgmlModel { pub hparams: Hparams, pub mel_filters: (usize, usize, Vec<f32>) /* n_mel, n_fft, data */, pub vocab: Vec<Vec<u8>>, pub tensors: Vec<GgmlTensor>, path: PathBuf }`
  - `pub fn GgmlModel::open(path: &Path) -> Result<GgmlModel, WhisperError>` reads everything except tensor data. `pub fn GgmlModel::tensor(&self, name: &str) -> Option<&GgmlTensor>`. `pub fn GgmlModel::read_f16(&self, t: &GgmlTensor) -> Result<Vec<half::f16>, WhisperError>` and `read_f32` (an F16 tensor read as f32 converts, an F32 tensor read as f16 rounds).
- File layout (whisper.cpp `whisper_model_load`, `whisper.cpp/src/whisper.cpp:1485` in `whisper-rs-sys-0.15.0`): `u32` magic `0x67676d6c`, 11 `i32` hparams in the order above, mel filters (`i32 n_mel`, `i32 n_fft`, `n_mel*n_fft` `f32`), vocab (`i32 n`, then for each token `u32 len` + bytes), then tensor records until EOF: `i32 n_dims`, `i32 name_len`, `i32 ttype`, `n_dims` × `i32 ne` (ggml order: `ne[0]` is the fastest), name bytes, data (`ttype` 0 = f32, 1 = f16, no padding). Reverse `ne` to get the PyTorch shape.
- Errors: a bad magic, a `ttype` other than 0/1, or a short read is `WhisperError::ModelLoad` with the reason.

- [ ] **Step 1: Write the failing tests**
  - `reads_a_tiny_file`: the test writes a small valid file (2 tokens, 1 f32 tensor `[2,3]`, 1 f16 tensor `[4]`) to a temp dir. `open` gives the hparams, the vocab, the two shapes in PyTorch order, and `read_f32` / `read_f16` give the values.
  - `rejects_quantized_tensor`: same file with `ttype = 8` → `Err(ModelLoad)` whose text names the tensor.
  - `rejects_truncated_file`: cut the file 3 bytes before the end → `Err(ModelLoad)`, no panic.
  - `#[ignore = "needs the downloaded model"] reads_large_v3_turbo`: `open(model_path())` gives `n_mels 128`, `n_audio_layer 32`, `n_text_layer 4`, `n_vocab 51866`, mel filters `128 × 201`; tensor `encoder.conv1.weight` has shape `[1280, 128, 3]`; `encoder.blocks.31.mlp.0.weight` has `[5120, 1280]`. Print every tensor name of block 0 for the lesson.
- [ ] **Step 2: Run them to see them fail** — `scripts\cargo.cmd test -p jeuvoice-whisper --test ggml`. Expected: `ggml` module missing.
- [ ] **Step 3: Implement `ggml.rs`.** Add `half` to `jeuvoice-whisper` dependencies. Read with `BufReader` and `seek` past tensor data.
- [ ] **Step 4: Run them to see them pass**, including `--ignored reads_large_v3_turbo`. Write the real tensor names and shapes into the development log (the encoder task uses them).
- [ ] **Step 5: Commit** — `feat(whisper): read the ggml model file in Rust`.

---

### Task 3: Reference tensors

**Files:**
- Create: `tools/reference/whisper_reference.py`, `tools/reference/README.md`
- Create: `crates/jeuvoice-gpu/src/reference.rs` (`pub mod reference;`)
- Modify: `crates/jeuvoice-gpu/Cargo.toml` (`safetensors = "0.6"` via workspace dependencies)
- Test: `crates/jeuvoice-gpu/tests/reference.rs`

**Interfaces:**
- Produces:
  - A Rust example `crates/jeuvoice-whisper/examples/dump_samples.rs <audio> <out.f32>`. It runs `jeuvoice_core::audio::{decode, to_whisper}` and writes the 16 kHz samples as raw little-endian f32. Both sides then use the same samples.
  - `whisper_reference.py <in.f32> <out.safetensors>` loads `openai/whisper-large-v3-turbo` with `transformers` in fp32, on CUDA if available. It reads the samples from `<in.f32>`. It saves these tensors, all as f32:

    | Key | Shape | Content |
    |---|---|---|
    | `samples` | `[n]` | the input samples |
    | `mel` | `[128, 3000]` | `WhisperFeatureExtractor` output of the first 30 s window |
    | `conv1`, `conv2` | `[1280, 3000]`, `[1280, 1500]` | after GELU |
    | `embed` | `[1500, 1280]` | after the positional embedding |
    | `block_00`, `block_15`, `block_31` | `[1500, 1280]` | hidden states after those blocks |
    | `ln_post` | `[1500, 1280]` | the encoder output |
  - `jeuvoice_gpu::reference::load(path: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, String>`.
  - Reference files go to `target/reference/whisper/<clip-stem>.safetensors` (not committed).
- [ ] **Step 1: Write the failing test** `reads_a_written_file`: write a 2-tensor safetensors file with the `safetensors` crate, `reference::load` returns both with shapes and values.
- [ ] **Step 2: Run it to see it fail, implement `reference.rs`, run it to see it pass.**
- [ ] **Step 3: Write the tool and make the three reference files.** Run from the repo root:
  ```
  cargo run -p jeuvoice-whisper --example dump_samples -- D:\MLProjelerim\tts_lab\data\ref\my_voice.ogg target\reference\whisper\my_voice.f32
  engine\.venv\Scripts\python.exe -I tools\reference\whisper_reference.py target\reference\whisper\my_voice.f32 target\reference\whisper\my_voice.safetensors
  ```
  and the same for `my_voice2.mp4` and `sesim\ses_001_0007.wav`. Expected: three files, each with the 9 keys above. `tools/reference/README.md` lists these commands and says the tool is for developers only and never ships.
- [ ] **Step 4: Commit** — `feat: add the Whisper reference tool and the reference loader`.

---

### Task 4: Element-wise kernels (lesson 1)

**Files:** `crates/jeuvoice-gpu/src/kernels/mod.rs`, `crates/jeuvoice-gpu/src/ops/elementwise.rs`, `crates/jeuvoice-gpu/tests/elementwise.rs`, `crates/jeuvoice-gpu/src/bin/gpu-bench.rs`, `docs/cuda-lessons/01-eleman-eleman.md`

**Interfaces:**
- Produces (all `F16`, f32 math inside):
  - `ops::bias_gelu(gpu, arena, x: &Tensor /*[rows, d]*/, bias: &Tensor /*[d]*/) -> Result<(), GpuError>`: in place, `x = gelu(x + bias)`, exact GELU `0.5·v·(1 + erf(v/√2))`.
  - `ops::bias_add(gpu, arena, x, bias)`: in place `x = x + bias`.
  - `ops::add_into(gpu, arena, acc: &Tensor, x: &Tensor)`: in place `acc = acc + x` (the residual add).
  - `ops::add_rows(gpu, arena, x: &Tensor /*[rows, d]*/, rows_src: &Tensor /*[rows, d]*/)`: `x += rows_src` (the positional embedding).
- [ ] **Step 1: Failing tests** vs a CPU f64 reference on f16-rounded inputs, shapes `[1500, 1280]` and `[7, 1000]`, `Tol { abs: 2e-3, rel: 1e-2 }`. GELU at the values `-6, -1, 0, 1, 6` matches `erf` GELU.
- [ ] **Step 2: Run to fail, implement: first one element per thread, then `f16x2` loads (two values per thread). Run to pass.**
- [ ] **Step 3: Bench rows** `bias_gelu 1500x5120` for both versions, with % of the memory limit. Expected: the `f16x2` version 80% or more.
- [ ] **Step 4: Lesson 1** (memory-bound work, coalescing, `f16x2`, why `erf` costs little here).
- [ ] **Step 5: Commit** — `feat(gpu): add the element-wise kernels and lesson 1`.

---

### Task 5: LayerNorm (lesson 2)

**Files:** `kernels/mod.rs`, `ops/layernorm.rs`, `tests/layernorm.rs`, `gpu-bench.rs`, `docs/cuda-lessons/02-layernorm.md`

**Interfaces:**
- Produces: `ops::layernorm(gpu, arena, x: &Tensor /*[rows, d]*/, gamma: &Tensor, beta: &Tensor, y: &Tensor, eps: f32) -> Result<(), GpuError>`, `F16` (also `F32` for the test), `eps = 1e-5`. One block of `ROW_BLOCK` threads per row, two `block_sum` calls (mean, then variance of `x - mean`, the two-pass form; write why one-pass `E[x²] - E[x]²` loses precision in fp32).
- [ ] **Step 1: Failing tests** vs CPU f64: `[1500, 1280]`, `[7, 1000]`, `[3, 64]`, plus `layernorm_large_offset`: rows with mean 1000 and spread 1 (one-pass would fail here). f16 `Tol { abs: 2e-3, rel: 1e-2 }`, f32 `Tol { abs: 1e-5, rel: 1e-4 }`.
- [ ] **Step 2: Run to fail, implement, run to pass.**
- [ ] **Step 3: Bench** `layernorm_f16 1500x1280`: 80% or more of the memory limit (x read, y written).
- [ ] **Step 4: Lesson 2. Step 5: Commit** — `feat(gpu): add LayerNorm and lesson 2`.

---

### Task 6: Softmax (lesson 3)

**Files:** `kernels/mod.rs`, `ops/softmax.rs`, `tests/softmax.rs`, `gpu-bench.rs`, `docs/cuda-lessons/03-softmax.md`

**Interfaces:**
- Produces: `ops::softmax_rows(gpu, arena, x: &Tensor /*[rows, n]*/, scale: f32, causal_offset: Option<usize>) -> Result<(), GpuError>`, in place, `F16` in and out, f32 math: `x[r][j] = exp(s·x[r][j] - m) / Σ`. With `Some(o)`, entries with `j > r + o` are `-inf` before the softmax (Phase 1b decoder uses it, the encoder passes `None`).
- Kernels: `softmax_3pass_f16` (max pass, sum pass, write pass) and `softmax_online_f16` (one read pass that keeps a running max and rescales the running sum, then one write pass).
- [ ] **Step 1: Failing tests** vs CPU f64: rows of length 1500 and 7, large values (±60 after scale) with no `inf`/NaN, the causal mask with `Some(0)` on a `[4, 4]` matrix.
- [ ] **Step 2: Run to fail, implement both kernels, run to pass.**
- [ ] **Step 3: Bench** `[20·1500, 1500]` (one encoder layer of scores) for both kernels.
- [ ] **Step 4: Lesson 3. Step 5: Commit** — `feat(gpu): add softmax and lesson 3`.

---

### Task 7: GEMM lesson and attention (lessons 4 and 5)

**Files:** `kernels/mod.rs`, `ops/gemm_lesson.rs`, `ops/attention.rs`, `tests/gemm_lesson.rs`, `tests/attention.rs`, `gpu-bench.rs`, `docs/cuda-lessons/04-gemm.md`, `docs/cuda-lessons/05-attention.md`

**Interfaces:**
- Produces:
  - Lesson GEMM kernels (product code keeps cuBLAS): `gemm_naive_f16` (one thread per output), `gemm_tiled_f16` (shared-memory tiles), `gemm_regtile_f16` (each thread computes a small block of outputs in registers), `gemm_tc_f16` (tensor cores through `cuda_device::wmma` or `mma.sync m16n8k8`, the `sm_75` forms). Host: `ops::gemm_lesson(gpu, arena, kind: GemmKind, a, b, c, m, n, k)` with row-major `C = A·B`, f16 in and out, f32 accumulate.
  - `ops::attention(gpu, arena, blas, q, k, v, out: &Tensor /*each [heads, seq, 64]*/, scratch: &Tensor /*[heads, seq, seq]*/, kind: AttnKind) -> Result<(), GpuError>` with `AttnKind::{Naive, Flash}`. Naive: batched cuBLAS `Q·Kᵀ` into `scratch`, `softmax_rows` with scale `1/√64`, batched cuBLAS `·V`. Flash: one kernel, tiles of K and V in shared memory, online softmax, no `scratch` read or write.
  - `ops::split_heads` / `ops::merge_heads`: `[seq, heads·64]` ↔ `[heads, seq, 64]` copies, in `ops/attention.rs`.
- [ ] **Step 1: Failing tests**
  - GEMM: each lesson kind vs cuBLAS at `m = 67, n = 129, k = 96` and `1500 × 1280 × 1280`, `Tol { abs: 3e-2, rel: 2e-2 }`.
  - Attention: both kinds vs a CPU f64 attention at `heads = 2, seq = 37, d = 64`, and Flash vs Naive at `heads = 20, seq = 1500, d = 64`, `Tol { abs: 3e-3, rel: 2e-2 }`.
- [ ] **Step 2: Run to fail, implement in the listed order, running the tests after each kernel.**
- [ ] **Step 3: Bench** all GEMM kinds at `1500 × 1280 × 1280` with % of the tensor limit next to cuBLAS, and both attention kinds at `20 × 1500 × 64`. `ops::attention` callers in Task 10 use the faster kind (write which in the lesson).
- [ ] **Step 4: Lessons 4 and 5** (lesson 4 also explains the row-major/column-major swap in `blas.rs`).
- [ ] **Step 5: Commit** — `feat(gpu): add the GEMM lesson kernels, attention and lessons 4-5`.

---

### Task 8: conv1d (lesson 6)

**Files:** `kernels/mod.rs`, `ops/conv1d.rs`, `tests/conv1d.rs`, `gpu-bench.rs`, `docs/cuda-lessons/06-conv1d.md`

**Interfaces:**
- Produces: `ops::conv1d(gpu, arena, blas, x: &Tensor /*[c_in, t_in]*/, w: &Tensor /*[c_out, c_in, 3]*/, y: &Tensor /*[c_out, t_out]*/, stride: usize, scratch: Option<&Tensor>, kind: ConvKind) -> Result<(), GpuError>`. Kernel size 3, padding 1, `t_out = (t_in + 2 - 3) / stride + 1`, `F16`.
- `ConvKind::Direct` is one kernel. `ConvKind::Im2col` writes `[c_in·3, t_out]` into `scratch` and calls cuBLAS.
- `conv1d` adds no bias and no GELU. The encoder calls `bias_gelu` after it (Task 10).
- [ ] **Step 1: Failing tests** vs CPU f64 for both kinds: conv1 shape (`128 → 1280`, `t = 3000`, stride 1) and conv2 shape (`1280 → 1280`, `t = 3000 → 1500`, stride 2), plus a tiny `[2, 5]` case checked by hand.
- [ ] **Step 2: Run to fail, implement, run to pass.**
- [ ] **Step 3: Bench** both kinds for both shapes. Task 10 uses the faster one per shape.
- [ ] **Step 4: Lesson 6. Step 5: Commit** — `feat(gpu): add conv1d and lesson 6`.

---

### Task 9: Log-mel spectrogram (lesson 7)

**Files:** `kernels/mod.rs`, `ops/mel.rs`, `crates/jeuvoice-whisper/src/cuda/mod.rs`, `crates/jeuvoice-whisper/src/cuda/mel.rs`, `crates/jeuvoice-whisper/src/mel_cpu.rs`, `crates/jeuvoice-whisper/Cargo.toml` (feature `cuda-kernels = ["dep:jeuvoice-gpu"]`), tests `crates/jeuvoice-whisper/tests/mel_cpu.rs` and `crates/jeuvoice-whisper/tests/cuda_mel.rs`, `docs/cuda-lessons/07-mel.md`

**Interfaces:**
- Produces:
  - `jeuvoice_whisper::mel_cpu::log_mel(samples: &[f32], filters: &[f32] /*[128, 201]*/) -> Vec<f32> /*[128, n_frames]*/`: a direct CPU port of whisper.cpp `log_mel_spectrogram` (`whisper.cpp:3170`): reflect-pad 200 at the start, 200 + 480,000 zeros at the end, periodic Hann window of 400, power spectrum of bins 0–200, mel = filters × power, `log10(max(v, 1e-10))`, clamp to `max - 8`, then `(v + 4) / 4`. It is the test oracle for the GPU mel.
  - GPU kernels in `jeuvoice-gpu`: `frames_hann_f16` (padded samples → `[n_frames, 400]` windowed frames), `power_mel_f16` helpers as needed, `log_clamp_norm_f32` (log10, global max with a block reduction, clamp, scale). The DFT is a GEMM: frames `[n_frames, 400]` × a fixed `[400, 402]` matrix of `cos` and `sin` columns, then power, then the mel GEMM `[n_frames, 201] × filtersᵀ`.
  - `jeuvoice_whisper::cuda::mel::GpuMel::new(gpu, blas, filters) -> Result<GpuMel, WhisperError>`. `GpuMel` has its own arena for frames and the spectrum.
  - `GpuMel::window_into(&self, samples_30s: &[f32], arena: &Arena, out: &Tensor /*[128, 3000] F16*/) -> Result<(), WhisperError>` writes the mel of one window into a tensor of another arena (the encoder input, Task 10).
- [ ] **Step 1: Failing tests**
  - `mel_cpu_matches_reference` (`#[ignore]`, needs Task 3 files): `log_mel` of `samples` vs the reference `mel`, `Tol { abs: 0.05, rel: 0.05 }` (HF pads both ends with reflection, whisper.cpp pads the end with zeros: write the measured difference in the lesson).
  - `gpu_mel_matches_cpu` (feature `cuda-kernels`): random 20 s audio, `GpuMel` vs `log_mel`, `Tol { abs: 2e-2, rel: 2e-2 }` (fp16 DFT).
  - `mel_short_and_empty_audio`: 0 samples and 1 sample give a `[128, 3000]` window with no NaN, on CPU and GPU.
- [ ] **Step 2: Run to fail, implement the CPU port, then the GPU path, run to pass.**
- [ ] **Step 3: Bench** one 30 s window, GPU vs CPU time (CPU timed with `std::time::Instant` in the test with `--nocapture`).
- [ ] **Step 4: Lesson 7. Step 5: Commit** — `feat(whisper): add the log-mel spectrogram on the GPU and lesson 7`.

---

### Task 10: The encoder

**Files:** `crates/jeuvoice-whisper/src/cuda/encoder.rs`, `crates/jeuvoice-whisper/src/cuda/weights.rs`, `crates/jeuvoice-whisper/tests/cuda_encoder.rs`, `docs/development-log.md`

**Interfaces:**
- Produces:
  - `cuda::weights::EncoderWeights::load(gpu: &Gpu, model: &GgmlModel) -> Result<EncoderWeights, WhisperError>`: uploads every `encoder.*` tensor as `F16` into one weights arena (biases and norm weights too), by the names found in Task 2.
  - `cuda::encoder::Encoder::new(gpu, blas, weights) -> Result<Encoder, WhisperError>` plans one activations arena for one 30 s window. The sizes come from the hparams. The plan includes the `[128, 3000]` `F16` mel input tensor.
  - `Encoder::input(&self) -> (&Arena, &Tensor)` gives the arena and the mel input tensor. `GpuMel::window_into` writes into it, and the reference test uploads into it.
  - `Encoder::run(&mut self) -> Result<&Tensor /*[1500, 1280] F16*/, WhisperError>` encodes the current input.
  - `Encoder::run_with_taps(&mut self, taps: &mut dyn FnMut(&str, &Arena, &Tensor)) -> Result<&Tensor, WhisperError>` also calls `taps` after each step. The names are `conv1`, `conv2`, `embed`, `block_00` … `block_31` and `ln_post`, the same as the reference file.
  - Block math (OpenAI Whisper): `x = x + out(attn(ln1(x)))`, `x = x + mlp2(gelu(mlp1(ln2(x))))`. Q and V have biases, K has none, `out` has a bias. Attention scale `1/√64`. Conv: `gelu(conv1(mel))`, `gelu(conv2(·))` with stride 2, transpose to `[1500, 1280]`, add `encoder.positional_embedding`.
- [ ] **Step 1: Failing test** `#[ignore = "needs the model and the Task 3 reference files"] encoder_matches_reference`: for each of the 3 clips, upload the reference `mel` (not our mel) into `Encoder::input`, call `run_with_taps`, and compare each tapped tensor with the reference. Tolerance per tap, written in the test, starting at `Tol { abs: 5e-2, rel: 5e-2 }` for `block_31` / `ln_post` (fp16 through 32 layers). Also assert no `inf` or NaN in any tap. Print the `Diff` of each tap.
- [ ] **Step 2: Run it to see it fail, implement `weights.rs` and `encoder.rs`, run it to see it pass.** If a tap is out of tolerance, find the first tap that is out and fix that layer before going on (superpowers:systematic-debugging).
- [ ] **Step 3: Measure.** Add `#[ignore] encoder_timing`: one warm-up window, then the mean GPU time of 5 windows with `bench::time_on_stream`, and the VRAM of both arenas. Write the numbers and the per-tap `Diff` values to `docs/development-log.md`. There is no speed target in 1a. Phase 1b compares the whole transcription with whisper.cpp.
- [ ] **Step 4: Commit** — `feat(whisper): run the Whisper encoder on Rust kernels`.
