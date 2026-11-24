# Rust CUDA inference — design (Phase 0 and Phase 1)

Date: 2026-11-24. Status: waiting for approval. Reviewed by a builder check.

## Why

The user wants three things, in this order of weight:

1. **Learn CUDA kernel programming in practice.** This is the main reason. Every kernel gets a lesson that says what it does, why the model needs it, and how it gets faster.
2. **Remove the Python setup.** Today the voice engine is a Python process (`engine/`, PyTorch + OmniVoice). Its setup takes 8 GB of disk and 10–15 minutes on the first start.
3. **Faster generation, and less VRAM if possible.**

The user decided: all inference code is Rust, and the kernels are also Rust, written with NVIDIA's CUDA Rust platform (`github.com/NVIDIA/cuda-rust`, the **cuda-oxide** SIMT model). The large matrix multiplies use **cuBLAS**.

## The whole project, and what this document covers

The full work is five phases. Each phase gets its own design, plan and review. **This document covers Phase 0 and Phase 1.** Phases 2–5 are listed only to show the order.

| Phase | Content | Python goes away? |
|---|---|---|
| **0. GPU base** | Toolchain, build integration, the runtime crate, cuBLAS, weight loading, test and benchmark harness, lesson 0 | no |
| **1. Whisper** | Whisper large-v3-turbo in Rust kernels, in place of whisper.cpp on the GPU path | no |
| 2. OmniVoice main model | Qwen3-0.6B backbone (28 layers, hidden 1024, 16/8 GQA heads), masked iterative decoding with `num_step` passes, classifier-free guidance, LoRA merge | no |
| 3. Audio decoder | The DAC decoder of the Higgs Audio v2 tokenizer (conv + snake), on the GPU in fp16 | no |
| 4. Clone encoder | HuBERT + DAC encoder, for voice-clone prompts | no |
| 5. Python removal | Port `dataprep.py` and `fetch_adapter` to Rust, remove `engine/` and the Python setup step | **yes** |

Whisper comes first (decided with the user): it is smaller, it already works, and whisper.cpp gives a fixed speed and accuracy target to measure against. The kernels it needs (LayerNorm, softmax, attention, GELU, GEMM, conv1d) are mostly the same as OmniVoice needs.

## Goals

### Phase 0
1. The project builds with cuda-oxide kernels on this Windows machine with one command, through `scripts\cargo.cmd`.
2. A new crate `crates/jeuvoice-gpu` holds the runtime base (see "Phase 0 parts").
3. cuBLAS GEMM in fp16 with fp32 accumulate works from Rust on the same stream as our kernels.
4. Every kernel has a test against reference values and a benchmark with CUDA events.
5. Lesson 0 is written (see "Lessons").

### Phase 1
1. A new type `CudaWhisper` implements the existing `Transcriber` trait (`crates/jeuvoice-whisper/src/lib.rs`). The app changes only where it loads the transcriber (`crates/jeuvoice-app/src/main.rs`).
2. It reads the model file that users already have: `ggml-large-v3-turbo.bin` (fp16). There is no new download.
3. It gives the same results as the current whisper.cpp path (see "Success criteria").
4. It is not slower than whisper.cpp, and it uses no more VRAM.
5. Every kernel in it has a lesson.

## Non-goals

- Training. Training stays in Python on Kaggle. Only the trained files (`.safetensors`) must load in Rust later (Phase 2).
- A general tensor library. The runtime holds only what these models need: no autograd, no graph compiler, no dynamic shapes beyond sequence length.
- macOS GPU support. On macOS, Transcribe keeps whisper-rs with Metal.
- A CPU version of the new kernels. On a machine without a usable NVIDIA GPU, Transcribe keeps whisper-rs on the CPU. **[Claude's call: today the app falls back to the CPU, and this keeps that behavior.]**
- cutile-rs (the Tile model). It is only tested on Linux and JIT-compiles on the user's machine. One lesson may compare it later, outside these phases.

## Anti-goals

- A faster Transcribe that gives worse text or worse word times than today.
- An app that does not start, or crashes, on a machine without CUDA or with an old driver. The CUDA path must fail with a clear error and the app must use whisper-rs.
- Kernels that work but that the user cannot follow. A kernel without its lesson is not done.
- A build that only works on this one machine with hand-set paths.

## Facts the design rests on (checked on 2026-11-24)

Toolchain test in `D:\MLProjelerim\cuda-rust-spike` (throwaway code, outside the repo):

- cuda-rust commit `5e5c77b19a4b3204d6542bab2f0fde2a4eed824b` (2026-11-23). It is an alpha research project. Expect API changes.
- cuda-oxide supports `x86_64-pc-windows-msvc`. It needs Rust **1.99.1** with `rust-src`, `rustc-dev`, `llvm-tools`, CUDA 13.0+, LLVM/libclang, and `libffi` from vcpkg. `cargo oxide doctor` passes on this machine (CUDA 13.4, driver 621.14, RTX 3050 Ti Laptop, sm_86, 4 GB).
- `C:\mingw64` on PATH breaks the `cuda-bindings` build: libclang reads the MinGW headers. `scripts\cargo.cmd` already gives bindgen the MSVC target and include folders for whisper-rs. The same fix must apply to cuda-bindings.
- A Rust RMSNorm kernel (one block of 256 threads per row, warp shuffle + shared memory reduction) on 4096×1024 fp32: error vs CPU 1.9e-6, 190 µs per launch, 176 GB/s. The card's memory limit is about 192 GB/s.
- The PTX shows `shfl.sync.bfly`, `bar.sync` and `sqrt.rn.f32` as planned. The default PTX target is `sm_80`. The driver JIT-compiles it for sm_86.
- cuda-oxide supports Volta and newer (`sm_70+`). A kernel that uses only basic features can target `sm_75`.
- Shared memory access is `unsafe` unless the kernel uses the Lean proof (`#[verify]`).
- `cargo oxide` is regular cargo with extra `CARGO_ENCODED_RUSTFLAGS` and profile settings.

The current app:

- `whisper-rs 0.16.0` (whisper.cpp) with the `cuda` feature on Windows and `metal` on macOS. Greedy decoding, `best_of: 1`, `token_timestamps: true`, DTW with the `LargeV3Turbo` preset. Word times come from `t_dtw` (when it is set) or from `t0`/`t1`.
- `scripts\cargo.cmd` builds for `CMAKE_CUDA_ARCHITECTURES=75;86;89`. The release runs on Turing, Ampere and Ada cards.
- The installer already ships `cublas64_13.dll` and `cublasLt64_13.dll` (`packaging/windows/collect.ps1`). cuBLAS adds nothing to the download.
- Measured speed (README): 69 s of Turkish speech in 3 files transcribes in 4.3 s.

## Approaches considered

| | Approach | Trade-off | Result |
|---|---|---|---|
| A | Rust host code (`cudarc`) + CUDA C++ kernels compiled by `nvcc` | Most CUDA material is in C++, so the lessons transfer directly. But the kernels are not Rust. | Not picked |
| **B** | **Rust host code + Rust kernels with cuda-oxide + cuBLAS** | Everything is Rust, from NVIDIA. Alpha project, pinned toolchain, extra build tools. The toolchain test passed. | **Picked by the user** |
| C | candle as the base, with only the hot kernels written by hand | The fastest way to remove Python. But the user writes few kernels, so it does not serve the main goal. | Not picked |

cutile-rs was dropped: Linux only, and it hides the thread-level concepts the user wants to learn.

## Phase 0 parts

### Toolchain and build

- `rust-toolchain.toml`: pin `1.99.1` with the cuda-oxide components. **[Claude's call: cuda-oxide needs this exact version.]** All workspace crates must build with it.
- `scripts\cargo.cmd`: add the cuda-oxide environment (CUDA paths, `LIBCLANG_PATH`, vcpkg `libffi`, MinGW removed from PATH). A missing tool gives one clear line that says what to install.
- **Build integration — first task of Phase 0, a gate.** Find out which of these two ways works, and use it:
  1. The whole workspace builds with the cuda-oxide codegen backend (`cargo oxide build -p jeuvoice-app`). The kernels sit in `jeuvoice-gpu` as a `#[cuda_module]`.
  2. Only a kernels crate builds with `cargo oxide`. Its PTX goes into the app as bytes (`include_bytes!`), and the app crate builds with regular cargo. The cuda-oxide examples `cuda_module_in_lib` and `cross_crate_kernel` show the parts.

  Pick 1 if the app (rune, cpal, keyring, whisper-rs) builds and runs with it. Otherwise pick 2. Write the result and the reason in `docs/development-log.md`. (The examples are in `cuda-oxide/crates/rustc-codegen-cuda/examples/`.)
- **A build without the cuda-oxide tools must still work.** The repo is public, and macOS uses the same workspace. The new GPU path sits behind a Cargo feature, `cuda-kernels`, on `jeuvoice-whisper`. It is off by default. `scripts\cargo.cmd` and the release script turn it on. Without the feature, the app builds as today with whisper-rs only. **[Claude's call.]**
- The `rust-toolchain.toml` pin to 1.99.1 applies to all platforms. 1.99.1 is a stable release, so macOS builds still work.
- Pin `cuda-rust` to commit `5e5c77b19a` as a git dependency.
- GPU targets: the release must run on `sm_75`, `sm_86` and `sm_89`, as today. Kernels that need `sm_80` features (for example `mma.sync` m16n8k16 or `cp.async`) need a second version for `sm_75`, or the app must use whisper-rs on Turing cards. The builder decides per kernel and writes it in the lesson.
- `packaging/windows/collect.ps1` and `scripts/make-installer.cmd` must build the release with the new build steps.

### The `jeuvoice-gpu` crate

- **Device and context:** open device 0, read its name, compute capability and free VRAM. All errors map to one error type with plain words.
- **Memory:** allocate the weights once at model load. Allocate the activations in one buffer per model with fixed offsets (an "arena") sized for the longest input. **[Claude's call: one allocation makes VRAM use fixed and easy to measure. No allocator in the hot loop.]**
- **Tensor view:** a device pointer, shape, strides and dtype (`f16`, `f32`, `i32`). No ownership logic beyond the arena.
- **cuBLAS:** thin hand-written `extern "C"` bindings, loaded at run time from `cublas64_13.dll` / `cublasLt64_13.dll`. The app must start when the DLL or the GPU is missing. Functions: `cublasGemmEx` and `cublasGemmStridedBatchedEx` (fp16 in, fp32 accumulate, tensor cores). The stream must be the cuda-core stream of our kernels. The lesson explains the row-major vs column-major swap.
- **Weight loading:** a reader for the ggml file format (Phase 1) and for `.safetensors` (needed in Phase 2, written in Phase 2).
- **Tests:** each kernel has a test that compares its output with a reference tensor. The tolerance is set per kernel and written in the test.
- **Benchmarks:** one benchmark binary that times each kernel with CUDA events and prints µs, GB/s or TFLOP/s, and the percent of the card's limit.

### Reference values

- A dev-only tool, `tools/reference/`, writes reference tensors (inputs, every layer output, final tokens) to `.safetensors` files. It uses Python with `transformers` and the `openai/whisper-large-v3-turbo` weights.
- It is a developer tool. It never ships and the app never calls it. **[Claude's call: Python is the most direct way to get trusted per-layer values. The goal is to remove Python from the user's install, not from the developer's machine.]**
- The reference files are not committed (they are large). The tool writes them to `target/reference/`. Tests that need them are `#[ignore]`, like the current real-model tests.

## Phase 1: Whisper

### What runs where

| Step | Where | Kernel or library | Lesson |
|---|---|---|---|
| Audio to 16 kHz mono | CPU | existing `jeuvoice_core::audio::to_whisper` | — |
| Log-mel spectrogram (n_fft 400, hop 160, 128 mel bands) | GPU | framing kernel + STFT as a GEMM with a fixed DFT matrix (cuBLAS) + power, mel GEMM, log and clamp kernel | 7 |
| Encoder conv stem (2 × conv1d k=3, GELU) | GPU | conv1d kernel (direct, or im2col + GEMM: the builder measures both) | 6 |
| Encoder, 32 layers, 1500 positions | GPU | LayerNorm, GEMM, attention, GELU, add | 1–5 |
| Decoder, 4 layers | GPU | embedding lookup, LayerNorm, self-attention with a KV cache, cross-attention, GEMM, logits | 8 |
| Token choice and the timestamp rules | GPU logits, CPU rules | argmax kernel, logit suppression | 8 |
| Word times (DTW on the alignment heads) | cross-attention weights on the GPU, DTW on the CPU | — | 9 |
| Tokens to words | CPU | existing `tokens_to_words` | — |

- Number format: fp16 weights and activations, fp32 accumulate, LayerNorm and softmax math in fp32. **[Claude's call.]**
- The DFT matrix for the STFT means no cuFFT DLL in the installer. **[Claude's call.]**
- The decode loop uses CUDA Graphs (one graph per decoder step) when the lesson shows a gain.

### Decoding behavior

`CudaWhisper` must decode the same way as `WhisperRs` does today. That means `whisper_full` in the whisper.cpp version inside `whisper-rs-sys 0.15.0`, with the parameters set in `lib.rs` and the whisper.cpp defaults for the rest: 30 s windows and seek, timestamp token rules, temperature fallback, no-speech check, and the text context from the previous window. The builder reads that whisper.cpp source and lists the rules in lesson 8 before writing the code.

### Fallback

- `WhisperRs` stays **with its CUDA build** (the `cuda` feature of whisper-rs) through Phase 1. It is the fallback and the measurement baseline. The decision to remove whisper.cpp CUDA comes after Phase 1, with the measured results. Because of this, Turing (`sm_75`) cards can use whisper.cpp CUDA, and an `sm_75` version of our kernels is not required in Phase 1. **[Claude's call.]**
- On load, the app tries `CudaWhisper` first. If CUDA does not start, or the GPU is older than `sm_75`, or the model does not fit in free VRAM, the app uses `WhisperRs` and writes the reason to the log.
- A setting is not needed for this. **[Claude's call.]**

### Success criteria

Test set: the 3 files of the README speed test (69 s), plus the reference clip in `tests/real_model.rs`.

1. **Text:** the word error rate between `CudaWhisper` and `WhisperRs` output is 2% or less on each file. Each difference is checked by hand and explained in the development log.
2. **Word times:** the median start-time difference per word is 50 ms or less.
3. **Speed:** the 69 s test set transcribes in 4.3 s or less (release build, same card).
4. **VRAM:** peak use is not more than whisper.cpp on the same test.
5. The existing tests and the `#[ignore]` real-model tests pass with `CudaWhisper`.
6. The app starts and transcribes on the CPU when CUDA is not available.

**Phase 1 is not done until criteria 1–6 all pass** (decided by the user). The plan must leave open time for optimization. Until all criteria pass, `CudaWhisper` is not the default: it runs only when the developer sets the `JEUVOICE_CUDA_WHISPER=1` environment variable. **[Claude's call: this keeps the app usable during the work.]**

## Lessons

- Folder: `docs/cuda-lessons/`. One file per lesson, numbered.
- Language: **Turkish.** Code comments stay English, as in the rest of the repo. **[Claude's call: the lessons are for the user. The repo stays English.]**
- Each lesson has the same parts:
  1. What the operation does in the model, with the formula.
  2. Why the model needs it, and how often it runs.
  3. A naive kernel first, with its measured speed.
  4. Each change that makes it faster, one at a time, with the reason and the new measured speed.
  5. The limit of the card for this kernel (memory or compute bound), and how close the kernel gets.
  6. What to look at in `cargo oxide inspect` (PTX) and in Nsight Compute.
  7. Small exercises for the user.
- Lesson plan for Phases 0 and 1:

| No | Subject | Teaches |
|---|---|---|
| 0 | GPU basics and the toolchain | threads, warps, blocks, the memory types, what PTX is, the RMSNorm kernel from the toolchain test |
| 1 | Element-wise kernels: bias add, GELU | memory-bound work, coalesced reads, vector loads (`f16x2`) |
| 2 | LayerNorm | reductions, warp shuffles, shared memory, fp32 math on fp16 data |
| 3 | Softmax | numerical stability, the online softmax |
| 4 | GEMM | cuBLAS (row-major vs column-major, tensor cores), then our own GEMM: shared memory tiles, register tiles, `mma.sync`, compared with cuBLAS |
| 5 | Attention | naive (GEMM + softmax + GEMM), then flash attention: tiles and the online softmax in one kernel |
| 6 | conv1d | direct convolution vs im2col + GEMM |
| 7 | Log-mel spectrogram | STFT as a matrix multiply |
| 8 | The decoder loop | KV cache, argmax, CUDA Graphs, the whisper.cpp decoding rules |
| 9 | Word times | cross-attention alignment heads, DTW |

A cuBLAS GEMM stays in the product if it is faster than our own GEMM. Our GEMM stays as a lesson and a test.

## Constraints

- Windows 10/11 x64, NVIDIA driver with CUDA 13.x (R580+), cards `sm_75` and newer, 4 GB VRAM must work.
- Rust 1.99.1, pinned.
- The cuda-rust project is alpha. Any problem with it goes to `docs/development-log.md` with the work-around.
- **If cuda-oxide blocks the work** (a compiler bug, a missing feature such as `mma.sync` or `f16`): patch it in a fork of `NVIDIA/cuda-rust` under the user's GitHub account, pin our dependency to the fork commit, and report the bug upstream (decided by the user). The kernels stay in Rust. Switching to CUDA C++ is not allowed without a new decision from the user.
- No change to the app UI in Phases 0 and 1.

## Left to the builder

- The build integration choice (way 1 or 2 above), after the test.
- The block sizes, tile sizes and launch settings of each kernel, based on measurements.
- Direct conv1d or im2col + GEMM.
- Which kernels need an `sm_75` version.
- When to use the Lean proof (`#[verify]`) and when to use `unsafe` with a `SAFETY` comment.
- The exact tolerance of each kernel test.
- The file layout inside `jeuvoice-gpu` and `jeuvoice-whisper`.
