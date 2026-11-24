# AGENTS.md

Rules and commands for coding agents in this repo. Read this first.

**Current work:** the Rust CUDA inference project. Read `docs/superpowers/handoff/2026-11-24-rust-cuda-handoff.md` for where it stopped and what comes next.

## The project

JeuVoice is a Windows desktop app in Rust for Turkish speech: transcription (Whisper), text-to-speech (OmniVoice), voice making and training. The README has the user view.

| Crate / folder | Content |
|---|---|
| `crates/jeuvoice-app` | the app (rune UI) |
| `crates/jeuvoice-core` | audio, projects, subtitles, text tools |
| `crates/jeuvoice-whisper` | Whisper: whisper-rs today, the ggml reader, the new GPU path behind the `cuda-kernels` feature |
| `crates/jeuvoice-gpu` | **all GPU kernels** (Rust, cuda-oxide), cuBLAS, the arena, test and bench helpers |
| `crates/jeuvoice-engine-client`, `crates/jeuvoice-kaggle` | the Python engine client, the Kaggle API |
| `engine/` | the Python voice engine (Phase 5 removes it) |
| `docs/cuda-lessons/` | CUDA lessons in Turkish, one per kernel |
| `tools/reference/` | developer-only Python tool for reference tensors |

## Commands (Windows, PowerShell)

| Command | What it does |
|---|---|
| `scripts\cargo.cmd test` | plain tests of the default members. No cuda-oxide tools needed. |
| `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu` | GPU kernel tests through the cuda-oxide backend |
| `scripts\cargo.cmd oxide test -- -p jeuvoice-whisper --features cuda-kernels` | Whisper GPU tests (after plan 1a Task 9 adds the feature) |
| `scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release` | build the benchmark table |
| `target\release\gpu-bench.exe [filter]` | run it. Put `%CUDA_PATH%\bin\x64` on PATH first, for cuBLAS. |

Always use `scripts\cargo.cmd`, not `cargo`. It sets up Visual Studio, bindgen and the cuda-oxide environment. Tests that need the model file or the test audio are `#[ignore]`: run them with `-- --ignored` or `--include-ignored`.

The GPU tools on this machine: Rust 1.99.1 (pinned), CUDA 13.4, LLVM (`C:\Program Files\LLVM`), vcpkg with libffi (`D:\vcpkg`), `cargo-oxide`. The install list is in `docs/development-log.md`, section "Status (Rust CUDA, Phase 0)".

## Hard rules

1. **Kernels are Rust** (`#[cuda_module]`, `#[kernel]` in `crates/jeuvoice-gpu/src/kernels/mod.rs`). Do not write CUDA C++. If cuda-oxide blocks the work, patch a fork of `NVIDIA/cuda-rust` under the user's GitHub account and pin the dependency to it.
2. **Device code lives only in `jeuvoice-gpu`.** A second crate with kernels brings back the PTX file race.
3. **Arena kernels take raw device pointers and a length.** Every `unsafe` block has a `SAFETY` comment that says why it is safe.
4. **PTX target `sm_75`.** No kernel may need `sm_80` features.
5. **Numbers:** fp16 data, fp32 accumulate. Norm and softmax math in fp32.
6. **Tests first.** Write the test, run it, watch it fail, then write the code. Every kernel has a test against a CPU f64 reference or a PyTorch reference tensor, and a `gpu-bench` row.
7. **Every kernel has a lesson** in `docs/cuda-lessons/NN-name.md`, in **Turkish**, with these 7 parts:
   1. what the operation does in the model, with the formula
   2. why the model needs it, and how often it runs
   3. a naive kernel first, with its measured speed
   4. each change that makes it faster, one at a time, with the reason and the new measured speed
   5. the card's limit for this kernel and how close it gets
   6. what to look at in the PTX and in Nsight Compute
   7. exercises
8. **Code comments and docs other than the lessons are in English**, in ASD-STE100 style: active voice, short sentences, no contractions, no semicolons, simple words.
9. **Measure, do not guess.** Check a speed change with `gpu-bench`, and check the PTX in `target\oxide-ptx\jeuvoice_gpu.ptx`.
10. **Errors in plain words** that the user can act on (see `GpuError` and `WhisperError`).
11. **Do not change the app UI** in Phase 1.
12. **Git:** work on a branch. Commit after each plan task. Merge to `main` and push only when the user says so.

## Pitfalls already found

- `C:\mingw64` on PATH breaks bindgen. `scripts\cargo.cmd oxide` removes it.
- `vcvars64.bat` sets `VCPKG_ROOT` to the Visual Studio copy, which has no libffi. The script looks in `JEUVOICE_VCPKG`, `C:\vcpkg`, `D:\vcpkg` and `%USERPROFILE%\vcpkg`.
- `cargo oxide run` gives the words after `--` to the program. Use `build`, then run the exe.
- `DisjointSlice::from_raw_parts` needs a type, for example `let out: DisjointSlice<f32> = ...`.
- The built-in `f16` needs `#![feature(f16)]` (already in `jeuvoice-gpu`).
- Two GPU test runs at the same time can run out of the 4 GB of video memory. Run one at a time.
- **Save every file as UTF-8 without a BOM.** One edit tool read UTF-8 files as Windows-1252 and saved them again, so `Q·Kᵀ` became `QÂ·Káµ€` in 7 files. Commit `fix: repair double-encoded comments` repaired them. After an edit, check that non-ASCII characters (`·`, `ᵀ`, `²`, Turkish letters) still look right.
- An `[f16; 2]` load compiles to two 2-byte loads. That is fine here: the L1 cache merges them (lesson 1).
