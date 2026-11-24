# Development log

These are the status notes from each build plan, as they were at the end of that plan. Some details are old. For example, the built-in voice `ses_0007` is now named **Sami**, and the tr2 adapter is on Hugging Face now. For the current state, read the [README](../README.md) and the [user guide](user-guide.md).

## Status

Plan 1 (Transcribe) is done.

### What works

- Drop audio or video files (`.mp4`, `.ogg`, `.wav`, `.mp3` and more). JeuVoice reads the sound itself. It needs no ffmpeg.
- A batch runs one file at a time. If one file is bad, the others still finish. The bad file shows a plain error.
- Paths with Turkish letters and spaces work.
- Words come with start and end times. You can play the take, click a word to jump to it, and fix a word.
- You can paste or open a script. JeuVoice marks the words that differ from what it heard.
- You can export SRT, VTT or plain text.

### Measured speed

Checked on 2026-11-09 with the release build. Model: `large-v3-turbo`. Language: Turkish.

| File | Sound | Time | Speed |
|---|---|---|---|
| my_voice2.mp4 | 33.6 s | 2.3 s | 14.9 min per min |
| my_voice.ogg | 27.0 s | 1.2 s | 22.1 min per min |
| çekim 1.wav | 8.4 s | 0.8 s | 10.7 min per min |
| All three | 69.0 s | 4.3 s | 16.2 min per min |

"min per min" means minutes of sound for each minute of work. The model ran on the GPU (NVIDIA, CUDA). Short files look slower, because each file has a fixed start cost. The model load time is not in these numbers.

The script check of one take against its own text gave 20 words the same and 0 different.

To repeat the check:

```
scripts\cargo.cmd test -p jeuvoice-app --release --test real_check -- --ignored --nocapture
```

### Known limits

- Word start times can be 0.1 to 0.3 seconds late.
- The macOS build is not tested.
- The RunPod, Voice and Train tabs are not in this plan.
- A file that is not audio gives a message with a technical detail at the end.
- An automatic check cannot see the screen or hear sound. The window opens and stays open for 10 seconds without a crash. The list below needs your eyes and ears.

### Try it

1. Start the app. Download the model when it asks.
2. Drop an `.mp4`, an `.ogg`, and a file from a folder with Turkish letters.
3. Wait until all three are done.
4. Press play. Check that the sound and the cursor move together.
5. Click a word. Check that playback jumps to that word.
6. Fix one wrong word. Check that the new text stays.
7. Paste a script. Check that the marked words are the real differences.
8. Export SRT. Open it next to the video in a player. Check the timing.

## Status (Plan 2, Voice)

The Voice tab generates Turkish voice-overs with the OmniVoice model and the tr2 adapter, keeps projects in `Documents/JeuVoice`, and exports WAV + SRT.

### First start: model files

- The Hugging Face token lives in the OS keychain. Open Settings and paste a token with read permission, then Save. The token never lands in `settings.json`.
- Without a token, point "Use a local adapter folder" at a tr2 checkpoint folder on disk (for example `tts_lab/outputs/omnivoice_tr2/checkpoint-6000`). The local folder wins over every Hugging Face adapter.
- The default preset `ses_0007` installs on first start. It clones `ses_001_0007.wav` with the tr2 `checkpoint-6000` adapter.

### How to use it

1. Open the Voice tab. Pick a preset, paste a two-paragraph script, press **Generate**. The engine loads once ("Loading the voice model…"), then renders paragraph by paragraph.
2. Press play. Click a word to play from it.
3. Drag a region on the wave and press **Retake region**: three new takes appear. Press a take to hear it, then **Use this take** to splice it in. **+ Try 3 more** renders three more.
4. Double-click a word to fix how it sounds ("Say it as"): the fix is stored as a respell and the paragraph renders again.
5. A name in Latin letters gets an "Add to English terms?" hint next to the script check. Adding it puts the name in the project's English terms.
6. **Undo** and **Redo** (or Ctrl+Z / Ctrl+Y) step through project changes. Every change saves `project.json` at once.
7. **Export WAV** writes the 24 kHz timeline. **Export SRT** writes the timeline words.
8. The **Advanced** panel (closed by default) holds guidance, base seed, pause, paragraph pause, the term lists and per-paragraph guidance and seed. **Apply** saves them into the project; the next render uses them.

### Measured speed

Measured on this machine (Windows, NVIDIA RTX 3050 Ti Laptop GPU, 4.29 GB VRAM; 16 GB RAM, no page file) with the release build. Default preset: base `k2-fsa/OmniVoice` + tr2 adapter `checkpoint-6000`, reference clip `ses_001_0007`, `guidance_scale = 2.5`, `num_step = 32`:

| Step | Measured |
|---|---|
| Engine start + voice model load on CUDA | 25.5 s (2.37 GB VRAM used) |
| Render 2 paragraphs (5.6 s of audio) | 12.1 s (0.46 audio-sec per wall-sec) |
| Whisper `large-v3-turbo` on the CPU next to the loaded engine | 52.6 s, no out-of-memory |
| Retake one paragraph with a new seed + splice planning | 33.2 s |
| Export | 140,640-sample WAV + non-empty SRT |

Whisper ran on the CPU while OmniVoice stayed on CUDA: no CUDA out-of-memory on the 4 GB card. VRAM stayed at about 2.4 GB of 4.3 GB through render and retake.

Run it yourself:

```
scripts\cargo.cmd test -p jeuvoice-app --release --test real_voice -- --ignored --nocapture
```

### Known limits

- Retakes render whole paragraphs; a retake never renders a lone sentence.
- The term hint only flags Latin-letter words with inner capitals (`iPhone`, `NATO`).
- The macOS build is not tested, and the voice engine has no Mac spike result yet: the Voice tab stays off there.
- GPU cards under 8 GB run Whisper on the CPU while the voice model is loaded. Starting a Transcribe batch unloads the engine first; the next Voice job loads it again.

### Try it by ear

1. Set the Hugging Face token or the local adapter folder in Settings.
2. Paste your real script (two paragraphs to start). Press **Generate** and listen from start to end.
3. Compare a paragraph with `tts_lab/outputs/omni_tr2_para_gs25_s4/full.wav`. They should sound like the same voice.
4. Drag a region over one sentence and press **Retake region**. Use the take that sounds best.
5. Open **Advanced**, set guidance to 3, and retake the word "damara". Keep the take that reads it cleanly.

## Status (Plan 3, Train)

The Train tab takes clips to a Kaggle LoRA run: data (drop files or fetch Hugging Face datasets), start-from model, target (This PC, Kaggle, RunPod), and the run itself (script, start, progress, checkpoints). Each project lives in `Documents/JeuVoice/Training/<name>/` with `train_project.json` (sources, clips, settings, target) and `runs/<run>/` (manifests, bundle, notebook, `run.json`, outputs).

### Credentials

- Open the Train tab, **Add Kaggle key** on the Kaggle card: the Kaggle user name plus the key from `kaggle.com/settings` → API → Create New Token (or **Import kaggle.json**). Both legacy keys and new `KGAT_...` tokens work: legacy keys go as HTTP Basic, `KGAT_` tokens as a bearer token, like the official client.
- The key lives in the OS keychain (service `JeuVoice-Kaggle`), never in a file. If the app forgets the key between starts, type it in again.

### What goes up to Kaggle

- One **private** dataset per run: `<run>.zip` with the `bundle_<run>.txt` marker, `config/train.json` (the tr2 template with your steps/LoRA settings), `config/data.json`, `manifests/train.jsonl` + `dev.jsonl` (relative `audio/…` paths), and `audio.pack` (a tar of the cleaned 24 kHz WAVs under another suffix, because Kaggle unpacks every `.tar`). Excluded clips never go up.
- One **private** notebook running the shipped script: unpack, check every clip, extract audio tokens on Kaggle (`tokens/<split>/data.lst`), train with `accelerate launch -m omnivoice.cli.train` (older checkpoints pruned to `adapter_config.json` + `adapter_model.safetensors`), copy adapters plus `train.log` to `/kaggle/working/result`, print `JEUVOICE_RUN_DONE`.
- The base model (`k2-fsa/OmniVoice`) is downloaded on Kaggle's own connection, not uploaded.

### Time and cost

Kaggle T4 ×2 is free within your GPU quota: about an hour per 1,000 steps (6,000 steps ≈ 6 hours). The card scales the estimate with your steps.

Tiny real check on this machine (20 own-voice clips, steps 50, save every 25; release build):

| Step | Measured |
|---|---|
| Whisper text for 20 clips (CPU) | ~8 min |
| Bundle (10.4 MB) | seconds |
| Dataset upload | 4–12 s |
| Kernel queue + pip + token extraction + 50 steps | 9–16 min |
| Two checkpoints (`checkpoint-25`, `checkpoint-50`) download | seconds |
| Test render with `checkpoint-50` (fixed sentence) | 3.9 s of audio |

Repeat it yourself (needs `KAGGLE_TEST_USER`/`KAGGLE_TEST_KEY` in the env, the engine venv, the Whisper model and Kaggle GPU quota; the tts_lab clips stay read-only):

```
scripts\cargo.cmd test -p jeuvoice-app --release --test real_train -- --ignored --nocapture
```

### How to resume

- A run keeps going on Kaggle when the app closes. Its `runs/<name>/run.json` holds the dataset, kernel, start time and state; Queued/Running runs resume polling on the next start.
- On `Error` the tab shows the last 30 log lines and keeps the bundle: **Download bundle only** saves the zip plus the `.ipynb` for a manual run.

### Known limits

- Local training needs a 12 GB CUDA card; this PC's card turns This PC off with the reason. RunPod says "Coming later".
- A kernel title may not repeat a dataset title: the app names the dataset `<project> <run> data`.
- The client waits (up to 10 minutes) for a fresh dataset version to read `READY` before pushing.
- The loss curve reads `Eval Loss:` lines, so runs shorter than `eval_steps` show no curve.
- The macOS build is not tested.

## Status (Plan 4, Setup and installers)

The user guide for teammates is in [docs/user-guide.md](user-guide.md).

### Done for the MVP

- Locked engine packages for Windows (CUDA 12.6) and macOS.
- The app finds its files in dev mode, in the Windows install folder, and in a macOS `.app`.
- First-start setup makes the engine environment with a progress screen. An interrupted setup cleans up and starts again. A changed lock file runs setup again. An unchanged lock file keeps the environment.
- An engine from another JeuVoice version does not start. Settings shows **Repair**, which runs setup again.
- With no NVIDIA card, Settings shows "No supported graphics card found. Speech will be slow." Transcribe runs on the CPU. The voice engine does not start and says why.
- Windows installer: `scripts\make-installer.cmd` makes `dist\windows\jeuvoice-app_<version>_x64-setup.exe`. It installs per user with no admin rights. The uninstaller asks before it deletes the engine and the models. It never deletes `Documents\JeuVoice`.
- macOS: `scripts/make-dmg.sh` makes the `.app` and the `.dmg` on an Apple Silicon Mac.
- Tests: `scripts\cargo.cmd test --workspace` passes. The engine tests pass (`pytest`: 28 passed, 1 skipped. `-m gpu --gpu`: 1 passed).

### What you must still do

1. **Fresh-account test on Windows** (about 30 minutes): install on a second Windows user account or a VM. Fill in the test record in [packaging/windows/README.md](../packaging/windows/README.md).
2. **Signing on Windows** (about 10 minutes): run the `signtool` commands in [packaging/windows/README.md](../packaging/windows/README.md) with your certificate.
3. **Mac run** (about 1 hour): a teammate with an Apple Silicon Mac runs the Mac spike ([docs/mac-spike.md](mac-spike.md)) and `scripts/make-dmg.sh`, then signs and notarizes ([packaging/macos/README.md](../packaging/macos/README.md)).
4. **Upload the tr2 adapter to Hugging Face**, so that teammates without a local checkpoint can use the Voice tab.
5. **Checks by ear**: follow "Try it by ear" in the Plan 2 section.

### Known gaps

- The free-space check does not work on macOS. It never blocks setup there.
- The macOS build is not tested.

## Status (Rust CUDA, Phase 0: GPU base)

Plan: `docs/superpowers/plans/2026-11-24-rust-cuda-phase0.md`. Spec: `docs/superpowers/specs/2026-11-24-rust-cuda-inference-design.md`.

### Build way 1: the whole workspace builds with the cuda-oxide backend

Checked on 2026-11-24. With `rust-toolchain.toml` on 1.99.1, `cargo oxide build -- -p jeuvoice-app --release` built the release app in 26 min (most of it is whisper.cpp with CUDA). The app started and opened its window. So the kernels live inside the workspace, in `crates/jeuvoice-gpu`, and no separate PTX build step is needed.

`jeuvoice-gpu` is a workspace member but not a default member. A plain `cargo build` or `cargo test` leaves it out, so the app still builds without the cuda-oxide tools.

### Commands

| Command | What it does |
|---|---|
| `scripts\cargo.cmd test` | Plain tests of the default members. No cuda-oxide tools needed. |
| `scripts\cargo.cmd oxide test -- -p jeuvoice-gpu` | GPU tests through the cuda-oxide backend. |
| `scripts\cargo.cmd oxide build -- -p jeuvoice-gpu --bin gpu-bench --release`, then `target\release\gpu-bench.exe [filter]` | The kernel benchmark table. (`cargo oxide run` gives the words after `--` to the program, not to cargo.) |

### Tools for the GPU build (Windows)

1. Rust 1.99.1 with `rust-src`, `rustc-dev`, `llvm-tools` (rustup installs them from `rust-toolchain.toml`).
2. The CUDA 13 toolkit (13.4 here).
3. LLVM for `libclang.dll`: `winget install --id LLVM.LLVM -e`.
4. libffi from vcpkg: `vcpkg install libffi:x64-windows`. Set `JEUVOICE_VCPKG` to the vcpkg folder if it is not in `C:\vcpkg`, `D:\vcpkg` or `%USERPROFILE%\vcpkg`.
5. cargo-oxide: `cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide`, then `cargo oxide setup` once.

### Problems found and the fixes

- **MinGW on PATH breaks the cuda-bindings build.** libclang reads the MinGW headers and fails. `scripts\cargo.cmd oxide ...` removes every PATH entry with `mingw` in it.
- **vcvars64.bat sets `VCPKG_ROOT`** to the Visual Studio copy of vcpkg, which has no libffi. The script looks for `ffi.lib` in the usual folders instead.
- **The backend needs the exact toolchain.** With `channel = "stable"`, cargo-oxide stops with "the cuda-oxide main clone needs Rust 1.99.1".
- **Arena kernels take raw device pointers.** The synchronous cuda-oxide launch accepts `&DeviceBuffer<T>`, `&mut DeviceBuffer<T>` or raw pointers. An arena tensor is a part of one buffer, so kernels take `*const T` / `*mut T` and a length, and build a `DisjointSlice` inside with `from_raw_parts`.
- **Fixed: "embedded artifact payload is empty".** cuda-oxide writes the PTX of a crate to one file per crate name. Cargo built the `jeuvoice-gpu` lib and its lib test at the same time, so one build read the file while the other rewrote it. `jeuvoice-gpu` now has `[lib] test = false`, and all its tests are in `tests/`.
- **The PTX folder.** cuda-oxide writes `<crate>.ptx` and `.ll` files to the current folder. `scripts\cargo.cmd oxide` sets `CUDA_OXIDE_PTX_DIR` to `target\oxide-ptx`.
- **fp16 in kernels** uses the built-in `f16` type with `#![feature(f16)]`, as in the cuda-oxide examples.

### Measured speed (gpu-bench)

Checked on 2026-11-24, release build, RTX 3050 Ti Laptop (limits: 192 GB/s memory, 15.2 TFLOP/s fp16 tensor with fp32 accumulate).

| Kernel | Size | Time | Rate | % of limit |
|---|---|---|---|---|
| `gemm_f16` (cuBLAS) | 1500x1280x1280 | 364.0 µs | 13.5 TFLOP/s | 89% |
| `gemm_f16` (cuBLAS) | 1500x5120x1280 | 1473.3 µs | 13.3 TFLOP/s | 88% |
| `gemm_f16` (cuBLAS) | 4096x1024x1024 | 594.0 µs | 14.5 TFLOP/s | 95% |
| `rmsnorm_naive_f32` | 4096x1024 | 932.6 µs | 36.0 GB/s | 19% |
| `rmsnorm_f32` | 4096x1024 | 192.2 µs | 174.6 GB/s | 91% |
| `rmsnorm_f16` | 4096x1024 | 98.5 µs | 170.3 GB/s | 89% |

### Runs without the CUDA toolkit

`gpu-bench.exe` ran from a folder with only `cublas64_13.dll` and `cublasLt64_13.dll` next to it, with `PATH` set to `C:\Windows\System32;C:\Windows` and no `CUDA_PATH`. All rows ran with the same speed. So the embedded PTX needs only the NVIDIA driver (the driver compiles the PTX for the card on first load), and `--materialize-cubin` is not needed. The installer for Phase 1 needs only the two cuBLAS DLLs, as today.

### Fixes after the branch review

- **PTX target sm_75.** The cuda-oxide default target is sm_80, and the driver compiles PTX only for its target or newer cards. So an RTX 20 or GTX 16 card would open but fail to load the kernels. `scripts\cargo.cmd oxide` now sets `CUDA_OXIDE_TARGET=sm_75`. The test `ptx_targets_turing` checks the built PTX. With sm_75 PTX on the sm_86 card: `rmsnorm_f32` 199.6 µs (88%), `rmsnorm_f16` 98.7 µs (89%).
- **Checked sizes.** `Tensor` fields are private, and only `ArenaPlan::add` makes a `Tensor`. Sizes, offsets and the batched GEMM extents use checked math, so a huge shape or stride panics with a clear message instead of wrapping to a small size. `Element` is sealed to `half::f16`, `f32` and `i32`.
- **Workspace-wide commands.** `cargo test --workspace` and `cargo clippy --workspace` include `jeuvoice-gpu`, which builds only through `scripts\cargo.cmd oxide`. Use the default members (`scripts\cargo.cmd test`) for a plain build. Every build also needs GitHub access once, to fetch the pinned `NVIDIA/cuda-rust` commit.

## Status (Rust CUDA, Phase 1a: Whisper encoder)

Plan: `docs/superpowers/plans/2026-11-24-rust-cuda-phase1a-encoder.md`.

### Phase 0 review follow-ups

- Fixed: the arena waits for its stream before it frees memory (M1). `Blas` holds its context and checks it on each call (M2). Uploads and downloads bind the context, so other threads can use a `Gpu` (M3). Empty tensors return at once, and grid sizes over `u32` panic with a clear message (M4). `block_sum` ends with a barrier, so a kernel can call it twice (M6). `Blas::load` tries the DLL next to the exe first (M9).
- Still open: M5 (a real out-of-memory at allocation comes back as a driver error), M7 (`cargo.cmd` limits: CUDA folder order, `!` in PATH, very long PATH), M8 (`--workspace` commands include `jeuvoice-gpu`), M10 (`compare` ignores NaN in `max_rel`).

### The ggml model file (large-v3-turbo)

`jeuvoice_whisper::ggml::GgmlModel` reads it in plain Rust. Hparams: `n_mels 128`, `n_audio_ctx 1500`, `n_audio_state 1280`, `n_audio_head 20`, `n_audio_layer 32`, `n_text_layer 4`, `n_vocab 51866`. Mel filters `128 x 201`. 2-D weights are f16. Biases, norm weights and `encoder.positional_embedding` are f32.

| Encoder tensor | Type | Shape |
|---|---|---|
| `encoder.conv1.weight` / `.bias` | f16 / f32 | `[1280, 128, 3]` / `[1280, 1]` |
| `encoder.conv2.weight` / `.bias` | f16 / f32 | `[1280, 1280, 3]` / `[1280, 1]` |
| `encoder.positional_embedding` | f32 | `[1500, 1280]` |
| `encoder.blocks.N.attn_ln.weight` / `.bias` | f32 | `[1280]` |
| `encoder.blocks.N.attn.query.weight` / `.bias` | f16 / f32 | `[1280, 1280]` / `[1280]` |
| `encoder.blocks.N.attn.key.weight` (no bias) | f16 | `[1280, 1280]` |
| `encoder.blocks.N.attn.value.weight` / `.bias` | f16 / f32 | `[1280, 1280]` / `[1280]` |
| `encoder.blocks.N.attn.out.weight` / `.bias` | f16 / f32 | `[1280, 1280]` / `[1280]` |
| `encoder.blocks.N.mlp_ln.weight` / `.bias` | f32 | `[1280]` |
| `encoder.blocks.N.mlp.0.weight` / `.bias` | f16 / f32 | `[5120, 1280]` / `[5120]` |
| `encoder.blocks.N.mlp.2.weight` / `.bias` | f16 / f32 | `[1280, 5120]` / `[1280]` |
| `encoder.ln_post.weight` / `.bias` | f32 | `[1280]` |

The decoder blocks have the same names under `decoder.blocks.N`, plus `cross_attn.*` and `cross_attn_ln.*`.

### Kernels written so far (Phase 1a)

| Kernel | Lesson | Result on this card |
|---|---|---|
| `bias_gelu` (`[f16; 2]`), `bias_add`, `add_into`, `add_rows` | 1 | 178 us, 90% of the memory limit at `[1500, 5120]` |
| `layernorm` (register pair kernel) | 2 | 47.4 us, 84% at `[1500, 1280]`; 91% at 4x rows |
| `softmax_rows` (register kernel) | 3 | 993 us, 94% at `[30000, 1500]` |
| `gemm_naive/tiled/regtile` | 4 | 9.3 / 6.7 / 2.5 ms at `1500x1280x1280`; cuBLAS 0.29-0.36 ms |

### Inline PTX (`ptx_asm!`) does not build in the test profile

`cuda_device::ptx_asm!` works in a `cargo oxide build` (dev) or release build, and fails in `cargo oxide test`:

```text
Unsupported construct: ptx_asm template must be a byte string literal, got MIR operand Move(_3)
```

The macro passes the template through a temporary, and the test profile keeps that temporary in the MIR. The cuda-oxide MIR importer accepts only a constant operand, so the device codegen stops with `invalid input program`. Reproduced with a 15-line crate on the pinned rev `5e5c77b`.

Consequences:

- `gemm_tc_f16` (the tensor-core lesson kernel, `mma.sync.aligned.m16n8k8`) is deferred. The ready-made `cuda_device::wmma` mma wrappers need `sm_80` shapes (`m16n8k16`), and the PTX target is `sm_75`. The product path uses cuBLAS, so no product code depends on this.
- A direct call to the marker (`cuda_device::ptx::__ptx_asm_out_0::<u32, 21, 2, 0>(b"...", b"=r", b"")`) does build in the test profile. A small wrapper macro in `jeuvoice-gpu` is the next thing to try.
- Watch the cuda-oxide repository for a fix. Patching a fork is the last resort (AGENTS.md rule 1).

### The encoder (plan 1a, Task 10)

`jeuvoice_whisper::cuda::encoder::Encoder` runs the large-v3-turbo encoder on our kernels and cuBLAS. Checked on 2026-11-24 against PyTorch fp32, on the 3 test clips, with the reference mel as input:

| Tap | Max difference | Mean difference | Values out of tolerance |
|---|---|---|---|
| conv1, conv2, embed | 0.0020–0.0032 | 0.00002–0.00014 | 0 |
| block_00, block_15 | 0.0031–0.0113 | 0.00024–0.00035 | 0 |
| block_31 | 3.09–8.46 | 0.0019–0.0031 | 0.53–0.65% |
| ln_post | 1.33–2.24 | 0.0030–0.0069 | 0.75–2.06% |

The deep taps are as far from fp32 as **PyTorch itself in fp16** (0.56–0.75% at block_31, 0.75–2.04% at ln_post), with the bad values in the same rows. So the kernels are right, and the difference is fp16 activations through 32 blocks. whisper.cpp keeps activations in f32. If plan 1b misses the word error target, an f32 residual stream is the first thing to try.

| Number | Value |
|---|---|
| One 30 s window | 252 ms (about 140 ms of GEMM at cuBLAS speed, about 80 ms of attention) |
| Weights (f16) | 1214 MB |
| Activations | 165 MB |

Decisions in this task:

- The weights and the activations share **one arena**, because each op takes one arena for all its tensors. The decoder in plan 1b can join the same arena.
- The encoder activations are **time first** (`[1500, 1280]`), so each bias runs along the last dimension. `conv1d_with_layout` reads the mel channels first and writes time first. The GEMM computes `colᵀ · wᵀ`, so no transpose kernel runs.
- Attention uses `AttnKind::Naive` (cuBLAS). `Encoder::set_attention` can choose `Flash` for tests and lessons.

## Status (Rust CUDA, tensor cores)

Plan: `docs/superpowers/plans/2026-11-24-rust-cuda-tensor-cores.md`.

- **The `ptx_asm!` problem no longer blocks the tensor cores.** cuda-oxide generates `cuda_device::wmma::mma_m16n8k8_f32_f16` ("PTX 6.5 and sm_75+") and `ldmatrix_x1/x2/x4` with `_trans`. The kernels call them directly. The `ptx_asm!` test-profile bug is still there for other inline PTX.
- **Load width comes from the type alignment.** A `[u32; 4]` load compiles to four `ld.global.b32`. A `#[repr(C, align(16))]` type (`kernels::Q16`) compiles to one `ld.global.v4`.

| Kernel (1500 x 1280 x 1280) | Time | Rate |
|---|---|---|
| `gemm_regtile_f16` (SIMT) | 2533 µs | 1.9 TFLOP/s |
| `gemm_tc_f16` (v1, 2-byte tile copy) | 658–662 µs | 7.4 TFLOP/s |
| `gemm_tc_vec_f16` (v2, 16-byte tile copy) | 384–394 µs | 12.5–12.8 TFLOP/s |
| `gemm_tc_pre_f16` (v3, v2 + register prefetch) | 379–381 µs | 12.9–13.0 TFLOP/s |
| cuBLAS | 287–364 µs | 13.5–17.1 TFLOP/s |

The product keeps cuBLAS for GEMM.

### Tensor-core flash attention

`flash_attn_tc_f16` (`AttnKind::FlashTc`): `Q·Kᵀ` and `P·V` with `mma.sync.m16n8k8`, the online softmax on the accumulator registers, and P passed from the S accumulators straight into A fragments (FlashAttention-2).

| Attention (20 x 1500 x 64) | Time |
|---|---|
| `Naive` (cuBLAS + softmax) | 2.49 ms |
| `Flash` (scalar) | 26.6 ms |
| `FlashTc`, loops not unrolled | 6.8 ms (80 `ld.local`, 136 `st.local`) |
| **`FlashTc`, with `unroll!`** | **0.90–0.92 ms** (no local memory) |

- **Local memory finding:** an array indexed by a loop variable goes to local memory when the compiler does not unroll the loop. The `unroll!` macro in `kernels/mod.rs` writes the body out with constant indexes.
- The encoder now uses `FlashTc`: one 30 s window takes **176.4 ms** (it was 252 ms). The reference test passes on all 3 clips. Values out of tolerance went down on 2 clips (ses_001_0007 ln_post 2.37% to 1.80%, my_voice2 ln_post 0.75% to 0.50%) and up a little on my_voice (ln_post 1.08% to 1.14%). The f32 scores in registers are the likely cause of the gains, but not the only factor.

### The score tensor only for Naive attention

`ops::attention` takes the `[heads, seq, seq]` scratch tensor as `Option<&Tensor>`. Only `AttnKind::Naive` needs it. `Encoder::load` (FlashTc) does not plan it, and `Encoder::load_with_attention(.., AttnKind::Naive)` does. Measured on 2026-11-25: the encoder activations went from **165 MB to 79 MB**, with the same speed (175.0 ms per 30 s window).

## Status (Rust CUDA, Phase 1b)

Plan: `docs/superpowers/plans/2026-11-25-rust-cuda-phase1b-decoder.md`. Branch: `rust-cuda-phase1b`.

### The whisper.cpp baseline (Task 0, 2026-11-25)

The example `dump_whisper_rs` runs `WhisperRs` with the app's parameters and writes `target/reference/whisper/<clip>.rs.json`. Two runs give byte-equal files, so whisper-rs is deterministic on these clips.

| Clip | Length | Windows | Segments | Tokens | Warm time (median of runs 2 to 4) | VRAM above idle (peak) |
|---|---|---|---|---|---|---|
| my_voice.ogg | 27.0 s | 1 | 3 | 138 | 1.127 s | 2162 MiB |
| my_voice2.mp4 | 33.6 s | 2 | 6 | 189 | 1.874 s | 2160 MiB |
| ses_001_0007.wav | 8.4 s | 1 | 3 | 52 | 0.741 s | 2134 MiB |
| **69 s set** | | 4 | 12 | 379 | **3.74 s** | **2162 MiB** |

- The time does not include the model load. Run 1 is slower (CUDA start-up), so the table uses the median of runs 2 to 4. The spec target of criterion 3 stays at 4.3 s. Report both numbers.
- The VRAM is the whole card (`nvidia-smi --query-gpu=memory.used` every 100 ms) minus the idle value before the run. Per-process numbers show `[N/A]` on this Windows driver (WDDM).
- The fallback count is not in the whisper-rs API (`n_fail_p` is on the state, and `print_timings` prints the context's own state). The mean token logprob of each segment is between −0.19 and −0.006, far above `logprob_thold = −1.0`. So the 4 windows most likely decode at temperature 0.

### The no-speech prob in whisper.cpp

whisper.cpp computes the no-speech prob from row 0 of `state->logits` (line 7154). The prompt decode fills only its last row. So row 0 holds the raw logits of the last 1-token decode, or zeros in a new state. Measured: the first window of each clip gives 1.93e-5 (a zero row: 1/51866 with f32 sum error). The second window of my_voice2 gives 1.5e-10 (the last step of window 1). The port copies this behavior, because criterion 1 compares with whisper.cpp.

### The token embedding (Task 7, 2026-11-25)

`ops::embed` (many rows, the ids uploaded after a host check) and `ops::embed_one` (one row, the id as a launch argument). `Tensor::rows(start, n)` gives a view of rows. It panics when the row does not start on the 256-byte arena alignment, so the alignment rules of the other kernels still hold. Medians of 3 runs of `gpu-bench embed` and `ids_upload` (table 51866 x 1280):

| Row | 1 row | 224 rows |
|---|---|---|
| `embed_x1` (with the ids upload) | 19.7 µs | 25.1 µs |
| `embed_x2` (with the ids upload) | 17.6 µs | 26.2 µs |
| `embed_one` (id as an argument) | 9.0 µs | — |
| `ids_upload` alone (448 `i32`) | 10.0 µs | — |

The upload and its wait cost more than the kernel. A decoder step uses `embed_one`. Lesson 8, part A, has the details.

### The decoder attention (Task 8, 2026-11-25)

`ops::attention_decode` covers the self-attention with the KV cache (causal or one step), the cross-attention on the 1500 encoder positions, and the softmax of chosen heads for DTW (`ProbsOut`). It reads the `[rows, 1280]` layout of the GEMM outputs, so the decoder needs no head split. Medians of 3 runs of `gpu-bench attn_dec` (20 heads):

| Case | Naive | Split (flash decoding) | Gain |
|---|---|---|---|
| 1 x 448 (self step) | 56.2 µs | 22.9 µs (52% of the limit) | 2.5x |
| 1 x 1500 (cross step) | 175.5 µs | 59.1 µs (68% of the limit) | 3.0x |
| 224 x 224 causal (prompt) | 1024 µs | 482 µs | 2.1x |
| 224 x 1500 with probs (DTW) | 12.6 ms | 3.47 ms | 3.6x |

The product uses `Split`. It needs a scratch tensor of `decode_split_scratch(heads, nq, nk)` f32. The multi-row cases are limited by the scalar math on shared memory. A tensor-core version is lesson 8 exercise 4.

### The logits GEMM (Task 9, 2026-11-25)

`Blas::gemm_f16_f32out`: f16 A and B, f32 C and f32 math, for the decoder logits `x · W_tokᵀ` (`W_tok` is `[51866, 1280]`). Measured (3 runs, all the same):

| Size | Time | Rate | % of the 192 GB/s limit |
|---|---|---|---|
| 1 x 51866 x 1280 | 718 µs | 185 GB/s | 96% |
| 7 x 51866 x 1280 | 736 µs | 182 GB/s | 95% |

One decoder step reads the whole 133 MB table, so this is the largest single cost of a step. cuBLAS is at the limit, so there is no kernel work to do here.
