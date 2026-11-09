# JeuVoice Plan 4 — First-Start Setup and Installers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A teammate who does not code installs JeuVoice from one file on Windows or macOS. On first start the app makes its own Python engine environment with a progress screen. No terminal, no config files.

**Architecture:** The installer carries the app, a portable Python, `uv`, the engine source, the default preset and (on Windows) the CUDA runtime DLLs. On first start the app runs `uv` to make the engine environment in the app data folder from a locked requirements file. Model files download later, when a tab first needs them (as in Plans 1–2).

**Tech Stack:** `cargo-packager` (NSIS `.exe` on Windows, `.app` + `.dmg` on macOS), python-build-standalone (CPython 3.12), `uv` (pinned version), `uv pip compile` lock files.

**Spec:** `docs/design.md` — sections 2 (anti-goals), 4.4, 4.5.

## Before you start

- Plans 2 and 3 must be merged into `main` (the user decides). Branch `plan4-setup` from `main`.

## Global Constraints

- All Plan 2 Global Constraints still apply.
- The app never touches `D:\MLProjelerim\.venv_ml` or tts_lab at run time. In development the engine runs from `engine/.venv`. In an installed app it runs from `<app data>/engine-env`.
- App data folder: `directories::ProjectDirs::from("com", "Jeu", "JeuVoice").data_dir()`. User files stay in `Documents/JeuVoice` and are never deleted by setup, update or uninstall.
- Pins: Python 3.12 (python-build-standalone, `install_only` build for `x86_64-pc-windows-msvc` and `aarch64-apple-darwin`), `uv` 0.12.5 (the version on the dev machine), engine packages from the Plan 2 Global Constraints, locked with hashes.
- Disk needed for the engine: about 5 GB. Setup checks free space first and needs 8 GB free.
- CUDA kernels: `CMAKE_CUDA_ARCHITECTURES=75;86;89` for release builds (Turing, Ampere, Ada).
- Code signing and Apple notarization need the user's own certificates and accounts. The agent prepares the config and the commands, and the user runs the signing steps.
- Nothing is uploaded or published by the agent. The user decides where the installers go.

## Review Focus

1. **Setup interrupted** (power loss, app closed, network drop): the next start sees a half-made environment, removes it, and starts again with a clear message. Test in Task 3.
2. **No NVIDIA GPU or an old driver on Windows:** the app starts, the Transcribe tab works on the CPU, and Settings shows "No supported graphics card found. Speech will be slow." The voice engine refuses to start with a plain message. Test in Task 3 and Task 6.
3. **Paths with Turkish letters or spaces** in the Windows user name (`C:\Users\Çağrı Öz\AppData\...`): setup, `uv` and the engine all work. Test in Task 3 (temp folder with such a name).
4. **Update over an old version:** the engine environment is made again only when the lock file hash changes. Presets and projects stay. Test in Task 3.
5. **Antivirus or a locked file** during setup on Windows: the error names the file and says "Close other programs and click Try again." Test in Task 3.

---

### Task 1: Lock files for the engine environment

**Files:** Create `engine/requirements.in`, `engine/requirements-win-cu126.txt`, `engine/requirements-macos.txt`, `scripts/lock-engine.cmd`. Modify `scripts/engine-dev.cmd` (install from the lock file).

- [ ] **Step 1:** `requirements.in` = the Plan 2 pins (without torch index details). `lock-engine.cmd` runs `uv pip compile` twice: Windows with the CUDA 12.6 torch index and `--python-platform x86_64-pc-windows-msvc`, macOS with `--python-platform aarch64-apple-darwin`, both `--python-version 3.12 --generate-hashes`.
- [ ] **Step 2:** Make `engine/.venv` again from the Windows lock file and run the engine tests (`pytest`, fake and gpu). Expected: PASS.
- [ ] **Step 3:** Commit: `build(engine): locked requirements per platform`.

### Task 2: Runtime layout and engine paths

**Files:** Create `crates/jeuvoice-app/src/runtime_paths.rs` (+ tests). Modify `services/engine.rs`.

**Interfaces:**
- Produces: `struct RuntimePaths { resources: PathBuf /* next to the exe, or the .app Resources */, data: PathBuf, engine_env: PathBuf, python: PathBuf /* in engine_env */, uv: PathBuf, engine_src: PathBuf, lock_file: PathBuf, assets: PathBuf }`, `fn detect() -> RuntimePaths` (dev mode when `engine/.venv` exists next to the workspace root and the exe is under `target/`. Installed mode otherwise), `fn lock_hash(&self) -> String` (SHA-256 of the lock file).
- [ ] **Step 1: Write failing tests:** dev and installed layouts from fake folder trees. Windows and macOS python paths inside the env. Lock hash changes when the file changes.
- [ ] **Step 2–4:** Implement, run, PASS. `EngineService` uses `RuntimePaths` instead of hard-coded dev paths.
- [ ] **Step 5:** Commit: `feat(app): runtime paths for dev and installed modes`.

### Task 3: First-start setup screen

**Files:** Create `crates/jeuvoice-app/src/setup/{mod.rs,state.rs,view.rs,steps.rs,tests.rs}`.

**Interfaces:**
- Produces: `enum SetupState { NotNeeded, Checking, Running { step: SetupStep, progress: f32, detail: String }, Failed { message: String }, Done }`, `enum SetupStep { CheckDisk, CheckGpu, UnpackPython, MakeEnvironment, InstallPackages, Verify }`, `fn needs_setup(paths: &RuntimePaths) -> bool` (env missing, `env/.jeuvoice-ok` missing, or its stored lock hash differs).
- Steps (all on a worker):
  1. Free disk ≥ 8 GB, else "JeuVoice needs 8 GB of free space on <drive>. Free some space and click Try again."
  2. GPU: Windows — `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` (missing → CPU mode message, Review Focus 2). MacOS — Apple Silicon check.
  3. Unpack the bundled Python archive into `<data>/python`.
  4. `uv venv <engine_env> --python <data>/python/...` .
  5. `uv pip install --require-hashes -r <lock_file>` with `UV_CACHE_DIR=<data>/uv-cache`. Parse `uv` output lines for a progress fraction and the package name.
  6. Verify: start the engine with `--backend omnivoice` only far enough to import torch and report the device (`/health`), then stop it. Write `<engine_env>/.jeuvoice-ok` with the lock hash.
- A failed or interrupted setup deletes the half-made `engine_env` on the next try (Review Focus 1). Errors map to plain texts (Review Focus 5).
- The screen shows the step list with ticks, a progress bar, the current package, "About 10–15 minutes. You can keep using the Transcribe tab." The Transcribe tab stays usable during setup. The Voice and Train tabs show "Setting up the voice engine…" until it is done.
- [ ] **Step 1: Write failing headless tests** with a fake command runner (trait over process calls): happy path ends in Done and writes the ok file. Low disk gives the message. A failure in step 5 then a new start removes the env first. A changed lock hash makes `needs_setup` true. A temp root with "Çağrı Öz" in the path works (Review Focus 3).
- [ ] **Step 2–4:** Implement, run, PASS. Snap the setup screen and look.
- [ ] **Step 5:** Commit: `feat(app): first-start engine setup`.

### Task 4: Windows release bundle with CUDA runtime

**Files:** Create `packaging/windows/collect.ps1`, `Packager.toml` (or `[package.metadata.packager]` in `crates/jeuvoice-app/Cargo.toml`). Modify `scripts/cargo.cmd` (release default archs `75;86;89`).

- [ ] **Step 1:** Build release with the new archs. Find which CUDA DLLs `jeuvoice-app.exe` needs (`dumpbin /dependents`, recursive for `ggml-cuda` DLLs if any). Copy only those from the CUDA 13.4 toolkit `bin` folder into the bundle folder. Check the CUDA EULA list of redistributable files and write the list and the check in `packaging/windows/README.md`.
- [ ] **Step 2:** `collect.ps1` makes `dist/windows/JeuVoice/` with: the exe, the CUDA DLLs, `resources/python-3.12-*.tar.gz` (download the pinned python-build-standalone file and check its SHA-256), `resources/uv.exe` (pinned, SHA-256 checked), `resources/engine/` (engine source without `.venv`, tests), `resources/requirements-win-cu126.txt`, `resources/assets/` (presets, recipes, kaggle template).
- [ ] **Step 3:** Start `dist/windows/JeuVoice/jeuvoice-app.exe` from a folder with a space and Turkish letters in its name. Check it finds its resources (installed mode) and shows the setup screen.
- [ ] **Step 4:** Commit: `build(windows): release bundle with CUDA runtime and resources`.

### Task 5: Windows installer

**Files:** Packager config, `packaging/windows/installer.nsi` hooks if needed, `scripts/make-installer.cmd`.

- [ ] **Step 1:** `cargo-packager` NSIS target: per-user install (no admin) into `%LOCALAPPDATA%\Programs\JeuVoice`, Start menu shortcut, an uninstaller that removes the program folder and asks "Also remove the voice engine and downloaded models (about 8 GB)?" (default No). Never touch `Documents\JeuVoice`.
- [ ] **Step 2:** Make the installer. Install it under a fresh Windows user account (or a VM if the user has one): first start → setup → Transcribe a file → Voice tab generates with the local adapter or the HF token. Write each result in `packaging/windows/README.md`. This needs the user's help for the second account: ask.
- [ ] **Step 3:** Signing: write the `signtool` command for the user's certificate in the README. The user runs it.
- [ ] **Step 4:** Commit: `build(windows): NSIS installer`.

### Task 6: macOS bundle and disk image

**Files:** Packager macOS config, `packaging/macos/Info.plist` additions, `packaging/macos/README.md`, `scripts/make-dmg.sh`.

The agent has no Mac. Prepare everything so a teammate with an Apple Silicon Mac runs one script.

- [ ] **Step 1:** Packager `.app` + `.dmg` targets: resources as on Windows, but `requirements-macos.txt`, the macOS Python archive, the macOS `uv`, and no CUDA files. Minimum macOS 13. Whisper builds with `metal`.
- [ ] **Step 2:** `make-dmg.sh` builds release with `cargo build --release -p jeuvoice-app`, collects resources, makes the dmg. Signing and notarization commands (`codesign --options runtime`, `xcrun notarytool submit`, `xcrun stapler staple`) are in the README with placeholders for the user's team id and profile.
- [ ] **Step 3:** If the Plan 2 Mac spike failed, the app shows the Voice tab message from Plan 2 and setup still installs the engine for the Train tab's data prep. Make sure the macOS build does not break when CUDA code is absent (`cfg` checks).
- [ ] **Step 4:** Commit: `build(macos): app bundle and dmg script`. Tell the user what to run on the Mac and what to send back.

### Task 7: Update path and version checks

**Files:** Modify `setup/state.rs`, `services/engine.rs`. Create `crates/jeuvoice-app/src/version.rs`.

- [ ] **Step 1: Write failing tests:** an engine that reports another `version` than the app expects gives "The voice engine is from another JeuVoice version. Click Repair." and **Repair** runs setup again. An unchanged lock hash after an update skips setup (Review Focus 4). Presets and projects survive a simulated update (copy new resources over old ones in a temp tree).
- [ ] **Step 2–4:** Implement, run, PASS.
- [ ] **Step 5:** Commit: `feat(app): safe updates and engine repair`.

### Task 8: Final check and user guide

- [ ] **Step 1:** Full tests: `scripts\cargo.cmd test --workspace`, engine `pytest` (fake + gpu).
- [ ] **Step 2:** Write `docs/user-guide.md` in plain short sentences for teammates: install, first start, Transcribe, Voice (presets, generate, retake, fix word, export), Train (Kaggle key, data, start, test checkpoints), and "When something goes wrong" (each plain error text from the app and what to do).
- [ ] **Step 3:** Update README status: what is done for the MVP, what the user must still do (signing, Mac run, uploading the tr2 adapter to HF, by-ear checks).
- [ ] **Step 4:** Commit: `docs: user guide and MVP status`.
