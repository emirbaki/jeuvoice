# Windows release bundle

`collect.ps1` makes `dist\windows\JeuVoice`. The folder holds the app, the CUDA
runtime, the Python archive and the engine source. On first start the app
unpacks Python, makes the engine environment and installs the locked packages.
This takes about 10 to 15 minutes.

## Make the bundle

1. Install Visual Studio with the "Desktop development with C++" workload.
2. Install the CUDA 13.4 toolkit.
3. Run `powershell -ExecutionPolicy Bypass -File packaging\windows\collect.ps1`.
4. Wait for the message `Bundle ready`.

The script builds `target\release\jeuvoice-app.exe` with the cuda-oxide backend (the Rust
GPU kernels) and the `cuda-kernels` feature. The PTX of the kernels is inside the exe, so no
file goes next to it. CUDA archs 75, 86 and 89. Use `-SkipBuild` to keep an exe that is
already built. Use `-CudaPath` when the toolkit is not in the default folder.

The exe transcribes with whisper.cpp by default. Set `JEUVOICE_CUDA_WHISPER=1` to use the
Rust kernels on the GPU. When the GPU model does not load, the app runs whisper.cpp and the
transcription still works.

## What the bundle contains

| Item | Purpose |
|---|---|
| `jeuvoice-app.exe` | The app. |
| `uv.exe` | Makes the engine environment and installs the lock. |
| `python-3.12.16-20261110.tar.gz` | The Python that setup unpacks into the app data folder. |
| `cublas64_13.dll`, `cublasLt64_13.dll` | The CUDA libraries that Whisper loads. |
| `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll` | The Visual C++ runtime. |
| `requirements-win-cu126.txt` | The locked engine packages. |
| `engine\` | The engine source and `overrides.txt`. No venv and no tests. |
| `assets\` | Presets, recipes and the Kaggle template. |

The app finds these files next to the exe. Do not move them into a `resources`
subfolder. `runtime_paths.rs` reads the layout from the exe folder.

## CUDA DLLs

`dumpbin /dependents target\release\jeuvoice-app.exe` shows `cublas64_13.dll`.
That DLL imports `cublasLt64_13.dll`. No other CUDA DLL is in the import table.
`collect.ps1` copies these two files from `bin\x64` of the CUDA 13.4 toolkit.
CUDA 13 keeps the x64 DLLs in `bin\x64`. Older toolkits use `bin`.

The CUDA EULA, Attachment A ("Redistributable Software"), lists `cublas.dll`
and `cublasLt.dll` for the CUDA Toolkit. The versioned files in this bundle are
the same libraries. Read the list before you ship a new CUDA version:
<https://docs.nvidia.com/cuda/eula/index.html>

## Pinned downloads

| File | Version | SHA-256 |
|---|---|---|
| `python-3.12.16-20261110.tar.gz` | python-build-standalone `20261110` | `e23fe5b16926de1d81661b1a97352e89fb167549c55a6825796ccd0f32ec7f1b` |
| `uv-x86_64-pc-windows-msvc.zip` (then `uv.exe`) | uv `0.13.7` | `0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021` |

`collect.ps1` downloads each file one time and stops on a wrong hash. The uv
hash is for the downloaded zip; the script checks it before it extracts
`uv.exe`. A new version needs a new tag, file name and hash in the script and
in this table.

## Test the bundle

1. Copy `dist\windows\JeuVoice` to a folder with a space and Turkish letters,
   for example `C:\Users\Çağrı Öz\Desktop\JeuVoice Test`.
2. Start `jeuvoice-app.exe`.
3. Make sure the Voice tab shows "Setting up the voice engine" and the step
   list.
4. Open the Transcribe tab while setup runs. It must work.
5. When the screen shows "Ready", make a short render on the Voice tab.

Setup writes only to the app data folder (`%APPDATA%\Jeu\JeuVoice\data` by
default). It never deletes presets or projects in `Documents\JeuVoice`.

## App icon

`make-icon.py` draws the icon as SVG: a black tile, white "JEU" with an orange
3D layer stack, and an orange soundwave. It writes `icon.svg` (the source),
`icon.png` (1024 px) and `icon.ico` (16 to 256 px) in this folder.

```cmd
engine\.venv\Scripts\pip install resvg-py
engine\.venv\Scripts\python packaging\windows\make-icon.py
```

The letter paths, layer colors and wave heights are the constants at the top
of the script. The script uses `resvg-py` to render the SVG and Pillow to write
the ICO.

`crates\jeuvoice-app\build.rs` embeds `icon.ico` in the exe, so Explorer, the
taskbar and the window use it. `cargo-packager` reads the same file from
`[package.metadata.packager] icons` and `nsis.installerIcon` for the installer.

## Make the installer

1. Run `scripts\make-installer.cmd` from any folder.
2. The script makes the bundle, then runs `cargo packager -p jeuvoice-app
   --release` with `[package.metadata.packager]` from
   `crates\jeuvoice-app\Cargo.toml`.
3. Find `dist\windows\jeuvoice-app_<version>_x64-setup.exe`.

The installer needs `cargo-packager` 0.11.8 or newer
(`cargo install cargo-packager --locked`). It also needs NSIS; the packager
downloads its own copy when NSIS is not installed.

The installer is per user and needs no admin rights. It installs to
`%LOCALAPPDATA%\Programs\JeuVoice` and makes a Start menu shortcut. The
template `packaging\windows\installer.nsi` is the stock cargo-packager 0.11.8
template with two changes: the install folder above, and the uninstall
checkbox text.

The uninstaller removes the program folder and shows one checkbox:
"Also remove the voice engine and downloaded models (about 8 GB)?". It is
unchecked by default, so the engine, models and app settings stay. It deletes
`%APPDATA%\Jeu\JeuVoice` only when the user ticks the box. It never touches
`Documents\JeuVoice`.

## Sign the installer

The user runs these commands with their own certificate. Sign the app before
packaging and the setup exe after. Use the real installer name from
`dist\windows`.

```cmd
signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a ^
  dist\windows\JeuVoice\jeuvoice-app.exe

scripts\make-installer.cmd

signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /a ^
  dist\windows\jeuvoice-app_0.0.1_x64-setup.exe
```

`signtool` is in the Windows SDK. `/a` picks the best certificate in the
store; replace it with `/sha1 <thumbprint>` to select one. Check the result
with `signtool verify /pa /v <file>`.

## Installer test record

Run these on a fresh Windows user account. Write the result of each step here.

| Step | Date | Result |
|---|---|---|
| Install with no admin prompt; Start menu shortcut works. | | |
| First start shows the setup screen; setup reaches "Ready". | | |
| Transcribe one file. | | |
| Voice tab makes a render with the local adapter or the HF token. | | |
| Uninstall keeps `%APPDATA%\Jeu\JeuVoice` with the box unticked. | | |
| Uninstall removes it with the box ticked, and keeps `Documents\JeuVoice`. | | |

### Smoke test on the dev account (2026-11-11)

This test does not replace the fresh-account test above.

| Step | Result |
|---|---|
| Silent install (`/S /D=%TEMP%\JV test Çağrı`), no admin prompt | PASS. Exit 0, 31 files, about 780 MB. |
| Start menu shortcut `JeuVoice\JeuVoice.lnk` | PASS. Made. |
| Uninstall registry key under HKCU | PASS. Made by install, removed by uninstall. |
| Silent uninstall (box unticked) | PASS. Program files and shortcut removed. `%APPDATA%\Jeu\JeuVoice` stays. |

## Size

The bundle is about 1 GB. Most of it is `cublasLt64_13.dll` (about 500 MB). The
engine environment in the app data folder is about 5 GB after setup, and models
are extra.
