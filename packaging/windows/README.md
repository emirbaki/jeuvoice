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

The script builds `target\release\jeuvoice-app.exe` with CUDA archs 75, 86 and
89. Use `-SkipBuild` to keep an exe that is already built. Use `-CudaPath` when
the toolkit is not in the default folder.

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

## Size

The bundle is about 1 GB. Most of it is `cublasLt64_13.dll` (about 500 MB). The
engine environment in the app data folder is about 5 GB after setup, and models
are extra.
