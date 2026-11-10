@echo off
rem Lock the engine environment per platform (Plan 4 Task 1). Needs network:
rem git refs are cloned to resolve their hashes.
setlocal
cd /d "%~dp0.."

rem Windows: torch from the CUDA 12.6 index, the rest from PyPI.
uv pip compile engine/requirements.in ^
    --python-version 3.12 ^
    --python-platform x86_64-pc-windows-msvc ^
    --index-url https://download.pytorch.org/whl/cu126 ^
    --extra-index-url https://pypi.org/simple ^
    --generate-hashes ^
    -o engine/requirements-win-cu126.txt
if errorlevel 1 exit /b 1

rem macOS: everything from PyPI (CPU torch).
uv pip compile engine/requirements.in ^
    --python-version 3.12 ^
    --python-platform aarch64-apple-darwin ^
    --generate-hashes ^
    -o engine/requirements-macos.txt
if errorlevel 1 exit /b 1

echo Done. Review the diff, then remake the venv: scripts\engine-dev.cmd
