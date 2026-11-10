@echo off
rem Lock the engine environment per platform (Plan 4 Task 1). Needs network:
rem git refs are cloned to resolve their hashes.
setlocal
cd /d "%~dp0.."

rem Windows lock. The torch wheels come by URL from requirements.in; the rest
rem comes from PyPI, so no index flags are needed.
uv pip compile engine/requirements.in ^
    --override engine/overrides.txt ^
    --python-version 3.12 ^
    --python-platform x86_64-pc-windows-msvc ^
    --generate-hashes ^
    -o engine/requirements-win-cu126.txt
if errorlevel 1 exit /b 1

rem macOS: everything from PyPI (CPU torch).
uv pip compile engine/requirements.in ^
    --override engine/overrides.txt ^
    --python-version 3.12 ^
    --python-platform aarch64-apple-darwin ^
    --generate-hashes ^
    -o engine/requirements-macos.txt
if errorlevel 1 exit /b 1

echo Done. Review the diff, then remake the venv: scripts\engine-dev.cmd
