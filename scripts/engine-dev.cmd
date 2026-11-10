@echo off
rem Make engine/.venv with uv and install the locked engine environment.
setlocal
cd /d "%~dp0.."

if not exist engine\.venv (
    uv venv engine\.venv --python 3.12
    if errorlevel 1 exit /b 1
)

uv pip install --python engine\.venv\Scripts\python.exe --require-hashes -r engine\requirements-win-cu126.txt
if errorlevel 1 exit /b 1

echo Done. Run: engine\.venv\Scripts\python -m pytest engine/tests -q
