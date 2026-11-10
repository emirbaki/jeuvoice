@echo off
rem Make engine/.venv with uv and install the locked engine environment.
rem The lock file pins versions, hashes, git commits and wheel URLs.
rem --override is needed here too: installing re-resolves, and OmniVoice's own
rem pins would fight the lock without it. No --require-hashes: uv cannot hash
rem git URLs.
setlocal
cd /d "%~dp0.."

if not exist engine\.venv (
    uv venv engine\.venv --python 3.12
    if errorlevel 1 exit /b 1
)

uv pip install --python engine\.venv\Scripts\python.exe ^
    --override engine\overrides.txt ^
    -r engine\requirements-win-cu126.txt
if errorlevel 1 exit /b 1

echo Done. Run: engine\.venv\Scripts\python -m pytest engine/tests -q
