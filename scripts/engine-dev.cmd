@echo off
rem Make engine/.venv with uv and the pinned packages (Plan 2 Global Constraints).
setlocal
cd /d "%~dp0.."

if not exist engine\.venv (
    uv venv engine\.venv --python 3.12
    if errorlevel 1 exit /b 1
)

rem Windows: torch from the CUDA 12.6 index, the rest from the default index.
uv pip install --python engine\.venv\Scripts\python.exe ^
    --index-url https://download.pytorch.org/whl/cu126 ^
    torch==2.14.0
if errorlevel 1 exit /b 1

uv pip install --python engine\.venv\Scripts\python.exe ^
    transformers==5.19.0 peft==0.21.2 accelerate==1.15.0 ^
    "omnivoice[lora] @ git+https://github.com/k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" ^
    "trnorm @ git+https://github.com/ozguradmin/ahenk-turkish-tts" ^
    soundfile numpy huggingface_hub pytest
if errorlevel 1 exit /b 1

echo Done. Run: engine\.venv\Scripts\python -m pytest engine/tests -q
