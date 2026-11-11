@echo off
rem Makes the Windows installer: the release bundle, then the NSIS setup exe.
rem Run it from anywhere; the script goes to the repo root first.
rem Sign dist\windows\JeuVoice\jeuvoice-app.exe before the packager line when
rem the app itself must be signed too (see packaging\windows\README.md).
setlocal
cd /d "%~dp0.."

rem The bundle: app, CUDA DLLs, Visual C++ runtime, Python, uv, engine, assets.
powershell -ExecutionPolicy Bypass -File packaging\windows\collect.ps1
if errorlevel 1 exit /b 1

rem The packager reads [package.metadata.packager] in crates\jeuvoice-app.
cargo packager -p jeuvoice-app --release
if errorlevel 1 exit /b 1

echo.
echo Installer: dist\windows\jeuvoice-app_*_x64-setup.exe
