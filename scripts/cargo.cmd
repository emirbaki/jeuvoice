@echo off
rem Runs cargo with the Visual Studio, CMake and bindgen settings that whisper-rs-sys
rem needs on Windows. Use it in place of "cargo": scripts\cargo.cmd test -p jeuvoice-whisper
setlocal enabledelayedexpansion

set "VSINSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
if not exist "%VSINSTALLER%\vswhere.exe" (
    echo Visual Studio is not installed: vswhere.exe is missing. Install Visual Studio 2022 with the C++ tools.
    exit /b 1
)

rem Find Visual Studio 2022 first, because the CMake generator below is "Visual Studio 17 2022".
rem If it is missing, use the newest Visual Studio with the C++ tools.
set "VSDIR="
pushd "%VSINSTALLER%"
for /f "usebackq delims=" %%i in (`.\vswhere.exe -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    for /f "usebackq delims=" %%i in (`.\vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
)
popd
if not defined VSDIR (
    echo Visual Studio with the C++ tools was not found. Install the "Desktop development with C++" workload.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
if errorlevel 1 exit /b 1

rem bindgen must use the MSVC target and the MSVC include folders, not the mingw clang defaults.
set "ARGS=--target=x86_64-pc-windows-msvc"
for %%p in ("%INCLUDE:;=";"%") do (
    if not "%%~p"=="" set ARGS=!ARGS! -isystem "%%~p"
)
set "BINDGEN_EXTRA_CLANG_ARGS=!ARGS!"

set "CMAKE_GENERATOR=Visual Studio 17 2022"
rem Turing (75), Ampere (86) and Ada (89) cards: the release bundle runs on all three.
if not defined CMAKE_CUDA_ARCHITECTURES set "CMAKE_CUDA_ARCHITECTURES=75;86;89"

cd /d "%~dp0.."
cargo %*
exit /b %errorlevel%
