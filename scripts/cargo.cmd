@echo off
rem Runs cargo with the Visual Studio, CMake and bindgen settings that whisper-rs-sys
rem needs on Windows. Use it in place of "cargo": scripts\cargo.cmd test -p jeuvoice-whisper
rem GPU kernel crates: scripts\cargo.cmd oxide test -- -p jeuvoice-gpu
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

rem "scripts\cargo.cmd oxide ..." builds with the cuda-oxide GPU kernel compiler (crates\jeuvoice-gpu).
if /i "%~1"=="oxide" (
    call :oxide_env
    if errorlevel 1 exit /b 1
)

cd /d "%~dp0.."
cargo %*
exit /b %errorlevel%

:oxide_env
rem libclang (bindgen in cuda-bindings) reads the MinGW headers when MinGW is on PATH. Remove it.
set "NEWPATH="
for %%p in ("%PATH:;=";"%") do (
    set "ENTRY=%%~p"
    if not "!ENTRY!"=="" if "!ENTRY:mingw=!"=="!ENTRY!" set "NEWPATH=!NEWPATH!;!ENTRY!"
)
set "PATH=!NEWPATH:~1!"

if not defined CUDA_PATH (
    for /d %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v13.*") do set "CUDA_PATH=%%d"
)
if not defined CUDA_PATH (
    echo The CUDA 13 toolkit was not found. Install CUDA 13 from https://developer.nvidia.com/cuda-downloads.
    exit /b 1
)
set "CUDA_TOOLKIT_PATH=%CUDA_PATH%"
if not defined LIBCLANG_PATH set "LIBCLANG_PATH=%ProgramFiles%\LLVM\bin"
if not exist "%LIBCLANG_PATH%\libclang.dll" (
    echo libclang.dll was not found. Install LLVM: winget install --id LLVM.LLVM -e
    exit /b 1
)
rem vcvars64.bat sets VCPKG_ROOT to the Visual Studio copy of vcpkg, which usually has no libffi.
rem JEUVOICE_VCPKG wins, then VCPKG_ROOT, then the usual install folders.
set "FFIROOT="
for %%r in ("%JEUVOICE_VCPKG%" "%VCPKG_ROOT%" "C:\vcpkg" "D:\vcpkg" "%USERPROFILE%\vcpkg") do (
    if not defined FFIROOT if not "%%~r"=="" if exist "%%~r\installed\x64-windows\lib\ffi.lib" set "FFIROOT=%%~r"
)
if not defined FFIROOT (
    echo libffi was not found. Install it with vcpkg: vcpkg install libffi:x64-windows. Then set JEUVOICE_VCPKG to the vcpkg folder.
    exit /b 1
)
set "VCPKG_ROOT=%FFIROOT%"
where cargo-oxide.exe >nul 2>nul
if errorlevel 1 (
    echo cargo-oxide was not found. Install it: cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
    exit /b 1
)
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%LIBCLANG_PATH%;%VCPKG_ROOT%\installed\x64-windows\bin;%PATH%"
exit /b 0
