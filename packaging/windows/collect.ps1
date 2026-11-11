# Makes dist\windows\JeuVoice from the release exe and the pinned resources.
#
#   powershell -ExecutionPolicy Bypass -File packaging\windows\collect.ps1
#
# The Python archive and uv are downloaded once and checked against the pinned
# SHA-256. The script never touches Documents\JeuVoice or the app data folder.

[CmdletBinding()]
param(
    # Where the bundle goes. Keep the folder name; the app looks next to itself.
    [string]$OutDir,
    # Skip `cargo build --release` when the exe is already fresh.
    [switch]$SkipBuild,
    # The CUDA 13.4 toolkit (the release exe links cublas64_13.dll).
    [string]$CudaPath = $env:CUDA_PATH,
    # The repo root.
    [string]$Repo
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# $PSScriptRoot is empty in param defaults on Windows PowerShell 5.1.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutDir) { $OutDir = Join-Path $here '..\..\dist\windows\JeuVoice' }
if (-not $Repo) { $Repo = (Resolve-Path (Join-Path $here '..\..')).Path }

# Pinned downloads. A new version means a new tag, file, and SHA-256.
$PyTag = '20261110'
$PyAsset = 'cpython-3.12.16+20261110-x86_64-pc-windows-msvc-install_only.tar.gz'
$PySha = 'e23fe5b16926de1d81661b1a97352e89fb167549c55a6825796ccd0f32ec7f1b'
$PyName = 'python-3.12.16-20261110.tar.gz'
$PyUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/$PyTag/$([uri]::EscapeDataString($PyAsset))"

$UvVersion = '0.13.7'
$UvAsset = 'uv-x86_64-pc-windows-msvc.zip'
$UvSha = '0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021'
$UvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$UvAsset"

# The CUDA DLLs the exe loads (dumpbin /dependents: cublas64_13.dll, which
# imports cublasLt64_13.dll). Both are on the CUDA EULA redistributable list.
$CudaDlls = @('cublas64_13.dll', 'cublasLt64_13.dll')
# The Visual C++ runtime the exe imports; copied from the VS redist folder.
$CrtDlls = @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')

function Assert-File([string]$Path, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$What is missing: $Path"
    }
}

function Get-PinnedFile([string]$Url, [string]$Sha, [string]$Dest) {
    if (Test-Path -LiteralPath $Dest -PathType Leaf) {
        $have = (Get-FileHash -LiteralPath $Dest -Algorithm SHA256).Hash
        if ($have -ieq $Sha) {
            Write-Host "using $Dest"
            return
        }
        Remove-Item -LiteralPath $Dest -Force
    }
    Write-Host "downloading $Url"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest
    $got = (Get-FileHash -LiteralPath $Dest -Algorithm SHA256).Hash
    if ($got -ine $Sha) {
        Remove-Item -LiteralPath $Dest -Force
        throw "SHA-256 mismatch for $Url`n  expected $Sha`n  got      $got"
    }
}

Write-Host "JeuVoice Windows bundle -> $OutDir"

$exe = Join-Path $Repo 'target\release\jeuvoice-app.exe'
if (-not $SkipBuild) {
    & (Join-Path $Repo 'scripts\cargo.cmd') build --release -p jeuvoice-app
    if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
}
Assert-File $exe 'the release exe (run the build without -SkipBuild)'

# A clean bundle: never copy over an old one, so removed files do not linger.
if (Test-Path -LiteralPath $OutDir) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Copy-Item -LiteralPath $exe -Destination $OutDir

# CUDA: only the two DLLs in the exe's import table. CUDA 13 keeps x64 DLLs in
# bin\x64; older toolkits use bin.
if (-not $CudaPath) {
    $CudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
}
$cudaBin = Join-Path $CudaPath 'bin\x64'
if (-not (Test-Path -LiteralPath $cudaBin -PathType Container)) {
    $cudaBin = Join-Path $CudaPath 'bin'
}
foreach ($dll in $CudaDlls) {
    $src = Join-Path $cudaBin $dll
    Assert-File $src "CUDA $dll (install the CUDA 13.4 toolkit or pass -CudaPath)"
    Copy-Item -LiteralPath $src -Destination $OutDir
}

# Visual C++ runtime, from the newest MSVC redist on this machine.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
Assert-File $vswhere 'vswhere.exe (install Visual Studio with the C++ tools)'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio with the C++ tools was not found.' }
$crt = Get-ChildItem (Join-Path $vs 'VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT') -Directory |
    Sort-Object FullName -Descending |
    Select-Object -First 1
if (-not $crt) { throw "The Visual C++ redist folder is missing under $vs" }
foreach ($dll in $CrtDlls) {
    $src = Join-Path $crt.FullName $dll
    Assert-File $src "the Visual C++ runtime $dll"
    Copy-Item -LiteralPath $src -Destination $OutDir
}

# Python: the archive setup unpacks into <data>\python on first start.
Get-PinnedFile $PyUrl $PySha (Join-Path $OutDir $PyName)

# uv: the installer runs it for `uv venv` and `uv pip install`.
$tmp = Join-Path ([IO.Path]::GetTempPath()) ('jeuvoice-uv-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
try {
    $zip = Join-Path $tmp $UvAsset
    Get-PinnedFile $UvUrl $UvSha $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    $uv = Join-Path $tmp 'uv.exe'
    Assert-File $uv 'uv.exe in the downloaded zip'
    Copy-Item -LiteralPath $uv -Destination $OutDir
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

# Engine source without the dev venv, caches and tests. The lock file stays at
# the bundle root; runtime_paths.rs looks for it next to the exe. robocopy skips
# the venv instead of copying and deleting it.
$engineSrc = Join-Path $Repo 'engine'
$engineOut = Join-Path $OutDir 'engine'
$null = robocopy $engineSrc $engineOut /E /XD .venv __pycache__ .pytest_cache .mypy_cache tests /NFL /NDL /NJH /NJS
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
Copy-Item -LiteralPath (Join-Path $engineSrc 'requirements-win-cu126.txt') -Destination $OutDir

# Presets, recipes and the Kaggle template.
Copy-Item -LiteralPath (Join-Path $Repo 'assets') -Destination (Join-Path $OutDir 'assets') -Recurse

$required = @(
    'jeuvoice-app.exe',
    'uv.exe',
    $PyName,
    'cublas64_13.dll',
    'cublasLt64_13.dll',
    'requirements-win-cu126.txt',
    'engine\jeuvoice_engine\__main__.py',
    'engine\overrides.txt',
    'assets\presets',
    'assets\recipes'
)
foreach ($rel in $required) {
    $p = Join-Path $OutDir $rel
    if (-not (Test-Path -LiteralPath $p)) { throw "the bundle is incomplete: $rel is missing" }
}
if (Test-Path -LiteralPath (Join-Path $engineOut '.venv')) { throw 'the dev venv leaked into the bundle' }
if (Test-Path -LiteralPath (Join-Path $engineOut 'tests')) { throw 'the engine tests leaked into the bundle' }

$size = (Get-ChildItem -LiteralPath $OutDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Bundle ready: {0} ({1:N0} MB)" -f $OutDir, ($size / 1MB))
Write-Host 'Test: copy the folder to a path with spaces and Turkish letters and start jeuvoice-app.exe.'
