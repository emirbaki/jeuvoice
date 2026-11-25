# Measures the Phase 1 criteria: whisper.cpp (whisper-rs) against CudaWhisper on the test
# clips. Each backend runs in its own process. The script samples the used video memory of
# the whole card every 100 ms and takes the peak minus the idle value before the run (the
# per-process value is not available on the Windows WDDM driver).
#
#   powershell -ExecutionPolicy Bypass -File scripts\compare-whisper.ps1 [clip ...]
param([string[]]$Clips = @(
    'D:\MLProjelerim\tts_lab\data\ref\my_voice.ogg',
    'D:\MLProjelerim\tts_lab\data\ref\my_voice2.mp4',
    'D:\MLProjelerim\tts_lab\data\ref\sesim\ses_001_0007.wav'
))
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

& (Join-Path $Repo 'scripts\cargo.cmd') oxide build -- -p jeuvoice-whisper --release --features cuda-kernels --example compare_whisper
if ($LASTEXITCODE -ne 0) { throw 'the build of compare_whisper failed' }
$env:PATH = "$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\bin;$env:PATH"
$exe = Join-Path $Repo 'target\release\examples\compare_whisper.exe'
$out = Join-Path $Repo 'target\compare'
New-Item -ItemType Directory -Force $out | Out-Null

function Get-UsedMiB { [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim() }

$peak = @{}
foreach ($b in @('rs', 'cuda')) {
    Start-Sleep -Seconds 2
    $idle = Get-UsedMiB
    $argList = @('run', '--backend', $b, '--out', "`"$out\$b.json`"") + ($Clips | ForEach-Object { "`"$_`"" })
    $p = Start-Process -FilePath $exe -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardOutput "$out\$b.txt" -RedirectStandardError "$out\$b.err"
    # Read the handle now: without it, ExitCode stays empty after the process ends.
    $null = $p.Handle
    $max = 0
    while (-not $p.HasExited) {
        $max = [Math]::Max($max, (Get-UsedMiB))
        Start-Sleep -Milliseconds 100
    }
    $p.WaitForExit()
    if ($p.ExitCode -ne 0) { Get-Content "$out\$b.err"; throw "the $b run failed" }
    Get-Content "$out\$b.txt"
    $peak[$b] = $max - $idle
    "$b peak VRAM above idle: $($peak[$b]) MiB (idle $idle MiB)"
}
& $exe report "$out\rs.json" "$out\cuda.json" --vram-rs $peak['rs'] --vram-cuda $peak['cuda']
