# The Phase 2 baseline: the Python engine's own path, stage by stage, and the peak VRAM of the
# card. It runs tools\reference\omnivoice_reference.py --bench, which loads the model in fp16 on
# the GPU, makes the voice prompt, and renders script s1 three times. The tool prints the load
# time and the median of the whole render, the token stage (our Phase 2) and the audio decode
# (Phase 3). This script adds the VRAM: the whole card, peak minus idle (WDDM does not report
# per process).
#
#   powershell -ExecutionPolicy Bypass -File scripts\tts-baseline.ps1 [-Renders 3]
param(
    [int]$Renders = 3
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'engine\.venv\Scripts\python.exe'
$tool = Join-Path $repo 'tools\reference\omnivoice_reference.py'
$out = Join-Path $repo 'target\compare'

if (-not (Test-Path $python)) { throw "the engine venv is missing: $python" }
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Get-UsedMiB { [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim() }

$idle = Get-UsedMiB
Write-Host "GPU memory idle: $idle MiB"

$log = Join-Path $out 'tts-baseline.txt'
$err = Join-Path $out 'tts-baseline.err'
$args = @('-I', $tool, '--bench', "$Renders", '--skip-taps', '--skip-product')
$p = Start-Process -FilePath $python -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
# Read the handle now: without it, ExitCode stays empty after the process ends.
$null = $p.Handle
$peak = 0
while (-not $p.HasExited) {
    $peak = [Math]::Max($peak, (Get-UsedMiB))
    Start-Sleep -Milliseconds 200
}
$peak = [Math]::Max($peak, (Get-UsedMiB))
if ($p.ExitCode -ne 0) { throw "the reference tool failed (exit $($p.ExitCode)): see $err" }

Get-Content $log | Where-Object { $_ -match 'model loaded|prompt in|engine path|whole render|tokens only|audio decode' }
Write-Host ""
Write-Host ("peak VRAM above idle: {0} MiB (peak {1}, idle {2})" -f ($peak - $idle), $peak, $idle)
