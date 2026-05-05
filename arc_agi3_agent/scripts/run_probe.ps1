# Single-pass probe-first collection on the focus-5 game subset, followed by
# probe evaluation. Wall-clock ~100s on a laptop CPU.
#
# Usage:
#   .\scripts\run_probe.ps1                    # default run name
#   .\scripts\run_probe.ps1 my_run_name        # custom run name
#
# Override game / seed list via env vars:
#   $env:PROBE_GAMES = "sp80,lp85"
#   $env:PROBE_SEEDS = "0,1"

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 1) { $args[0] } else { "probe_focus5_$Stamp" }
$OutRoot = Join-Path "outputs\probe_cache" $RunName

$Games = if ($env:PROBE_GAMES) { $env:PROBE_GAMES } else { "sp80,lp85,ar25,ls20,r11l" }
$Seeds = if ($env:PROBE_SEEDS) { $env:PROBE_SEEDS } else { "0,1,2,3" }

python -m src.collect.probe `
    --project-root "." `
    --output-root  "$OutRoot" `
    --games        "$Games" `
    --seeds        "$Seeds"

$EpisodesPath = Join-Path $OutRoot "collected\episodes.jsonl.gz"
$EvalPath     = Join-Path $OutRoot "probe_eval.json"

python -m src.eval.probe `
    --input  "$EpisodesPath" `
    --output "$EvalPath" `
    --label  probe_v3_1 `
    --print-overall

Write-Host ""
Write-Host "Probe single-pass run done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
