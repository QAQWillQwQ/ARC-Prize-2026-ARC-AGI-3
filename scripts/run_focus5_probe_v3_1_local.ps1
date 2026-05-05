# Focus-5 single-run v3.1 collector + eval.
# Mirrors scripts/run_focus5_probe_v3_1_local.sh but for Windows PowerShell.

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 1) { $args[0] } else { "probe_focus5_v3_1_$Stamp" }
$OutputRoot = Join-Path "Local_Output\Probe_Cache" $RunName

$Games = if ($env:PROBE_GAMES) { $env:PROBE_GAMES } else { "sp80,lp85,ar25,ls20,r11l" }
$Seeds = if ($env:PROBE_SEEDS) { $env:PROBE_SEEDS } else { "0,1,2,3" }

python -m src.collect_probe `
    --project-root "." `
    --output-root "$OutputRoot" `
    --games "$Games" `
    --seeds "$Seeds"

$EpisodesPath = Join-Path $OutputRoot "collected\episodes.jsonl.gz"
$EvalPath = Join-Path $OutputRoot "probe_eval.json"
python -m src.eval_probe `
    --input "$EpisodesPath" `
    --output "$EvalPath" `
    --label probe_v3_1 `
    --print-overall

Write-Host ""
Write-Host "Probe v3.1 single-run done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
