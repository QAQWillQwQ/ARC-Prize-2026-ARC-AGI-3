# Focus-5 staged collector run + triage.
# Equivalent to staged_v1 but with adaptive budget allocation and triage outputs.

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 1) { $args[0] } else { "probe_focus5_staged_v2_$Stamp" }
$OutputRoot = Join-Path "Local_Output\Probe_Cache" $RunName

$Games = if ($env:PROBE_GAMES) { $env:PROBE_GAMES } else { "sp80,lp85,ar25,ls20,r11l" }
$EpisodesPerGame = if ($env:PROBE_EPS_PER_GAME) { $env:PROBE_EPS_PER_GAME } else { "32" }
$Allocator = if ($env:PROBE_ALLOCATOR) { $env:PROBE_ALLOCATOR } else { "adaptive" }

python -m src.collect_probe_staged `
    --project-root "." `
    --output-root "$OutputRoot" `
    --games "$Games" `
    --episodes-per-game $EpisodesPerGame `
    --budget-allocator $Allocator

$EpisodesPath = Join-Path $OutputRoot "collected\episodes.jsonl.gz"
$EvalPath = Join-Path $OutputRoot "probe_eval.json"
$PriorsDir = Join-Path $OutputRoot "priors"
$TriageDir = Join-Path $OutputRoot "triage"

python -m src.eval_probe `
    --input "$EpisodesPath" `
    --output "$EvalPath" `
    --label probe_staged_v2 `
    --print-overall

python -m src.probe_triage `
    --input "$EpisodesPath" `
    --priors "$PriorsDir" `
    --output-dir "$TriageDir"

Write-Host ""
Write-Host "Focus-5 staged run done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
Write-Host "  triage:   $TriageDir\triage_summary.json"
