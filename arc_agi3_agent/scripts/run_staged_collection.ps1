# Staged probe collection on the focus-5 game subset, followed by probe
# evaluation and per-game triage. Runs the full five-stage pipeline:
#   probe_seed -> exploit_safe -> followup_focus -> rescue_reprobe -> harvest_best
#
# Usage:
#   .\scripts\run_staged_collection.ps1                # default run name
#   .\scripts\run_staged_collection.ps1 my_run_name    # custom run name
#
# Override game list / episodes / allocator via env vars:
#   $env:PROBE_GAMES        = "sp80,lp85"
#   $env:PROBE_EPS_PER_GAME = "32"
#   $env:PROBE_ALLOCATOR    = "adaptive"   # or "uniform"

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 1) { $args[0] } else { "probe_focus5_staged_$Stamp" }
$OutRoot = Join-Path "outputs\probe_cache" $RunName

$Games           = if ($env:PROBE_GAMES)        { $env:PROBE_GAMES }        else { "sp80,lp85,ar25,ls20,r11l" }
$EpisodesPerGame = if ($env:PROBE_EPS_PER_GAME) { $env:PROBE_EPS_PER_GAME } else { "32" }
$Allocator       = if ($env:PROBE_ALLOCATOR)    { $env:PROBE_ALLOCATOR }    else { "adaptive" }

python -m src.collect.probe_staged `
    --project-root      "." `
    --output-root       "$OutRoot" `
    --games             "$Games" `
    --episodes-per-game $EpisodesPerGame `
    --budget-allocator  $Allocator

$EpisodesPath = Join-Path $OutRoot "collected\episodes.jsonl.gz"
$EvalPath     = Join-Path $OutRoot "probe_eval.json"
$PriorsDir    = Join-Path $OutRoot "priors"
$TriageDir    = Join-Path $OutRoot "triage"

python -m src.eval.probe `
    --input  "$EpisodesPath" `
    --output "$EvalPath" `
    --label  probe_staged `
    --print-overall

python -m src.eval.triage `
    --input      "$EpisodesPath" `
    --priors     "$PriorsDir" `
    --output-dir "$TriageDir"

Write-Host ""
Write-Host "Staged collection done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
Write-Host "  triage:   $TriageDir\triage_summary.json"
