# Broad first pass over all 25 public games. ~250 episodes total, ~6.5 min on
# CPU. Produces the priors and triage that the harvest pass reads.
#
# Usage:
#   .\scripts\run_broad_collection.ps1                # default run name
#   .\scripts\run_broad_collection.ps1 my_broad_run   # custom run name

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 1) { $args[0] } else { "probe_all_broad_$Stamp" }
$OutRoot = Join-Path "outputs\probe_cache" $RunName

$EpisodesPerGame = if ($env:PROBE_EPS_PER_GAME) { $env:PROBE_EPS_PER_GAME } else { "10" }
$Allocator       = if ($env:PROBE_ALLOCATOR)    { $env:PROBE_ALLOCATOR }    else { "adaptive" }

python -m src.collect.probe_staged `
    --project-root         "." `
    --output-root          "$OutRoot" `
    --all-public-games `
    --episodes-per-game    $EpisodesPerGame `
    --budget-probe-seed    2 `
    --budget-exploit-safe  4 `
    --budget-followup-focus 2 `
    --budget-rescue-reprobe 1 `
    --budget-harvest-best  1 `
    --budget-allocator     $Allocator

$EpisodesPath = Join-Path $OutRoot "collected\episodes.jsonl.gz"
$EvalPath     = Join-Path $OutRoot "probe_eval.json"
$PriorsDir    = Join-Path $OutRoot "priors"
$TriageDir    = Join-Path $OutRoot "triage"

python -m src.eval.probe `
    --input  "$EpisodesPath" `
    --output "$EvalPath" `
    --label  probe_all_broad `
    --print-overall

python -m src.eval.triage `
    --input      "$EpisodesPath" `
    --priors     "$PriorsDir" `
    --output-dir "$TriageDir"

Write-Host ""
Write-Host "Broad pass done:"
Write-Host "  episodes:  $EpisodesPath"
Write-Host "  eval:      $EvalPath"
Write-Host "  triage:    $TriageDir\triage_summary.json"
Write-Host ""
Write-Host "Next step: review $TriageDir\triage_summary.csv, then run"
Write-Host "  .\scripts\run_harvest.ps1 $RunName"
Write-Host "to scale up promising games using the priors collected here."
