# Moderate first-pass 25-game broad rollout.
# Defaults to ~10 episodes/game across the staged pipeline so the run is
# scoped (~250 episodes total) and produces enough signal to triage.

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 1) { $args[0] } else { "probe_all_broad_$Stamp" }
$OutputRoot = Join-Path "Local_Output\Probe_Cache" $RunName

$EpisodesPerGame = if ($env:PROBE_EPS_PER_GAME) { $env:PROBE_EPS_PER_GAME } else { "10" }
$Allocator = if ($env:PROBE_ALLOCATOR) { $env:PROBE_ALLOCATOR } else { "adaptive" }

# Broad-pass stage budgets — keep harvest_best small since promising games are
# unknown until after triage. Sum should match $EpisodesPerGame approximately.
python -m src.collect_probe_staged `
    --project-root "." `
    --output-root "$OutputRoot" `
    --all-public-games `
    --episodes-per-game $EpisodesPerGame `
    --budget-probe-seed 2 `
    --budget-exploit-safe 4 `
    --budget-followup-focus 2 `
    --budget-rescue-reprobe 1 `
    --budget-harvest-best 1 `
    --budget-allocator $Allocator

$EpisodesPath = Join-Path $OutputRoot "collected\episodes.jsonl.gz"
$EvalPath = Join-Path $OutputRoot "probe_eval.json"
$PriorsDir = Join-Path $OutputRoot "priors"
$TriageDir = Join-Path $OutputRoot "triage"

python -m src.eval_probe `
    --input "$EpisodesPath" `
    --output "$EvalPath" `
    --label probe_all_broad `
    --print-overall

python -m src.probe_triage `
    --input "$EpisodesPath" `
    --priors "$PriorsDir" `
    --output-dir "$TriageDir"

Write-Host ""
Write-Host "25-game broad pass done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
Write-Host "  triage:   $TriageDir\triage_summary.json"
Write-Host "  per-stage: $OutputRoot\per_game_stage_summary.json"
Write-Host ""
Write-Host "Next step: review $TriageDir\triage_summary.csv, then run"
Write-Host "  scripts\run_harvest_promising_local.ps1 $RunName"
Write-Host "to scale up promising games using the priors collected here."
