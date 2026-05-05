# Second-pass harvest on games triaged as promising / signal_but_stuck.
# Reuses priors from a previous broad run (so warm-start carries forward).
#
# Usage:
#   scripts\run_harvest_promising_local.ps1 <broad_run_name>
# where <broad_run_name> matches a directory under Local_Output\Probe_Cache.

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    Write-Host "Usage: run_harvest_promising_local.ps1 <broad_run_name> [<output_run_name>]"
    Write-Host "  <broad_run_name> must already exist under Local_Output\Probe_Cache."
    exit 1
}

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$BroadRun = $args[0]
$BroadDir = Join-Path "Local_Output\Probe_Cache" $BroadRun
if (-not (Test-Path $BroadDir)) {
    Write-Host "Broad run directory not found: $BroadDir"
    exit 1
}

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 2) { $args[1] } else { "${BroadRun}_harvest_$Stamp" }
$OutputRoot = Join-Path "Local_Output\Probe_Cache" $RunName

# Seed the harvest run with the broad run's priors so warm-start carries.
$BroadPriors = Join-Path $BroadDir "priors"
$HarvestPriors = Join-Path $OutputRoot "priors"
if (Test-Path $HarvestPriors) {
    Write-Host "Refusing to clobber existing priors at $HarvestPriors"
    exit 1
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
Copy-Item -Path $BroadPriors -Destination $HarvestPriors -Recurse

$Triage = Join-Path $BroadDir "triage\triage_summary.json"
if (-not (Test-Path $Triage)) {
    Write-Host "No triage summary at $Triage. Run scripts\run_staged_all_local.ps1 first."
    exit 1
}

$Labels = if ($env:PROBE_HARVEST_LABELS) {
    $env:PROBE_HARVEST_LABELS
} else {
    "promising,signal_but_stuck,click_promising,movement_promising,followup_strong"
}

# Lean toward harvest_best + rescue_reprobe + followup_focus on these games.
python -m src.collect_probe_staged `
    --project-root "." `
    --output-root "$OutputRoot" `
    --all-public-games `
    --restrict-from-triage "$Triage" `
    --include-labels "$Labels" `
    --episodes-per-game 18 `
    --budget-probe-seed 1 `
    --budget-exploit-safe 4 `
    --budget-followup-focus 5 `
    --budget-rescue-reprobe 4 `
    --budget-harvest-best 4 `
    --budget-allocator adaptive

$EpisodesPath = Join-Path $OutputRoot "collected\episodes.jsonl.gz"
$EvalPath = Join-Path $OutputRoot "probe_eval.json"
$TriageDir = Join-Path $OutputRoot "triage"

python -m src.eval_probe `
    --input "$EpisodesPath" `
    --output "$EvalPath" `
    --label probe_harvest `
    --print-overall

python -m src.probe_triage `
    --input "$EpisodesPath" `
    --priors "$HarvestPriors" `
    --output-dir "$TriageDir"

Write-Host ""
Write-Host "Harvest pass done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
Write-Host "  triage:   $TriageDir\triage_summary.json"
Write-Host "  Combined dataset for training: pass both episodes.jsonl.gz files"
Write-Host "  to src.train via comma-separated --data."
