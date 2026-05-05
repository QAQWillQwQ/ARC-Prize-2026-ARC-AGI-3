# Second-pass harvest on games triaged as promising / signal_but_stuck.
# Reuses priors from a previous broad run so warm-start carries forward.
#
# Usage:
#   .\scripts\run_harvest.ps1 <broad_run_name> [<output_run_name>]
#
# <broad_run_name> must already exist under outputs\probe_cache and contain
# a priors\ folder and triage\triage_summary.json.

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    Write-Host "Usage: run_harvest.ps1 <broad_run_name> [<output_run_name>]"
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
$BroadDir = Join-Path "outputs\probe_cache" $BroadRun
if (-not (Test-Path $BroadDir)) {
    Write-Host "Broad run directory not found: $BroadDir"
    exit 1
}

$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 2) { $args[1] } else { "${BroadRun}_harvest_$Stamp" }
$OutRoot = Join-Path "outputs\probe_cache" $RunName

$BroadPriors   = Join-Path $BroadDir "priors"
$HarvestPriors = Join-Path $OutRoot  "priors"
if (Test-Path $HarvestPriors) {
    Write-Host "Refusing to clobber existing priors at $HarvestPriors"
    exit 1
}
New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null
Copy-Item -Path $BroadPriors -Destination $HarvestPriors -Recurse

$Triage = Join-Path $BroadDir "triage\triage_summary.json"
if (-not (Test-Path $Triage)) {
    Write-Host "No triage summary at $Triage. Run scripts\run_broad_collection.ps1 first."
    exit 1
}

$Labels = if ($env:PROBE_HARVEST_LABELS) {
    $env:PROBE_HARVEST_LABELS
} else {
    "promising,signal_but_stuck,click_promising,movement_promising,followup_strong"
}

python -m src.collect.probe_staged `
    --project-root          "." `
    --output-root           "$OutRoot" `
    --all-public-games `
    --restrict-from-triage  "$Triage" `
    --include-labels        "$Labels" `
    --episodes-per-game     18 `
    --budget-probe-seed     1 `
    --budget-exploit-safe   4 `
    --budget-followup-focus 5 `
    --budget-rescue-reprobe 4 `
    --budget-harvest-best   4 `
    --budget-allocator      adaptive

$EpisodesPath = Join-Path $OutRoot "collected\episodes.jsonl.gz"
$EvalPath     = Join-Path $OutRoot "probe_eval.json"
$TriageDir    = Join-Path $OutRoot "triage"

python -m src.eval.probe `
    --input  "$EpisodesPath" `
    --output "$EvalPath" `
    --label  probe_harvest `
    --print-overall

python -m src.eval.triage `
    --input      "$EpisodesPath" `
    --priors     "$HarvestPriors" `
    --output-dir "$TriageDir"

Write-Host ""
Write-Host "Harvest pass done:"
Write-Host "  episodes: $EpisodesPath"
Write-Host "  eval:     $EvalPath"
Write-Host "  triage:   $TriageDir\triage_summary.json"
Write-Host ""
Write-Host "For training, pass both broad and harvest episodes to src.train.train"
Write-Host "via comma-separated --data."
