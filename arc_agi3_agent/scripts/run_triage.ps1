# Re-run triage on an existing collected run. Cheap: reads episodes.jsonl.gz
# and rewrites the triage\ folder. Useful after editing triage thresholds.
#
# Usage:
#   .\scripts\run_triage.ps1 <run_name>

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    Write-Host "Usage: run_triage.ps1 <run_name>"
    exit 1
}

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Run = $args[0]
$Dir = Join-Path "outputs\probe_cache" $Run
if (-not (Test-Path $Dir)) {
    Write-Host "Run directory not found: $Dir"
    exit 1
}

$Episodes  = Join-Path $Dir "collected\episodes.jsonl.gz"
$Priors    = Join-Path $Dir "priors"
$TriageDir = Join-Path $Dir "triage"

python -m src.eval.triage `
    --input      "$Episodes" `
    --priors     "$Priors" `
    --output-dir "$TriageDir"

Write-Host "Triage rewritten to $TriageDir"
