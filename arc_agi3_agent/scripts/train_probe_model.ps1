# Train the object-centric policy on collected probe trajectories.
#
# Usage:
#   .\scripts\train_probe_model.ps1 <run_name> [<data_glob_or_csv>]
#
# Examples:
#   .\scripts\train_probe_model.ps1 train_focus5_v1
#       (auto-discovers episodes.jsonl.gz under outputs\probe_cache\*)
#
#   .\scripts\train_probe_model.ps1 train_focus5_v1 `
#       "outputs\probe_cache\probe_focus5_v1\collected\episodes.jsonl.gz,outputs\probe_cache\probe_all_broad_v1\collected\episodes.jsonl.gz"

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    Write-Host "Usage: train_probe_model.ps1 <run_name> [<data_csv>]"
    exit 1
}

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$RunName = $args[0]
$OutDir  = Join-Path "outputs\training" $RunName

if ($args.Count -ge 2) {
    $DataArg = $args[1]
} else {
    $matches = Get-ChildItem -Path "outputs\probe_cache" -Recurse -Filter "episodes.jsonl.gz" -ErrorAction SilentlyContinue
    if (-not $matches) {
        Write-Host "No episodes.jsonl.gz found under outputs\probe_cache. Run a collection first."
        exit 1
    }
    $DataArg = ($matches | ForEach-Object { $_.FullName }) -join ","
    Write-Host "Auto-discovered $($matches.Count) episode files."
}

$Profile = if ($env:ARC_HW_PROFILE) { $env:ARC_HW_PROFILE } else { "a100" }

python -m src.train.train `
    --project-root     "." `
    --data             "$DataArg" `
    --output-dir       "$OutDir" `
    --hardware-profile $Profile

Write-Host ""
Write-Host "Training run done:"
Write-Host "  output:     $OutDir"
Write-Host "  best ckpt:  $OutDir\checkpoints\best.pth"
Write-Host "  metrics:    $OutDir\metrics.csv"
