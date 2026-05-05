# Evaluate a trained checkpoint on the public ARC-AGI-3 games.
#
# Usage:
#   .\scripts\evaluate_checkpoint.ps1 <checkpoint_path> [<output_name>]
#
# Examples:
#   .\scripts\evaluate_checkpoint.ps1 outputs\training\train_focus5_v1\checkpoints\best.pth
#   .\scripts\evaluate_checkpoint.ps1 outputs\training\train_focus5_v1\checkpoints\best.pth focus5_eval

$ErrorActionPreference = "Stop"

if ($args.Count -lt 1) {
    Write-Host "Usage: evaluate_checkpoint.ps1 <checkpoint_path> [<output_name>]"
    exit 1
}

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

if (Test-Path ".\.venv313\Scripts\Activate.ps1") {
    . ".\.venv313\Scripts\Activate.ps1"
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . ".\.venv\Scripts\Activate.ps1"
}

$Checkpoint = $args[0]
if (-not (Test-Path $Checkpoint)) {
    Write-Host "Checkpoint not found: $Checkpoint"
    exit 1
}

$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$RunName = if ($args.Count -ge 2) { $args[1] } else { "eval_$Stamp" }
$OutFile = Join-Path "outputs\evaluation" "$RunName.json"
$OutDir  = Split-Path -Parent $OutFile
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$Games = if ($env:EVAL_GAMES) { "--games $env:EVAL_GAMES" } else { "" }

$cmd = "python -m src.eval.policy --project-root `".`" --checkpoint `"$Checkpoint`" --output `"$OutFile`" $Games"
Invoke-Expression $cmd

Write-Host ""
Write-Host "Evaluation done: $OutFile"
