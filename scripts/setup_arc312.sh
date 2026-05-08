#!/usr/bin/env bash
#
# Set up the `arc312` conda environment for ARC AGI 3 development.
#
# Equivalent to `conda env create -f environment.yml` but more verbose / scriptable.
# Use this if you prefer step-by-step setup or need to retry individual steps.
#
# Usage (from project root):
#   bash scripts/setup_arc312.sh             # creates env named "arc312"
#   bash scripts/setup_arc312.sh my_env_name # custom env name
#
# Requirements:
#   - Linux or WSL (bundled wheels are manylinux_2_28 / cp312 — won't work on Mac)
#   - miniconda or anaconda installed
#   - Internet access (PyTorch index, PyPI fallback for any extras)

set -euo pipefail

ENV_NAME="${1:-arc312}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] Project root: $PROJECT_ROOT"
echo "[setup] Target conda env: $ENV_NAME"
cd "$PROJECT_ROOT"

if [ ! -d "arc_agi_3_wheels" ]; then
    echo "[setup] ERROR: arc_agi_3_wheels/ not found. Run from project root." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
    echo "[setup] ERROR: 'conda' not on PATH. Install miniconda first." >&2
    exit 1
fi

echo "[setup] Sourcing conda from $CONDA_BASE"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] Env '$ENV_NAME' already exists. Activating and reinstalling deps."
    conda activate "$ENV_NAME"
else
    echo "[setup] Creating env '$ENV_NAME' with Python 3.12..."
    conda create -n "$ENV_NAME" python=3.12 -y
    conda activate "$ENV_NAME"
fi

python --version
echo "[setup] Active python: $(which python)"

echo "[setup] Upgrading pip..."
pip install --upgrade "pip>=24" wheel setuptools

echo "[setup] Installing PyTorch 2.5.1 with CUDA 12.1..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "[setup] Installing all bundled wheels (arc_agi, arcengine, cp312 deps)..."
pip install arc_agi_3_wheels/*.whl

echo "[setup] Installing helpful extras..."
pip install tqdm

echo "[setup] Verifying installation..."
python <<'PY'
import torch, arc_agi, arcengine
print(f"  torch:     {torch.__version__}")
print(f"  cuda?:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device:    {torch.cuda.get_device_name(0)}")
    print(f"  cuda ver:  {torch.version.cuda}")
print(f"  arc_agi:   {getattr(arc_agi, '__version__', 'installed')}")
print(f"  arcengine: {getattr(arcengine, '__version__', 'installed')}")
PY

echo ""
echo "[setup] Done."
echo ""
echo "To use this env in a new shell:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To use it in Claude Code:"
echo "  1. Exit Claude (/exit)"
echo "  2. conda activate $ENV_NAME"
echo "  3. Restart claude — the new env is inherited"
