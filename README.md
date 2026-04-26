# ARC Prize 2026 ARC AGI 3

This repo contains a minimal end to end pipeline for ARC AGI 3.

The current method uses structured search on 25 public games to collect useful trajectories, then trains a compact object centric policy model from those trajectories. The model predicts the next action, the click position for `ACTION6`, a value estimate & a small transition target for the next latent state. At inference time, the agent uses the trained policy together with short term memory about action effects, progress, and previously tried coordinates.

The repo is intentionally small. The only maintained project code lives in `src/` and `ColabNotebook/`. The bundled `ARC-AGI-3-Agents/` folder & `arc_agi_3_wheels/` folder are kept as official reference assets from the dataset package.

## Files

- `src/collect.py` collects public game trajectories with structured search
- `src/train.py` trains the policy model
- `src/evaluate.py` runs public game evaluation
- `src/competition.py` runs the agent in competition mode
- `ColabNotebook/train_arc_agi3_colab.ipynb` is the Colab training entry point

## Competition Notes

The current implementation follows the public rules checked on April 22, 2026.

- Kaggle submissions are generated automatically after the notebook interacts with the competition environments
- The competition notebook runs in `Competition Mode`
- In `Competition Mode`, the run uses one scorecard and each environment can only be created once
- The scorecard is closed at the end of the run

The local validation code uses the current public scoring cap of `115%` per level.

## Hardware Recommendation

The default recommendation for the first training runs is Colab `A100`.

- The public environment count is small, so search quality and generalization matter more than very large model scale
- The current model size and training loop fit well on A100 without wasting too much capacity
- H100 can be used, but it is not required for the first version

Suggested starting profile:

- GPU: `A100`
- Precision: `bf16`
- `model_dim=384`
- `depth=6`
- `num_slots=8`
- `batch_size=192`
- `epochs=16`

Fallback profile for `RTX 3070 Ti 8GB`:

- `model_dim=256`
- `depth=4`
- `batch_size=16`
- `grad_accum=8`

## Training Output

The Colab notebook copies project code from:

`ARC Prize 2026 - ARC-AGI-3/`

The Colab notebook writes outputs to:

`ARC Prize 2026_AGI_3/Training_Output/<timestamp>/`

Cached public trajectory collection can be stored separately at:

`ARC Prize 2026_AGI_3/Collection_Cache/<collect_tag>/`

For a local precompute run on a Mac M3 CPU, a lighter profile is available:

- hardware profile: `m3_cpu`
- intended use: run `collect` locally once, then reuse the cached `episodes.jsonl.gz` for multiple Colab training runs

The output folder includes:

- `metrics.csv`
- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `checkpoints/interrupt.pth` when training is stopped manually
- `summary.json`

The cached collection folder includes:

- `collected/episodes.jsonl.gz`

## Local Collect on Mac M3

You can precompute the public search trajectories on a local Mac M3 CPU, then upload the resulting folder to Google Drive and reuse it in Colab.

On macOS, install dependencies from PyPI in a local virtual environment. The bundled wheel folder includes several Linux-only dependency wheels and should not be used as the main install source on Mac.

Recommended first pass command:

```bash
cd '/Users/wangyiding/ARC Prize 2026 - ARC-AGI-3'
python -m src.collect \
  --project-root "." \
  --output-root "./Local_Output/Collection_Cache/public_search_m3_v1" \
  --hardware-profile m3_cpu \
  --seeds 0,1 \
  --max-steps 64
```

This creates:

`./Local_Output/Collection_Cache/public_search_m3_v1/collected/episodes.jsonl.gz`

After the local run finishes, copy the `public_search_m3_v1` folder to:

`ARC Prize 2026_AGI_3/Collection_Cache/public_search_m3_v1/`

Then set:

- `RUN_COLLECTION = False`
- `COLLECT_TAG = 'public_search_m3_v1'`

in the Colab notebook so training reuses the cached trajectories instead of recollecting them.

## Openlab Collect

For longer CPU-heavy collection runs on UCI ICS Openlab, use Slurm instead of keeping a long interactive shell job. Openlab documentation notes that long-running non-Slurm processes may be reniced or suspended, while Slurm jobs are exempt.

Copy the project to Openlab:

```bash
rsync -av --delete --exclude '.git' \
  '/Users/wangyiding/ARC Prize 2026 - ARC-AGI-3/' \
  yidingw6@openlab.ics.uci.edu:~/arc_agi3/
```

On Openlab, create a virtual environment and install from the bundled Linux wheels:

```bash
cd ~/arc_agi3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install arc_agi_3_wheels/*.whl
```

Single-node parallel collect on Openlab:

```bash
cd ~/arc_agi3
source .venv/bin/activate
python -m src.collect \
  --project-root "." \
  --output-root "./Local_Output/Collection_Cache/openlab_search_v1" \
  --hardware-profile a100 \
  --seeds 0,1,2,3 \
  --max-steps 96 \
  --workers 16
```

The `--workers` flag parallelizes collection across CPU processes on one node. Increase it only as far as the node memory and process limits allow.

## References

Official competition and toolkit references:

- [Kaggle Competition Overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview)
- [Kaggle Competition Data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/data)
- [ARC AGI 3 Scoring Methodology](https://docs.arcprize.org/methodology)
- [ARC AGI Toolkit](https://github.com/arcprize/arc-agi)
- [ARC AGI 3 Agents](https://github.com/arcprize/ARC-AGI-3-Agents)

Method references that informed the design:

- [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [Plan2Explore](https://arxiv.org/abs/2005.05960)
- [Slot Attention](https://arxiv.org/abs/2006.15055)
- [DITTO](https://openreview.net/forum?id=Ix4Ytiwor4U)
- [Transformers meet Neural Algorithmic Reasoners](https://arxiv.org/abs/2406.09308)
- [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)
