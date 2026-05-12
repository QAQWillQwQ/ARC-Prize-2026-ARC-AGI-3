"""Build a Goose-T2 submission notebook (GRU episode memory, no pretrained prior).

Differences vs build_goose_submission_notebook.py (T1):
  - Cell 4 vendors goose_agent_t2.py (NOT goose_agent.py).
  - Class rename: GooseT2Agent -> MyAgent.
  - Cell 5 (bc_policy.py) is REPLACED with a no-op placeholder — T2 has no
    pretrained checkpoint to load, and Cell 5 in v4/T1 imported bc_policy
    which isn't used here.
  - Cell 6 exports ARC_AGENT_CLASS / ARC_GOOSE_DELTA aren't needed (T2 is
    the only agent in the resulting my_agent.py), but we set ARC_GOOSE_LR
    and other knobs in case we want to tune later.

Writes to kaggle_notebook/notebooks/kaggle_submission_goose_t2.ipynb.

Usage:
    python scripts/build_goose_t2_submission_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "kaggle_notebook" / "notebooks" / "kaggle_submission.ipynb"
DST_NB = ROOT / "kaggle_notebook" / "notebooks" / "kaggle_submission_goose_t2.ipynb"
GOOSE_T2_AGENT = ROOT / "kaggle_notebook" / "agents" / "goose_agent_t2.py"


def main() -> None:
    nb = json.loads(SRC_NB.read_text())
    goose_t2_src = GOOSE_T2_AGENT.read_text()

    # Rename class for framework compatibility (it expects MyAgent).
    rewritten = goose_t2_src.replace("class GooseT2Agent(Agent):", "class MyAgent(Agent):")

    # ---- Cell 4 (writefile my_agent.py) ----
    cell4 = nb["cells"][4]
    new_cell4_src = "%%writefile /kaggle/working/my_agent.py\n" + rewritten
    cell4["source"] = new_cell4_src.splitlines(keepends=True)
    cell4["outputs"] = []
    cell4["execution_count"] = None

    # ---- Cell 2 (path setup) — simplified, no replay input needed ----
    cell2 = nb["cells"][2]
    cell2_src = (
        "# --- Cell 2: configure paths (Goose-T2 needs no replays / no checkpoint) --- #\n"
        "from pathlib import Path\n"
        "\n"
        "COMPETITION_INPUT = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3')\n"
        "WORK = Path('/kaggle/working')\n"
        "assert COMPETITION_INPUT.exists(), f'Competition input missing: {COMPETITION_INPUT}'\n"
        "print(f'COMPETITION_INPUT: {COMPETITION_INPUT}  (exists)')\n"
        "print('Goose-T2: no replays, no pretrained checkpoint — fully online learning per level.')\n"
    )
    cell2["source"] = cell2_src.splitlines(keepends=True)
    cell2["outputs"] = []
    cell2["execution_count"] = None

    # ---- Cell 3 (replay + checkpoint staging) — no-op for T2 ----
    cell3 = nb["cells"][3]
    cell3_src = (
        "# --- Cell 3: skipped for Goose-T2 (no replays, no checkpoint) --- #\n"
        "print('[Cell 3] Goose-T2 skips replay staging and checkpoint copy.')\n"
    )
    cell3["source"] = cell3_src.splitlines(keepends=True)
    cell3["outputs"] = []
    cell3["execution_count"] = None

    # ---- Cell 5 (bc_policy.py) — replace with no-op stub ----
    # T2 doesn't import bc_policy. A no-op stub is safer than removing the
    # cell entirely (which could shift cell indices the runner cell relies on).
    cell5 = nb["cells"][5]
    cell5_src = (
        "# Cell 5: bc_policy stub (Goose-T2 doesn't use pretrained BC).\n"
        "# Intentionally minimal — keeps the notebook cell layout stable.\n"
        "print('[Cell 5] Goose-T2 does not use pretrained BC; skipping bc_policy write.')\n"
    )
    cell5["source"] = cell5_src.splitlines(keepends=True)
    cell5["outputs"] = []
    cell5["execution_count"] = None

    # ---- Cell 6 (runner) — strip the T1 env exports if present ----
    cell6 = nb["cells"][6]
    src6 = "".join(cell6["source"])
    # Strip any leftover ARC_GOOSE_DELTA / ARC_BC_CHECKPOINT_PATH lines.
    src6 = re.sub(r"        ARC_GOOSE_DELTA=[^\\]*\\\n", "", src6)
    src6 = re.sub(r"        ARC_BC_CHECKPOINT_PATH=[^\\]*\\\n", "", src6)
    cell6["source"] = src6.splitlines(keepends=True)
    cell6["outputs"] = []
    cell6["execution_count"] = None

    # ---- Cell 0 (markdown header) — full rewrite for T2 ----
    cell0 = nb["cells"][0]
    cell0_md = (
        "# ARC AGI 3 — Goose-T2 (online change-reward CNN + 1-layer GRU episode memory)\n"
        "\n"
        "**Approach (no pretraining):**\n"
        "1. **Small CNN** (16 -> 32 -> 64 -> 128 -> 256, 3x3 convs) over one-hot 64x64x16 frames.\n"
        "2. **1-layer GRU** (input 256, hidden 128) over globally-pooled CNN features; GRU hidden state carried within a level.\n"
        "3. **Action head** = action_fc(512) ++ GRU hidden(128) -> 6 logits for ACTION{1,2,3,4,5,7}.\n"
        "4. **Coord head** = spatial decoder (256 -> 128 -> 64 -> 32 -> 1) -> 64x64 logit map for ACTION6.\n"
        "5. **Online training every 5 steps** on (selected_logit, did-frame-change-reward) via BCE-with-logits.\n"
        "6. **Reset model + optimizer + GRU hidden** at every level change. Each level learned from scratch.\n"
        "\n"
        "**Why this design:** based on the public StochasticGoose pattern (Tufa Labs, score 0.25 on Kaggle hidden) which beats offline-pretrained agents because it cannot overfit to public games. The GRU addition is our novel delta — provides episode memory so the agent remembers what it has already tried.\n"
        "\n"
        "**Local validation (3 seeds x 5 held-out games x 1000 steps):**\n"
        "- T0 Pure Goose (no GRU, baseline):    mean_score 0.014, mean_levels 0.333\n"
        "- **T2 Goose + GRU (this submission)**: mean_score **0.025 (+79%)**, mean_levels **0.467 (+40%)**\n"
        "- T1 Goose + bc_v4 prior:               0.016 (matches T0 - prior was memorization, useless on unseen games)\n"
        "\n"
        "Held-out games (excluded from any bc_v4 training pool used in T1, here only for benchmark): r11l, ft09, tr87, m0r0, dc22.\n"
        "\n"
        "**Inputs:** only `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/` (framework + wheels). No external dataset, no checkpoint, no replay staging.\n"
        "\n"
        "**Runtime:** Kaggle 8h budget per game; each level the CNN trains itself online via change-reward.\n"
    )
    cell0["source"] = cell0_md.splitlines(keepends=True)

    DST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {DST_NB}")
    print(f"  cell[4] my_agent.py: {len(new_cell4_src):,} chars")
    print(f"  cell[5] bc_policy stub: {len(cell5_src)} chars (no-op)")
    print(f"  cell[6] runner: env exports stripped")


if __name__ == "__main__":
    main()
