"""Build a Goose-T1 submission notebook from the existing v4 submission template.

Takes the current kaggle_notebook/notebooks/kaggle_submission.ipynb, swaps Cell 4
(my_agent.py) for the goose_agent.py contents with class renamed
GooseAgent -> MyAgent, leaves Cell 5 (bc_policy.py) untouched, and edits
Cell 6 to export the env vars Goose-T1 needs.

Writes to kaggle_notebook/notebooks/kaggle_submission_goose_t1.ipynb (does NOT
overwrite the original — manual sanity-check first).

Usage:
    python scripts/build_goose_submission_notebook.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "kaggle_notebook" / "notebooks" / "kaggle_submission.ipynb"
DST_NB = ROOT / "kaggle_notebook" / "notebooks" / "kaggle_submission_goose_t1.ipynb"
GOOSE_AGENT = ROOT / "kaggle_notebook" / "agents" / "goose_agent.py"


def main() -> None:
    nb = json.loads(SRC_NB.read_text())
    goose_src = GOOSE_AGENT.read_text()

    # Strip the GooseAgent->MyAgent rename and the my_agent lazy-imports (T3
    # path, not used by T1) so the resulting file is self-contained.
    # T1 only needs bc_policy.PolicyHelper which lives in Cell 5.
    rewritten = goose_src.replace("class GooseAgent(Agent):", "class MyAgent(Agent):")

    # Guard against T3 import (saliency + effect_dict from my_agent) — Cell 4
    # IS my_agent on Kaggle, importing from self would loop. We're shipping
    # T1, not T3, so just disable that branch with a stub.
    rewritten = rewritten.replace(
        "from my_agent import _load_action_effect_dict, _extract_saliency",
        "raise ImportError('T3 path disabled on Kaggle (would self-import)')",
    )

    # ---- Cell 4 (writefile my_agent.py) ----
    cell4 = nb["cells"][4]
    new_cell4_src = "%%writefile /kaggle/working/my_agent.py\n" + rewritten
    cell4["source"] = new_cell4_src.splitlines(keepends=True)
    cell4["outputs"] = []
    cell4["execution_count"] = None

    # ---- Cell 6 (runner) — add ARC_GOOSE_DELTA + ARC_BC_CHECKPOINT_PATH ----
    cell6 = nb["cells"][6]
    src6 = "".join(cell6["source"])
    if "ARC_GOOSE_DELTA" not in src6:
        # Insert env exports right before `python main.py`
        src6 = src6.replace(
            "        MPLBACKEND=agg \\",
            "        MPLBACKEND=agg \\\n"
            "        ARC_GOOSE_DELTA=t1_bc \\\n"
            "        ARC_BC_CHECKPOINT_PATH=/kaggle/working/best.pth \\",
        )
        cell6["source"] = src6.splitlines(keepends=True)
        cell6["outputs"] = []
        cell6["execution_count"] = None

    # ---- Cell 0 (markdown header) — full rewrite for T1 ----
    cell0 = nb["cells"][0]
    cell0_md = (
        "# ARC AGI 3 — Goose-T1 hybrid (online change-reward CNN + bc_v4 as decaying soft prior)\n"
        "\n"
        "**Approach (online learning + pretrained prior):**\n"
        "1. **Small CNN** (16 -> 32 -> 64 -> 128 -> 256) with action head (6 logits) + spatial coord head (64x64 logit map for ACTION6).\n"
        "2. **bc_v4 pretrained checkpoint** loaded at startup; its action_logits, x_logits, y_logits added as a DECAYING soft prior to Goose's logits.\n"
        "3. **Decay**: prior_weight=4.0 at step 0, linearly decays to 0 over 500 steps. After decay, Goose runs as pure online change-reward.\n"
        "4. **Online training every 5 steps** via BCE on (selected_logit, did-frame-change-reward).\n"
        "5. **Reset model + optimizer per level**. bc_v4 prior re-applied fresh each level.\n"
        "\n"
        "**Why this design (caveat below):** local A/B on masked-id full-25 games showed T1 at 0.163 mean_score vs Pure Goose's 0.063 (2.6x lift). After-the-fact holdout test (bc_v4 retrained on 20 games, eval on 5 unseen) showed this lift was largely memorization (T1_holdout=0.016 ~ T0_holdout=0.014). Kaggle hidden games are truly unseen, so bc_v4's prior may add ~no value here.\n"
        "\n"
        "**Realistic Kaggle range:** 0.15-0.25 (vs published Pure Goose 0.25). Submitted as one data point in our 2026-05-11 A/B; the next push (Goose-T2 with GRU memory) is the better-bet generalizer.\n"
        "\n"
        "**Inputs:** competition framework + `jihangli1121/arc-agi-3-replays-v1` Kaggle Dataset (for the bc_v4 best_action.pth checkpoint at /kaggle/working/best.pth).\n"
    )
    cell0["source"] = cell0_md.splitlines(keepends=True)

    DST_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {DST_NB}")
    print(f"  cell[4] my_agent.py: {len(new_cell4_src):,} chars")
    print(f"  cell[5] bc_policy.py: untouched (T1 needs it)")
    print(f"  cell[6] runner: env exports added")


if __name__ == "__main__":
    main()
