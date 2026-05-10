"""Inspect a bc_v2 collect .gz: per-game / per-bucket episode stats.

Streams the file once, never keeps transitions in memory. Writes a JSON
summary alongside the input + prints a human-readable report.

Usage:
    python scripts/inspect_bc_v2.py \\
        --gz Local_Output/Collection_Cache/staged_v2_gt/collected/episodes.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

BUCKETS = [
    "gt_verbatim",
    "gt_warmstart_perturbed",
    "gt_warmstart_partial",
    "heuristic_only",
]
STATES = ["WIN", "GAME_OVER", "NOT_FINISHED", "NOT_PLAYED"]


def init_gstats() -> Dict[str, Any]:
    return {
        "n": 0,
        "by_bucket": Counter(),
        "by_state": Counter(),
        "wins": 0,
        "wins_by_bucket": Counter(),
        "max_levels": 0,
        "levels_by_bucket": defaultdict(list),     # bucket -> [levels_completed per ep]
        "actions_by_bucket": defaultdict(list),    # bucket -> [actions_taken per ep]
        "transitions_total": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gz", type=str, required=True, help="Path to episodes.jsonl.gz")
    parser.add_argument("--out", type=str, default=None,
                        help="Where to write the JSON summary. Default: alongside input, named inspect_<stem>.json")
    args = parser.parse_args()

    gz_path = Path(args.gz)
    if not gz_path.exists():
        raise SystemExit(f"input not found: {gz_path}")
    out_path = Path(args.out) if args.out else gz_path.parent / f"inspect_{gz_path.stem.split('.')[0]}.json"

    print(f"[inspect] reading {gz_path} ({gz_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    t0 = time.time()

    games: Dict[str, Dict[str, Any]] = {}
    overall = {
        "n": 0,
        "transitions_total": 0,
        "by_bucket": Counter(),
        "by_state": Counter(),
        "wins": 0,
        "wins_by_bucket": Counter(),
    }
    levels_pct_by_bucket: Dict[str, Counter] = {b: Counter() for b in BUCKETS}

    with gzip.open(gz_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Speed hack: truncate before the giant "transitions" array.
            # collect_gt_warmstart.py writes summary keys first, then "transitions" LAST,
            # so chopping at `,"transitions":` and re-closing the dict gives us a tiny
            # JSON object with just the summary. ~100x faster than parsing the full line.
            idx = line.find(',"transitions":')
            if idx > 0:
                summary_text = line[:idx] + "}"
            else:
                summary_text = line
            try:
                ep = json.loads(summary_text)
            except json.JSONDecodeError:
                # Fallback: try parsing the full line (may be slow but won't drop data)
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
            short_id = ep.get("short_id", "?")
            bucket = ep.get("source", "?")
            state = ep.get("final_state", "?")
            levels = int(ep.get("levels_completed", 0) or 0)
            actions = int(ep.get("actions_taken", 0) or 0)
            transitions_n = int(ep.get("transitions_len", 0) or 0)
            if transitions_n == 0 and "transitions" in ep:
                transitions_n = len(ep["transitions"])
            won = state == "WIN"

            g = games.setdefault(short_id, init_gstats())
            g["n"] += 1
            g["by_bucket"][bucket] += 1
            g["by_state"][state] += 1
            g["max_levels"] = max(g["max_levels"], levels)
            g["levels_by_bucket"][bucket].append(levels)
            g["actions_by_bucket"][bucket].append(actions)
            g["transitions_total"] += transitions_n
            if won:
                g["wins"] += 1
                g["wins_by_bucket"][bucket] += 1

            overall["n"] += 1
            overall["transitions_total"] += transitions_n
            overall["by_bucket"][bucket] += 1
            overall["by_state"][state] += 1
            if won:
                overall["wins"] += 1
                overall["wins_by_bucket"][bucket] += 1
            levels_pct_by_bucket[bucket][levels] += 1

            if overall["n"] % 1000 == 0:
                elapsed = time.time() - t0
                rate = overall["n"] / max(0.001, elapsed)
                print(f"  ...read {overall['n']:,} eps in {elapsed:.0f}s ({rate:.0f}/s)", flush=True)

    elapsed = time.time() - t0
    print(f"[inspect] done. read {overall['n']:,} episodes in {elapsed:.0f}s", flush=True)

    # ---- Build report ----
    report_lines: List[str] = []
    p = report_lines.append

    p("")
    p("=" * 90)
    p(f"OVERALL ({overall['n']:,} episodes, {overall['transitions_total']:,} transitions, {overall['wins']:,} wins)")
    p("=" * 90)
    p("")
    p(f"  total wins:   {overall['wins']:,} / {overall['n']:,} = {100*overall['wins']/max(1,overall['n']):.1f}%")
    p("  by bucket:")
    for b in BUCKETS:
        n = overall["by_bucket"].get(b, 0)
        w = overall["wins_by_bucket"].get(b, 0)
        win_rate = 100 * w / max(1, n)
        p(f"    {b:28s}  n={n:>5,}  wins={w:>5,}  ({win_rate:5.1f}%)")
    p("  by final_state:")
    for s in STATES:
        n = overall["by_state"].get(s, 0)
        if n:
            p(f"    {s:14s}  n={n:>5,}  ({100*n/max(1,overall['n']):5.1f}%)")

    p("")
    p("=" * 90)
    p("LEVELS REACHED — distribution per bucket (count of episodes per max-level)")
    p("=" * 90)
    p(f"  {'bucket':28s}  {'lvl0':>6} {'lvl1':>6} {'lvl2':>6} {'lvl3':>6} {'lvl4':>6} {'lvl5':>6} {'lvl6+':>6}")
    for b in BUCKETS:
        c = levels_pct_by_bucket[b]
        n = sum(c.values())
        if n == 0:
            continue
        cells = []
        for lvl in [0, 1, 2, 3, 4, 5]:
            cnt = c.get(lvl, 0)
            cells.append(f"{cnt:>6}")
        cnt6 = sum(v for k, v in c.items() if k >= 6)
        cells.append(f"{cnt6:>6}")
        p(f"  {b:28s}  {' '.join(cells)}")

    p("")
    p("=" * 90)
    p("PER-GAME BREAKDOWN — sorted by max levels reached")
    p("=" * 90)
    p(f"  {'game':6s}  {'eps':>4}  {'wins':>4}  {'maxL':>4}  {'avgL':>5}  {'gt_v':>4} {'gt_p':>5} {'gt_pa':>6} {'heur':>5}  {'maxL by best bucket':s}")
    p(f"  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*4} {'-'*5} {'-'*6} {'-'*5}  {'-'*40}")

    by_max_levels = sorted(games.items(), key=lambda kv: (-kv[1]["max_levels"], -kv[1]["wins"], kv[0]))
    for short_id, g in by_max_levels:
        n = g["n"]
        wins = g["wins"]
        max_l = g["max_levels"]
        all_levels = [v for vals in g["levels_by_bucket"].values() for v in vals]
        avg_l = mean(all_levels) if all_levels else 0
        gt_v = g["by_bucket"].get("gt_verbatim", 0)
        gt_p = g["by_bucket"].get("gt_warmstart_perturbed", 0)
        gt_pa = g["by_bucket"].get("gt_warmstart_partial", 0)
        heur = g["by_bucket"].get("heuristic_only", 0)
        # which bucket reached the deepest level
        best_bucket_max: List[Tuple[str, int]] = []
        for b in BUCKETS:
            vals = g["levels_by_bucket"].get(b, [])
            if vals:
                best_bucket_max.append((b, max(vals)))
        best_bucket_max.sort(key=lambda x: -x[1])
        best_str = ", ".join(f"{b}=lv{lv}" for b, lv in best_bucket_max[:3])
        p(f"  {short_id:6s}  {n:>4}  {wins:>4}  {max_l:>4}  {avg_l:>5.1f}  {gt_v:>4} {gt_p:>5} {gt_pa:>6} {heur:>5}  {best_str}")

    # ---- "Hot" games where heuristic is reaching levels (suggests learnability) ----
    p("")
    p("=" * 90)
    p("LEARNABILITY HEAT MAP — games where HEURISTIC_ONLY (no GT) reached levels >= 1")
    p("=" * 90)
    p("  (high count here = the game is progress-able from level 0 by the agent alone — best generalization candidates)")
    heur_progress = []
    for short_id, g in games.items():
        heur_levels = g["levels_by_bucket"].get("heuristic_only", [])
        n_progress = sum(1 for lv in heur_levels if lv >= 1)
        n_total = len(heur_levels)
        if n_total > 0:
            heur_progress.append((short_id, n_progress, n_total, max(heur_levels) if heur_levels else 0))
    heur_progress.sort(key=lambda x: -x[1])
    p(f"  {'game':6s}  {'heur_progress':>13}  {'/heur_total':>11}  {'%':>5}  {'maxL_heur':>9}")
    for short_id, n_prog, n_tot, max_l in heur_progress:
        if n_prog > 0:
            p(f"  {short_id:6s}  {n_prog:>13,}  {n_tot:>11,}  {100*n_prog/n_tot:>4.0f}%  {max_l:>9}")

    # ---- Hardest games: zero progress in heuristic_only ----
    hardest = [(sid, n_prog) for sid, n_prog, _, _ in heur_progress if n_prog == 0]
    if hardest:
        p("")
        p(f"  HARDEST (0 heuristic_only progress): {', '.join(sid for sid, _ in hardest)}")

    # ---- Action-budget analysis ----
    p("")
    p("=" * 90)
    p("ACTION BUDGET — average actions_taken per episode by bucket (higher = more exploration time used)")
    p("=" * 90)
    for b in BUCKETS:
        all_actions = []
        for g in games.values():
            all_actions.extend(g["actions_by_bucket"].get(b, []))
        if all_actions:
            avg = mean(all_actions)
            med = median(all_actions)
            mn = min(all_actions)
            mx = max(all_actions)
            p(f"  {b:28s}  n={len(all_actions):>5,}  min={mn:>4}  median={med:>5.0f}  mean={avg:>6.1f}  max={mx:>5}")

    text_report = "\n".join(report_lines)
    print(text_report)

    # ---- JSON summary ----
    summary = {
        "input_path": str(gz_path),
        "input_size_bytes": gz_path.stat().st_size,
        "elapsed_read_s": elapsed,
        "n_episodes": overall["n"],
        "n_transitions": overall["transitions_total"],
        "n_wins": overall["wins"],
        "by_bucket": dict(overall["by_bucket"]),
        "by_state": dict(overall["by_state"]),
        "wins_by_bucket": dict(overall["wins_by_bucket"]),
        "per_game": {
            sid: {
                "n_episodes": g["n"],
                "wins": g["wins"],
                "max_levels": g["max_levels"],
                "by_bucket": dict(g["by_bucket"]),
                "by_state": dict(g["by_state"]),
                "wins_by_bucket": dict(g["wins_by_bucket"]),
                "transitions_total": g["transitions_total"],
                "max_levels_by_bucket": {
                    b: max(vals) if vals else 0
                    for b, vals in g["levels_by_bucket"].items()
                },
                "mean_levels_by_bucket": {
                    b: round(mean(vals), 2) if vals else 0
                    for b, vals in g["levels_by_bucket"].items()
                },
                "mean_actions_by_bucket": {
                    b: round(mean(vals), 1) if vals else 0
                    for b, vals in g["actions_by_bucket"].items()
                },
            }
            for sid, g in games.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[inspect] JSON summary -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
