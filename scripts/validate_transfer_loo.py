"""Leave-one-out validation for transfer-learning matching.

For each of the 25 known games G:
  1. Hide G's prior — pretend it's a hidden game.
  2. Compute G's match against the remaining 24 priors using:
     - Step 1: filter by available_actions Jaccard >= 0.5
     - Step 2: among filtered, rank by first_frame_color_hist cosine
     - Step 3: take top-3, vote on archetype (weighted by similarity)
  3. Compare predicted archetype to G's true archetype.

Pass criteria for shipping transfer learning F:
  - >= 60% archetype accuracy across the 25 LOO trials
  - On games where transfer correctly identifies archetype, the aggregated
    click_hot_spots and repeat_kept_actions should be reasonable matches

Output: per-game predictions + aggregate accuracy, written to
  Local_Output/transfer_loo_validation.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

PRIORS_PATH = ROOT / "Local_Output" / "per_game_priors.json"
OUT_PATH = ROOT / "Local_Output" / "transfer_loo_validation.json"


def jaccard(a: List[Any], b: List[Any]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def cos(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def aggregate_archetype(top3: List[Tuple[str, float, Dict[str, Any]]]) -> str:
    """Weighted vote across top-3."""
    if not top3:
        return "fallback"
    votes: Counter = Counter()
    for _, sim, prior in top3:
        votes[prior.get("archetype", "mixed")] += sim
    if not votes:
        return "fallback"
    return votes.most_common(1)[0][0]


def aggregate_hot_spots(top3: List[Tuple[str, float, Dict[str, Any]]],
                       cap: int = 8) -> List[Dict[str, int]]:
    """Union of top-3's hot_spots, weighted by source-game similarity."""
    accum: Dict[Tuple[int, int], float] = {}
    for _, sim, prior in top3:
        for hp in prior.get("click_hot_spots", []):
            try:
                xy = (int(hp["x"]), int(hp["y"]))
                accum[xy] = accum.get(xy, 0.0) + float(hp.get("count", 1)) * sim
            except Exception:
                continue
    sorted_pts = sorted(accum.items(), key=lambda kv: -kv[1])[:cap]
    return [{"x": x, "y": y, "score": round(s, 3)} for (x, y), s in sorted_pts]


def aggregate_repeat_kept(top3: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, float]:
    """Weighted sum of repeat_kept_actions counts."""
    accum: Dict[str, float] = {}
    for _, sim, prior in top3:
        for k, v in prior.get("repeat_kept_actions", {}).items():
            accum[str(k)] = accum.get(str(k), 0.0) + float(v) * sim
    return {k: round(v, 3) for k, v in sorted(accum.items(), key=lambda kv: -kv[1])}


def loo_predict(held_out_game: str,
                all_priors: Dict[str, Dict[str, Any]],
                jaccard_threshold: float = 0.5,
                sim_threshold: float = 0.5,
                k: int = 3) -> Dict[str, Any]:
    """For one held-out game, predict its archetype + aggregated prior.

    Returns dict with: predicted_archetype, top3, fallback (bool),
    aggregated_hot_spots, aggregated_repeat_kept.
    """
    held = all_priors[held_out_game]
    others = [(g, p) for g, p in all_priors.items() if g != held_out_game]

    held_actions = held.get("available_actions_union", [])
    held_hist = held.get("first_frame_color_hist", [])

    # Step 1: filter by action overlap.
    candidates = [
        (g, p)
        for g, p in others
        if jaccard(held_actions, p.get("available_actions_union", [])) >= jaccard_threshold
    ]
    used_filter = True
    if not candidates:
        # No good filter match — fall back to all 24 (lower-quality matching).
        candidates = list(others)
        used_filter = False

    # Step 2: rank by histogram cosine.
    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    for g, p in candidates:
        s = cos(held_hist, p.get("first_frame_color_hist", []))
        scored.append((g, s, p))
    scored.sort(key=lambda x: -x[1])
    top_k = scored[:k]

    # Step 3: confidence threshold.
    best_sim = top_k[0][1] if top_k else 0.0
    fallback = (not top_k) or (best_sim < sim_threshold)

    if fallback:
        return {
            "predicted_archetype": "fallback",
            "top3": [(g, round(s, 3)) for g, s, _ in top_k],
            "fallback": True,
            "best_sim": round(best_sim, 3),
            "used_action_filter": used_filter,
            "aggregated_hot_spots": [],
            "aggregated_repeat_kept": {},
        }

    return {
        "predicted_archetype": aggregate_archetype(top_k),
        "top3": [(g, round(s, 3)) for g, s, _ in top_k],
        "fallback": False,
        "best_sim": round(best_sim, 3),
        "used_action_filter": used_filter,
        "aggregated_hot_spots": aggregate_hot_spots(top_k),
        "aggregated_repeat_kept": aggregate_repeat_kept(top_k),
    }


def main() -> int:
    if not PRIORS_PATH.exists():
        print(f"missing {PRIORS_PATH} — run scripts/extract_game_priors.py first", file=sys.stderr)
        return 1
    priors = json.loads(PRIORS_PATH.read_text())

    results: Dict[str, Any] = {}
    for game in sorted(priors.keys()):
        truth = priors[game].get("archetype", "mixed")
        pred = loo_predict(game, priors)
        pred["truth"] = truth
        pred["correct"] = (pred["predicted_archetype"] == truth)
        results[game] = pred

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    # Print table
    print(f'{"game":<5} | {"truth":<18} | {"predicted":<18} | {"sim":>5} | {"top3":<60} | {"used_filter":<11} | {"correct":<7}')
    print('-' * 130)
    n_correct = 0
    n_fallback = 0
    n_used_filter = 0
    for game in sorted(results.keys()):
        r = results[game]
        marker = "✓" if r["correct"] else ("FB" if r["fallback"] else "✗")
        print(f'{game:<5} | {r["truth"]:<18} | {r["predicted_archetype"]:<18} | '
              f'{r["best_sim"]:>5} | {str(r["top3"])[:60]:<60} | '
              f'{str(r["used_action_filter"]):<11} | {marker:<7}')
        if r["correct"]:
            n_correct += 1
        if r["fallback"]:
            n_fallback += 1
        if r["used_action_filter"]:
            n_used_filter += 1

    n = len(results)
    print()
    print(f'=== AGGREGATE (n={n}) ===')
    print(f'  archetype accuracy:        {n_correct}/{n} = {100*n_correct/n:.1f}%')
    print(f'  fallback (no good match):  {n_fallback}/{n} = {100*n_fallback/n:.1f}%')
    print(f'  successful action filter:  {n_used_filter}/{n} = {100*n_used_filter/n:.1f}%')

    # Accuracy excluding fallback (which we'd never apply transferred prior on)
    eligible = [g for g in results if not results[g]["fallback"]]
    n_correct_eligible = sum(1 for g in eligible if results[g]["correct"])
    if eligible:
        print(f'  accuracy on non-fallback:  {n_correct_eligible}/{len(eligible)} = {100*n_correct_eligible/len(eligible):.1f}%')

    print()
    print(f'wrote {OUT_PATH}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
