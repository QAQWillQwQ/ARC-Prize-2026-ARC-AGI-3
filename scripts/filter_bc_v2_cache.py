"""Filter a bc_v2 episodes cache down to a smaller, locally-trainable subset.

Why: the full bc_v2 cache (22.85M transitions, ~93 GB frame RAM uncompressed)
cannot fit on a 4070 Super box (24 GB RAM). Most of the volume is
gt_warmstart_partial explore-phase transitions where the agent flailed
without clearing levels — low signal for BC. This script keeps the
high-signal slices and writes a new pickle cache.

Default keep rules (per source bucket):
- gt_verbatim:                keep all (gold standard)
- gt_warmstart_perturbed:     keep all (near-gold with action noise)
- gt_warmstart_partial:       random fraction of episodes (--keep-partial-rate),
                              and TRUNCATE each kept episode to its
                              ttt_phase=='warmstart' prefix only — drops the
                              explore tail where the agent stopped clearing
                              levels.
- heuristic_only:             random fraction of episodes (--keep-heuristic-rate),
                              full transitions retained (preserves some
                              explore-phase diversity).

The truncation step requires re-slicing the episode's `frames` array
((N+1, 64, 64) uint8) to match the kept transition count. We always keep
frames[0:K+1] when keeping transitions[0:K] (frames[i] is the prev-frame
of transition i, frames[i+1] is the next-frame of transition i).

Usage:
    python scripts/filter_bc_v2_cache.py \\
        --in  Local_Output/Collection_Cache/staged_v2_gt/collected/episodes_bc_v2.cache.pkl.gz \\
        --out Local_Output/Collection_Cache/staged_v2_gt/collected/episodes_bc_v2.filtered.cache.pkl.gz \\
        --keep-partial-rate 0.33 \\
        --keep-heuristic-rate 0.10 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np


# Source bucket → default keep behavior. Override via CLI.
SOURCE_KEEP_DEFAULTS = {
    "gt_verbatim": "all",
    "gt_warmstart_perturbed": "all",
    "gt_warmstart_partial": "warmstart_only_sampled",
    "heuristic_only": "sampled",
}


def _iter_episodes(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream episodes from a multi-member pickle gzip cache.

    Mirrors build_train_cache.py's `while: try pickle.load except EOFError`
    loader — keeps memory bounded to one episode at a time.
    """
    with gzip.open(path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError:
                break


def _truncate_to_warmstart_prefix(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Slice an episode to its leading run of ttt_phase=='warmstart' transitions.

    Returns None if the episode has no warmstart prefix (would be empty).
    Frames are sliced consistently: keeping transitions[0:K] → frames[0:K+1].
    """
    transitions = episode.get("transitions") or []
    if not transitions:
        return None
    cut = 0
    for t in transitions:
        if str(t.get("ttt_phase", "")).lower() == "warmstart":
            cut += 1
        else:
            break
    if cut == 0:
        return None
    if cut == len(transitions):
        # Episode was entirely warmstart already (deviation_pct == 100, e.g.
        # gt_verbatim-shaped). Return as-is via slice for consistency.
        return episode
    new_ep = dict(episode)
    new_ep["transitions"] = transitions[:cut]
    new_ep["frames"] = episode["frames"][: cut + 1]
    new_ep["actions_taken"] = cut
    return new_ep


def _decide_keep(
    source: str,
    rules: Dict[str, str],
    rates: Dict[str, float],
    rng: random.Random,
) -> str:
    """Return one of {'drop', 'all', 'warmstart_only', 'sampled', 'warmstart_only_sampled'}.

    'sampled' / 'warmstart_only_sampled' apply the per-source keep_rate
    Bernoulli at episode granularity.
    """
    rule = rules.get(source, SOURCE_KEEP_DEFAULTS.get(source, "drop"))
    if rule == "drop":
        return "drop"
    if rule == "all":
        return "all"
    if rule == "warmstart_only":
        return "warmstart_only"
    if rule == "sampled":
        rate = rates.get(source, 1.0)
        return "sampled" if rng.random() < rate else "drop"
    if rule == "warmstart_only_sampled":
        rate = rates.get(source, 1.0)
        return "warmstart_only" if rng.random() < rate else "drop"
    return "drop"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    parser.add_argument(
        "--keep-partial-rate", type=float, default=0.33,
        help="fraction of gt_warmstart_partial episodes to keep (sampled at episode grain)",
    )
    parser.add_argument(
        "--keep-heuristic-rate", type=float, default=0.10,
        help="fraction of heuristic_only episodes to keep (sampled at episode grain)",
    )
    parser.add_argument(
        "--keep-perturbed-rate", type=float, default=1.0,
        help="fraction of gt_warmstart_perturbed episodes to keep (default 1.0 = all)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report planned counts but don't write the output cache",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    # Compose effective rules. Treat keep-perturbed-rate < 1.0 as 'sampled' too.
    rules = dict(SOURCE_KEEP_DEFAULTS)
    rates = {
        "gt_warmstart_partial": args.keep_partial_rate,
        "heuristic_only": args.keep_heuristic_rate,
    }
    if args.keep_perturbed_rate < 1.0:
        rules["gt_warmstart_perturbed"] = "sampled"
        rates["gt_warmstart_perturbed"] = args.keep_perturbed_rate

    rng = random.Random(args.seed)

    print(f"[filter] in={in_path} ({in_path.stat().st_size/1e9:.2f} GB)", flush=True)
    print(f"[filter] out={out_path}{' (DRY RUN)' if args.dry_run else ''}", flush=True)
    print(f"[filter] rules={rules}", flush=True)
    print(f"[filter] rates={rates}", flush=True)

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = gzip.open(out_path, "wb")
    else:
        out_f = None

    n_in = 0
    n_written = 0
    n_dropped = 0
    transitions_in = 0
    transitions_out = 0
    by_source_in: Counter = Counter()
    by_source_out_eps: Counter = Counter()
    by_source_out_trans: Counter = Counter()
    transitions_per_game_out: Counter = Counter()

    t0 = time.time()
    try:
        for episode in _iter_episodes(in_path):
            n_in += 1
            source = str(episode.get("source", "unknown"))
            n_trans = len(episode.get("transitions") or [])
            transitions_in += n_trans
            by_source_in[source] += 1

            decision = _decide_keep(source, rules, rates, rng)
            if decision == "drop":
                n_dropped += 1
            else:
                if decision == "all":
                    out_ep = episode
                elif decision in ("warmstart_only", "warmstart_only_sampled"):
                    out_ep = _truncate_to_warmstart_prefix(episode)
                else:  # 'sampled' fall-through (keep full)
                    out_ep = episode

                if out_ep is None or not out_ep.get("transitions"):
                    n_dropped += 1
                else:
                    n_kept_trans = len(out_ep["transitions"])
                    transitions_out += n_kept_trans
                    by_source_out_eps[source] += 1
                    by_source_out_trans[source] += n_kept_trans
                    transitions_per_game_out[str(out_ep.get("short_id", out_ep.get("game_id", "?")))] += n_kept_trans
                    n_written += 1
                    if out_f is not None:
                        pickle.dump(out_ep, out_f, protocol=pickle.HIGHEST_PROTOCOL)

            if n_in % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = n_in / max(0.001, elapsed)
                print(
                    f"  ...read {n_in:,}/? eps  written={n_written:,} "
                    f"({transitions_out:,}/{transitions_in:,} trans, "
                    f"{100*transitions_out/max(1,transitions_in):.1f}%) "
                    f"in {elapsed:.0f}s ({rate:.1f}/s)",
                    flush=True,
                )
    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - t0
    print("", flush=True)
    print(f"[filter] DONE  read {n_in:,} eps  wrote {n_written:,} eps  dropped {n_dropped:,}", flush=True)
    print(f"[filter]       transitions: {transitions_in:,} → {transitions_out:,}  "
          f"({100*transitions_out/max(1,transitions_in):.1f}% kept)", flush=True)
    print(f"[filter]       elapsed: {elapsed:.0f}s", flush=True)

    # Per-source breakdown.
    print("[filter] per-source keep:", flush=True)
    for src in sorted(by_source_in):
        in_eps = by_source_in[src]
        out_eps = by_source_out_eps.get(src, 0)
        out_trans = by_source_out_trans.get(src, 0)
        rule = rules.get(src, "drop")
        rate = rates.get(src, 1.0 if rule == "all" else 0.0)
        print(
            f"    {src:30s} rule={rule:25s} keep_rate={rate:.2f}  "
            f"eps={out_eps:>5}/{in_eps:<5} ({100*out_eps/max(1,in_eps):>5.1f}%)  "
            f"trans={out_trans:>10,}",
            flush=True,
        )

    # Per-game balance check.
    if transitions_per_game_out:
        counts = list(transitions_per_game_out.values())
        print(
            f"[filter] per-game transitions: n_games={len(counts)} "
            f"min={min(counts):,} max={max(counts):,} mean={int(np.mean(counts)):,} "
            f"std={int(np.std(counts)):,}",
            flush=True,
        )

    # RAM estimate for downstream training.
    bytes_per_frame = 64 * 64  # uint8
    frames_total = transitions_out + n_written  # frames[0] + N next_frames per episode
    ram_frames_gb = frames_total * bytes_per_frame / 1e9
    ram_metadata_gb = transitions_out * 150 / 1e9  # rough
    print(
        f"[filter] estimated RAM at full-load: "
        f"frames={ram_frames_gb:.2f} GB + metadata={ram_metadata_gb:.2f} GB "
        f"= {ram_frames_gb + ram_metadata_gb:.2f} GB",
        flush=True,
    )

    if not args.dry_run and out_f is not None:
        out_size = out_path.stat().st_size
        print(
            f"[filter] output file: {out_size/1e6:.1f} MB "
            f"({100*out_size/in_path.stat().st_size:.1f}% of input)",
            flush=True,
        )

        # Roundtrip sanity check (incremental load, no full materialization).
        print("[filter] roundtrip sanity check (incremental load)...", flush=True)
        t1 = time.time()
        n_loaded = 0
        first_ep = None
        for ep in _iter_episodes(out_path):
            if first_ep is None:
                first_ep = ep
            n_loaded += 1
        print(f"[filter] loaded {n_loaded:,} eps in {time.time()-t1:.1f}s", flush=True)
        if first_ep is not None:
            print(
                f"  first: game={first_ep.get('game_id')} "
                f"src={first_ep.get('source')} "
                f"transitions={len(first_ep['transitions'])} "
                f"frames.shape={first_ep['frames'].shape}",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
