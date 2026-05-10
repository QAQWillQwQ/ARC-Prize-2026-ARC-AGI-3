"""Convert a bc_v2-style episodes.jsonl.gz into a fast-loading training cache.

Streaming + multiprocess version. Memory-safe (one-episode-at-a-time write
via pickle.dump in a loop) and CPU-parallel (multiprocessing.Pool for the
JSON parse). Builds the bc_v2 corpus (12,425 eps / 4.3 GB / ~37 GB JSON) in
~15-20 min on a 16-core machine vs ~2 hours single-threaded.

Cache format: a CONCATENATED pickle stream — one `pickle.dump(episode, ...)`
call per episode. Loader reads with `while: try pickle.load(...) except EOFError`.
This means episodes are loaded incrementally without ever materializing the
full list in memory, so the cache scales past available RAM.

Per-episode dict layout:
    {
      'game_id', 'short_id', 'source', 'deviation_pct', 'deviation_step',
      'perturb_rate', 'seed', 'strategy', 'score', 'levels_completed',
      'actions_taken', 'final_state', 'resets_used',
      'frames':       np.ndarray (N+1, 64, 64) uint8,   # frame[0] + next_frames
      'transitions':  List[Dict[str, Any]]              # per-transition metadata
    }

Usage:
    python scripts/build_train_cache.py \\
        --in  /tmp/episodes_in.jsonl.gz \\
        --out /tmp/episodes_bc_v2.cache.pkl.gz \\
        --workers 16
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _frame_to_uint8(frame: Any) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        arr = frame.astype(np.uint8, copy=False)
    else:
        try:
            arr = np.asarray(frame, dtype=np.uint8)
        except Exception:
            arr = np.zeros((64, 64), dtype=np.uint8)
    if arr.ndim != 2 or arr.shape != (64, 64):
        out = np.zeros((64, 64), dtype=np.uint8)
        h = min(64, arr.shape[0]) if arr.ndim >= 1 else 0
        w = min(64, arr.shape[1]) if arr.ndim >= 2 else 0
        if h > 0 and w > 0:
            out[:h, :w] = arr[:h, :w]
        arr = out
    return arr & 0x0F


def _convert_one_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one JSON line + convert to compact episode dict.
    Returns None on parse failure or empty episode."""
    line = line.strip()
    if not line:
        return None
    try:
        episode = json.loads(line)
    except json.JSONDecodeError:
        return None
    raw_transitions = list(episode.get("transitions", []))
    if not raw_transitions:
        return None

    n = len(raw_transitions)
    frames = np.zeros((n + 1, 64, 64), dtype=np.uint8)
    frames[0] = _frame_to_uint8(raw_transitions[0]["frame"])

    transitions_out: List[Dict[str, Any]] = []
    for i, t in enumerate(raw_transitions):
        frames[i + 1] = _frame_to_uint8(t["next_frame"])
        ad = t.get("action_data") or {}
        transitions_out.append({
            "available_actions": [int(v) for v in t.get("available_actions", [])],
            "action_id": int(t["action_id"]),
            "action_data": (
                {"x": int(ad.get("x", 0)), "y": int(ad.get("y", 0))}
                if ad else {}
            ),
            "levels_before": int(t.get("levels_before", 0)),
            "levels_after": int(t.get("levels_after", 0)),
            "state_before": str(t.get("state_before", "")),
            "state_after": str(t.get("state_after", "")),
            "delta_pixels": int(t.get("delta_pixels", 0)),
            "novelty": float(t.get("novelty", 0.0)),
            "step_index": int(t.get("step_index", i)),
            "ttt_phase": str(t.get("ttt_phase", "")),
        })

    out: Dict[str, Any] = {
        "frames": frames,
        "transitions": transitions_out,
    }
    for key in (
        "game_id", "short_id", "source", "deviation_pct", "deviation_step",
        "perturb_rate", "seed", "strategy", "score", "levels_completed",
        "actions_taken", "final_state", "resets_used",
    ):
        if key in episode:
            out[key] = episode[key]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=str, required=True)
    parser.add_argument("--out", dest="out_path", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="lines per Pool task. small chunks = better load balance, bigger = less IPC")
    args = parser.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache] reading {in_path} ({in_path.stat().st_size/1e9:.2f} GB)", flush=True)
    print(f"[cache] writing {out_path}  (streaming pickle, {args.workers} workers)", flush=True)

    t0 = time.time()
    n_in = 0
    n_skipped = 0
    n_written = 0
    total_transitions = 0

    # Open output for streaming writes. One pickle.dump call per episode.
    out_f = gzip.open(out_path, "wb")
    try:
        with gzip.open(in_path, "rt") as in_f, mp.Pool(processes=args.workers) as pool:
            # imap_unordered returns episodes as workers finish them. For
            # ordering doesn't matter (loader iterates them all anyway), and
            # unordered minimizes head-of-line blocking on slow lines.
            for episode in pool.imap_unordered(_convert_one_line, in_f, chunksize=args.chunk_size):
                n_in += 1
                if episode is None:
                    n_skipped += 1
                else:
                    pickle.dump(episode, out_f, protocol=pickle.HIGHEST_PROTOCOL)
                    n_written += 1
                    total_transitions += len(episode["transitions"])
                if n_in % args.progress_every == 0:
                    rate = n_in / max(0.001, time.time() - t0)
                    print(
                        f"  ...read {n_in:,} eps written={n_written:,} "
                        f"({total_transitions:,} trans) "
                        f"in {time.time()-t0:.0f}s ({rate:.1f}/s)",
                        flush=True,
                    )
    finally:
        out_f.close()

    elapsed = time.time() - t0
    print(f"[cache] read complete: {n_in:,} eps, {n_skipped} skipped, "
          f"{n_written:,} written, {total_transitions:,} transitions in {elapsed:.0f}s",
          flush=True)
    out_size = out_path.stat().st_size
    print(f"[cache] output size: {out_size/1e9:.2f} GB ({100*out_size/in_path.stat().st_size:.1f}% of input)",
          flush=True)

    # Roundtrip sanity check (incremental load)
    print("[cache] roundtrip sanity check...", flush=True)
    t1 = time.time()
    n_loaded = 0
    first_ep = None
    with gzip.open(out_path, "rb") as f:
        while True:
            try:
                ep = pickle.load(f)
            except EOFError:
                break
            if first_ep is None:
                first_ep = ep
            n_loaded += 1
    print(f"[cache] full incremental load: {n_loaded:,} eps in {time.time()-t1:.1f}s", flush=True)
    if first_ep is not None:
        print(f"  first episode: game_id={first_ep.get('game_id')}, "
              f"transitions={len(first_ep['transitions'])}, "
              f"frames.shape={first_ep['frames'].shape}, "
              f"frames.dtype={first_ep['frames'].dtype}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
