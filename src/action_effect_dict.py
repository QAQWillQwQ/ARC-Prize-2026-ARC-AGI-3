"""Phase α.5: action-effect retrieval dictionary built from GT replays.

For each of the ~14,847 replay transitions, we precompute a small visual feature
of the "before" frame and store `(feature_key, action_id, x, y) → effect`. At
eval time, for each candidate action the agent does a kNN lookup and gets a
predicted effect score that biases search toward action-effect patterns the
human demonstrator actually used productively.

This is the "compress 600 MB of replays into a queryable lookup table" idea
from the v3.5 plan (Idea 3). It complements:
- Replay coord_hint_points (Task #20) — known click locations per game
- TTT (Phase α.3) — per-game model adaptation
- GoalDecoder (Phase α.4) — predict the level's goal pattern

It serves as a *prior* (not a target) so:
- Hidden games whose mechanics overlap with public games get retrieved priors
- Hidden games with novel mechanics fall back to the heuristic floor
- No training required — just precompute and load

Architecture:
- **Feature key**: 256-d float32 vector. Computed from a 64x64 grid via:
  1. flatten the grid to a 4096-d color-index vector
  2. compute per-color counts (16 dim)
  3. compute per-row histogram over 16 colors aggregated per-quadrant of grid (16 quadrants × 16 colors = 256 dim)
  This is a cheap, deterministic, color-permutation-AWARE feature. We use
  per-quadrant histograms so the lookup respects spatial structure.
  Dimensionality fits in ~1KB/entry × 14k entries ≈ 14 MB index.
- **Index storage**: a single `.npz` file with the feature_key matrix and
  parallel arrays of (action_id, x, y, frame_delta, levels_after_delta,
  game_id_index). Load via numpy memmap for fast lookup.
- **Lookup**: cosine-similarity kNN (k=5 by default) restricted to entries
  matching the candidate's `(action_id, x, y)` query — so the agent asks
  "what's the typical effect of THIS action from THIS state?"
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .common import iter_jsonl_gz
from .replay_loader import _decode_frame, _normalize_action_id, replay_path, load_replay


GRID_SIZE = 64
NUM_COLORS = 16
QUADRANT_GRID = 4  # 4×4 = 16 quadrants of the 64×64 grid
FEATURE_DIM = QUADRANT_GRID * QUADRANT_GRID * NUM_COLORS  # 256


def compute_feature_key(frame: Sequence[Sequence[int]]) -> np.ndarray:
    """Cheap 256-d feature vector summarising a 64x64 frame.

    Per-quadrant (4x4 split → 16 quadrants), each computes a 16-dim color
    histogram. Concatenated → 256-d. L1-normalised within each quadrant so
    cosine similarity captures spatial-structural similarity.
    """
    arr = np.asarray(frame, dtype=np.int32)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        # Defensive: pad/truncate to 64x64
        out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        h, w = min(arr.shape[0], GRID_SIZE), min(arr.shape[1], GRID_SIZE)
        out[:h, :w] = arr[:h, :w]
        arr = out

    quad_size = GRID_SIZE // QUADRANT_GRID  # 16
    feat = np.zeros((QUADRANT_GRID, QUADRANT_GRID, NUM_COLORS), dtype=np.float32)
    for qy in range(QUADRANT_GRID):
        for qx in range(QUADRANT_GRID):
            patch = arr[qy * quad_size : (qy + 1) * quad_size, qx * quad_size : (qx + 1) * quad_size]
            patch_flat = np.clip(patch.ravel(), 0, NUM_COLORS - 1)
            counts = np.bincount(patch_flat, minlength=NUM_COLORS).astype(np.float32)
            total = counts.sum()
            if total > 0:
                counts /= total
            feat[qy, qx] = counts
    return feat.reshape(FEATURE_DIM)


class ActionEffectDictionary:
    """Lookup table from (frame, action) → predicted effect signal.

    Built from GT replay transitions. At inference time, given the current
    frame and a candidate action, retrieves the k nearest matching transitions
    (filtered by action_id; for ACTION6 also filtered by coord-proximity) and
    returns the mean predicted effect. The agent uses this as a candidate
    score bonus.
    """

    def __init__(
        self,
        feature_keys: np.ndarray,           # (N, 256) float32
        action_ids: np.ndarray,             # (N,) int8
        xs: np.ndarray,                     # (N,) int8
        ys: np.ndarray,                     # (N,) int8
        frame_deltas: np.ndarray,           # (N,) int32 — pixel change magnitude
        level_progresses: np.ndarray,       # (N,) int8 — 1 if level advanced this step
        game_id_idx: np.ndarray,            # (N,) int16 — index into game_ids
        game_ids: List[str],
    ) -> None:
        # Normalize feature_keys for cosine similarity
        norms = np.linalg.norm(feature_keys, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self.feature_keys = (feature_keys / norms).astype(np.float32)
        self.action_ids = action_ids.astype(np.int8)
        self.xs = xs.astype(np.int8)
        self.ys = ys.astype(np.int8)
        self.frame_deltas = frame_deltas.astype(np.int32)
        self.level_progresses = level_progresses.astype(np.int8)
        self.game_id_idx = game_id_idx.astype(np.int16)
        self.game_ids = list(game_ids)

        # Pre-compute a quick action-id index: action_id → boolean mask
        self._action_masks: Dict[int, np.ndarray] = {}
        for aid in range(1, 8):
            self._action_masks[aid] = (self.action_ids == aid)

    def __len__(self) -> int:
        return int(self.feature_keys.shape[0])

    def lookup(
        self,
        frame: Sequence[Sequence[int]],
        action_id: int,
        x: int = 0,
        y: int = 0,
        k: int = 5,
        game_id: Optional[str] = None,
        coord_tolerance: int = 4,
        precomputed_query: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Look up the predicted effect of `(action_id, x, y)` from this frame.

        Returns dict with keys:
            mean_delta:      mean frame_delta among the k nearest matches
            level_prob:      fraction of matches that produced level progress
            n_matches:       how many candidates were available before kNN
            similarity:      mean cosine similarity of top-k

        Returns all-zeros if no matches.

        `precomputed_query` (optional): a normalized 256-dim feature vector
        that callers may supply when ranking many candidates against the same
        frame. Avoids re-running compute_feature_key on every candidate.
        """
        # 1. filter by action_id
        mask = self._action_masks.get(int(action_id))
        if mask is None or not mask.any():
            return {"mean_delta": 0.0, "level_prob": 0.0, "n_matches": 0, "similarity": 0.0}

        # 2. for ACTION6, also restrict to coords within tolerance
        if int(action_id) == 6:
            coord_mask = mask & (
                (np.abs(self.xs.astype(np.int32) - int(x)) <= coord_tolerance)
                & (np.abs(self.ys.astype(np.int32) - int(y)) <= coord_tolerance)
            )
            if coord_mask.any():
                mask = coord_mask

        # 3. optional game-id restriction (use only when game-specific lookup is
        # desired, e.g., when running on a known public game). Not used for
        # hidden games.
        if game_id is not None and game_id in self.game_ids:
            gid_idx = self.game_ids.index(game_id)
            mask_with_gid = mask & (self.game_id_idx == gid_idx)
            if mask_with_gid.sum() >= 3:
                mask = mask_with_gid

        n_matches = int(mask.sum())
        if n_matches == 0:
            return {"mean_delta": 0.0, "level_prob": 0.0, "n_matches": 0, "similarity": 0.0}

        # 4. cosine similarity to all matches
        if precomputed_query is not None:
            query_normed = precomputed_query
        else:
            query_feat = compute_feature_key(frame)
            q_norm = float(np.linalg.norm(query_feat))
            if q_norm < 1e-8:
                return {"mean_delta": 0.0, "level_prob": 0.0, "n_matches": n_matches, "similarity": 0.0}
            query_normed = query_feat / q_norm

        candidate_idxs = np.where(mask)[0]
        sims = self.feature_keys[candidate_idxs] @ query_normed
        if sims.size == 0:
            return {"mean_delta": 0.0, "level_prob": 0.0, "n_matches": n_matches, "similarity": 0.0}

        # Take top-k
        k_actual = min(k, sims.size)
        top_idxs_local = np.argpartition(-sims, k_actual - 1)[:k_actual]
        top_idxs_global = candidate_idxs[top_idxs_local]
        top_sims = sims[top_idxs_local]

        return {
            "mean_delta": float(self.frame_deltas[top_idxs_global].mean()),
            "level_prob": float(self.level_progresses[top_idxs_global].mean()),
            "n_matches": n_matches,
            "similarity": float(top_sims.mean()),
        }

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            feature_keys=self.feature_keys,
            action_ids=self.action_ids,
            xs=self.xs,
            ys=self.ys,
            frame_deltas=self.frame_deltas,
            level_progresses=self.level_progresses,
            game_id_idx=self.game_id_idx,
            game_ids=np.asarray(self.game_ids, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "ActionEffectDictionary":
        z = np.load(path, allow_pickle=True)
        return cls(
            feature_keys=z["feature_keys"],
            action_ids=z["action_ids"],
            xs=z["xs"],
            ys=z["ys"],
            frame_deltas=z["frame_deltas"],
            level_progresses=z["level_progresses"],
            game_id_idx=z["game_id_idx"],
            game_ids=list(z["game_ids"].tolist()),
        )


def build_dictionary_from_replays(project_root: Path) -> ActionEffectDictionary:
    """Walk environment_files/<game_id>/replays/*.json and build the dict."""
    feat_chunks: List[np.ndarray] = []
    action_chunks: List[List[int]] = []
    x_chunks: List[List[int]] = []
    y_chunks: List[List[int]] = []
    delta_chunks: List[List[int]] = []
    progress_chunks: List[List[int]] = []
    gid_chunks: List[List[int]] = []
    game_ids: List[str] = []

    env_root = project_root / "environment_files"
    for game_dir in sorted(env_root.iterdir()):
        if not game_dir.is_dir():
            continue
        game_id = game_dir.name
        path = replay_path(project_root, game_id)
        if path is None:
            continue
        records = load_replay(path)
        if not records:
            continue
        gid_index = len(game_ids)
        game_ids.append(game_id)

        feats: List[np.ndarray] = []
        aids: List[int] = []
        xs_l: List[int] = []
        ys_l: List[int] = []
        deltas: List[int] = []
        progs: List[int] = []
        for i in range(len(records) - 1):
            d_t = records[i].get("data", {})
            d_tp1 = records[i + 1].get("data", {})
            ai = d_t.get("action_input")
            if not isinstance(ai, dict):
                continue
            aid = _normalize_action_id(ai.get("id"))
            if aid is None or not (1 <= aid <= 7):
                continue
            frame_t = _decode_frame(d_t.get("frame"))
            frame_tp1 = _decode_frame(d_tp1.get("frame"))

            ad = ai.get("data") or {}
            try:
                x = int(ad.get("x", 0)) if aid == 6 else 0
                y = int(ad.get("y", 0)) if aid == 6 else 0
            except (TypeError, ValueError):
                continue
            if aid == 6 and not (0 <= x < 64 and 0 <= y < 64):
                continue

            arr_t = np.asarray(frame_t, dtype=np.int32)
            arr_tp1 = np.asarray(frame_tp1, dtype=np.int32)
            if arr_t.shape != arr_tp1.shape:
                continue
            delta_pixels = int((arr_t != arr_tp1).sum())
            level_progress = int(int(d_tp1.get("levels_completed") or 0) > int(d_t.get("levels_completed") or 0))

            feats.append(compute_feature_key(frame_t))
            aids.append(int(aid))
            xs_l.append(int(x))
            ys_l.append(int(y))
            deltas.append(delta_pixels)
            progs.append(level_progress)

        if not feats:
            continue
        feat_chunks.append(np.stack(feats))
        action_chunks.append(aids)
        x_chunks.append(xs_l)
        y_chunks.append(ys_l)
        delta_chunks.append(deltas)
        progress_chunks.append(progs)
        gid_chunks.append([gid_index] * len(aids))

    if not feat_chunks:
        # Empty fallback dictionary
        return ActionEffectDictionary(
            feature_keys=np.zeros((0, FEATURE_DIM), dtype=np.float32),
            action_ids=np.zeros((0,), dtype=np.int8),
            xs=np.zeros((0,), dtype=np.int8),
            ys=np.zeros((0,), dtype=np.int8),
            frame_deltas=np.zeros((0,), dtype=np.int32),
            level_progresses=np.zeros((0,), dtype=np.int8),
            game_id_idx=np.zeros((0,), dtype=np.int16),
            game_ids=[],
        )

    feature_keys = np.concatenate(feat_chunks, axis=0)
    action_ids = np.concatenate([np.asarray(c, dtype=np.int8) for c in action_chunks])
    xs = np.concatenate([np.asarray(c, dtype=np.int8) for c in x_chunks])
    ys = np.concatenate([np.asarray(c, dtype=np.int8) for c in y_chunks])
    deltas = np.concatenate([np.asarray(c, dtype=np.int32) for c in delta_chunks])
    progs = np.concatenate([np.asarray(c, dtype=np.int8) for c in progress_chunks])
    gids = np.concatenate([np.asarray(c, dtype=np.int16) for c in gid_chunks])
    return ActionEffectDictionary(
        feature_keys=feature_keys,
        action_ids=action_ids,
        xs=xs,
        ys=ys,
        frame_deltas=deltas,
        level_progresses=progs,
        game_id_idx=gids,
        game_ids=game_ids,
    )


def main() -> int:
    """CLI: `python -m src.action_effect_dict`. Builds + saves the dict."""
    project_root = Path(__file__).resolve().parent.parent
    out_path = project_root / "Local_Output" / "action_effect_dict.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"building dict from replays under {project_root / 'environment_files'}...")
    dictionary = build_dictionary_from_replays(project_root)
    print(f"built: {len(dictionary)} entries across {len(dictionary.game_ids)} games")
    dictionary.save(out_path)
    print(f"saved to {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
