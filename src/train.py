from __future__ import annotations

import argparse
import gc
import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from arc_agi import Arcade, OperationMode
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .agent import PolicyGuidedAgent
from .common import (
    ACTION_IDS,
    ACTION_TO_INDEX,
    ARCHETYPE_LABELS,
    action_mask,
    append_metrics_row,
    compute_discounted_returns,
    compute_saliency_mask,
    episode_level_actions,
    iter_jsonl_gz,
    load_metadata_map,
    load_per_game_archetypes,
    merge_config,
    one_hot_frames,
    pad_history,
    rhae_score,
    safe_mean,
    save_json,
    scalar_features,
    seed_everything,
    split_games,
    transition_reward,
)
from .model import build_model, load_checkpoint, save_checkpoint


class EpisodeTransitionDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[Dict[str, Any]],
        history: int,
        max_steps: int,
        archetype_map: Optional[Dict[str, int]] = None,
        color_permutation_prob: float = 0.0,
        max_transitions_per_episode: Optional[int] = None,
    ) -> None:
        self.episodes: List[Dict[str, Any]] = []
        self.sample_index: List[Tuple[int, int]] = []
        self.history = history
        self.max_steps = max_steps
        self.num_episodes = 0
        self.archetype_map: Dict[str, int] = dict(archetype_map or {})
        # Memory cap: when set, only the LAST N transitions of each episode
        # are kept. The "last N" choice (vs first or stride sampling) is
        # deliberate — gt_warmstart_partial / heuristic_only episodes hit the
        # 2000-step max budget and the late transitions contain most of the
        # progress signal (deeper levels). gt_verbatim / perturbed episodes
        # are short (avg 572 / 348 actions) so this cap doesn't truncate them.
        # Stride sampling was rejected because it breaks the consecutive
        # next_frame relation the latent-prediction loss depends on.
        self.max_transitions_per_episode = max_transitions_per_episode
        # Per-sample probability of applying a random color permutation to all
        # frames in this sample. 0 disables (default). 0.5 = half of training
        # samples see a randomized palette → forces the model to learn structural
        # / positional features rather than per-game color → action mappings.
        # Color 0 (background) is always identity; only colors 1..15 permute.
        self.color_permutation_prob = float(color_permutation_prob)

        for episode in episodes:
            self.add_episode(episode)

    @staticmethod
    def _sample_color_perm() -> torch.Tensor:
        """Random permutation of colors 1..15. Returns a (16,) long tensor mapping old→new."""
        rest = list(range(1, 16))
        random.shuffle(rest)
        perm = [0] + rest
        return torch.tensor(perm, dtype=torch.long)

    @staticmethod
    def _apply_color_perm(frame_tensor: torch.Tensor, perm_table: torch.Tensor) -> torch.Tensor:
        """Apply ``perm_table`` (16,) to a frame tensor of any shape with values in [0, 15]."""
        long = frame_tensor.to(dtype=torch.long).clamp_(0, 15)
        return perm_table[long].to(dtype=frame_tensor.dtype)

    def add_episode(self, episode: Dict[str, Any]) -> bool:
        raw_transitions = list(episode.get("transitions", []))
        if not raw_transitions:
            return False
        # Memory cap (see __init__ doc): trim to last N transitions if set.
        # Discards the corresponding _cached_frames_array slice so add_episode
        # falls back to building the frames tensor from per-transition views.
        cap = self.max_transitions_per_episode
        if cap is not None and len(raw_transitions) > cap:
            raw_transitions = raw_transitions[-cap:]
            # Invalidate the cached frames array (it was for the full episode)
            episode = {k: v for k, v in episode.items() if k != "_cached_frames_array"}
        rewards = [transition_reward(transition) for transition in raw_transitions]
        returns = compute_discounted_returns(rewards)

        frame_list = [raw_transitions[0]["frame"]]
        transitions: List[Dict[str, Any]] = []
        last_action_before: List[Optional[int]] = []
        steps_before: List[int] = []
        last_action_id: Optional[int] = None
        steps_since_progress = 0

        for transition in raw_transitions:
            action_id = int(transition["action_id"])
            progress = int(transition["levels_after"]) - int(transition["levels_before"])
            last_action_before.append(last_action_id)
            steps_before.append(steps_since_progress)
            transitions.append(
                {
                    "available_actions": [int(value) for value in transition["available_actions"]],
                    "action_id": action_id,
                    "action_data": dict(transition.get("action_data") or {}),
                    "levels_before": int(transition["levels_before"]),
                    "levels_after": int(transition["levels_after"]),
                }
            )
            frame_list.append(transition["next_frame"])
            last_action_id = action_id
            if progress > 0:
                steps_since_progress = 0
            else:
                steps_since_progress += 1

        episode_index = len(self.episodes)
        final_state = str(episode.get("final_state", "NOT_FINISHED"))
        levels_completed = int(episode.get("levels_completed", 0))
        episode_score = float(episode.get("score", 0.0))
        if final_state == "WIN":
            weight_mult = 4.0
        elif levels_completed >= 2:
            weight_mult = 2.5
        elif levels_completed >= 1 or episode_score > 0.0:
            weight_mult = 1.5
        else:
            weight_mult = 0.4
        short_id = str(episode.get("game_id", "")).split("-", 1)[0]
        archetype_label = self.archetype_map.get(
            short_id, ARCHETYPE_LABELS["mixed"]
        )
        # Fast path: if the loader yielded a cached numpy frames array on the
        # episode dict, use it directly via torch.from_numpy (zero-copy). Else
        # stack-then-from_numpy avoids the slow torch.tensor(list-of-ndarrays)
        # path that the warning flagged.
        cached_frames = episode.get("_cached_frames_array")
        if cached_frames is not None:
            frames_tensor = torch.from_numpy(cached_frames).to(dtype=torch.uint8)
        else:
            import numpy as _np
            try:
                frames_arr = _np.stack(frame_list, axis=0)
                frames_tensor = torch.from_numpy(frames_arr).to(dtype=torch.uint8)
            except Exception:
                frames_tensor = torch.tensor(frame_list, dtype=torch.uint8)
        self.episodes.append(
            {
                "transitions": transitions,
                "frames": frames_tensor,
                "returns": returns,
                "score": episode_score,
                "last_action_before": last_action_before,
                "steps_before": steps_before,
                "weight_mult": weight_mult,
                "archetype_label": int(archetype_label),
            }
        )
        self.sample_index.extend((episode_index, idx) for idx in range(len(transitions)))
        self.num_episodes += 1
        return True

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        episode_index, transition_index = self.sample_index[index]
        episode = self.episodes[episode_index]
        transitions = episode["transitions"]
        transition = transitions[transition_index]
        next_transition = transitions[transition_index + 1] if transition_index + 1 < len(transitions) else transition
        frames = episode["frames"]

        history_frames = self._history_frames(frames, transition_index)
        next_history_frames = self._history_frames(frames, transition_index + 1)
        progress = int(transition["levels_after"]) - int(transition["levels_before"])
        next_steps_since_progress = 0 if progress > 0 else int(episode["steps_before"][transition_index]) + 1
        action_id = int(transition["action_id"])

        current_frame = history_frames[-1]
        next_frame = next_history_frames[-1]
        # Saliency target = "where the agent should look NOW" (mask of current frame).
        # Computed BEFORE color-permutation since the algorithm is invariant to
        # color labels (it counts pixels-per-color and finds rare-color centroids;
        # permuting labels reshuffles which int corresponds to which set of
        # pixels, but the SET of rare-pixel positions — hence the centroid mask —
        # is unchanged). Verified: rare-colors test by count, not by id.
        saliency_target = compute_saliency_mask(current_frame.tolist())

        # Color-permutation augmentation. Apply the SAME perm to obs, next_obs,
        # and next_frame_target so the BC objective is consistent under the
        # permutation. Probability gate happens once per sample.
        if self.color_permutation_prob > 0.0 and random.random() < self.color_permutation_prob:
            perm_table = self._sample_color_perm()
            history_frames = self._apply_color_perm(history_frames, perm_table)
            next_history_frames = self._apply_color_perm(next_history_frames, perm_table)
            next_frame = self._apply_color_perm(next_frame, perm_table)

        # Next-frame recon target = the frame after the current action, as long color indices.
        next_frame_target = next_frame.to(dtype=torch.long).clamp(0, 15)

        return {
            "obs": self._encode_history(history_frames),
            "next_obs": self._encode_history(next_history_frames),
            "scalar": scalar_features(
                available_actions=transition["available_actions"],
                last_action_id=episode["last_action_before"][transition_index],
                levels_completed=int(transition["levels_before"]),
                steps_since_progress=int(episode["steps_before"][transition_index]),
                step_index=transition_index,
                frame=history_frames[-1],
                max_steps=self.max_steps,
            ),
            "next_scalar": scalar_features(
                available_actions=next_transition["available_actions"],
                last_action_id=action_id,
                levels_completed=int(transition["levels_after"]),
                steps_since_progress=next_steps_since_progress,
                step_index=transition_index + 1,
                frame=next_history_frames[-1],
                max_steps=self.max_steps,
            ),
            "available_mask": torch.tensor(action_mask(transition["available_actions"]), dtype=torch.float32),
            "action_index": torch.tensor(ACTION_TO_INDEX[action_id], dtype=torch.long),
            "x": torch.tensor(int((transition.get("action_data") or {}).get("x", 0)), dtype=torch.long),
            "y": torch.tensor(int((transition.get("action_data") or {}).get("y", 0)), dtype=torch.long),
            "coord_mask": torch.tensor(1.0 if action_id == 6 else 0.0, dtype=torch.float32),
            "return_target": torch.tensor(float(episode["returns"][transition_index]), dtype=torch.float32),
            "archetype_label": torch.tensor(int(episode["archetype_label"]), dtype=torch.long),
            "saliency_target": saliency_target,
            "next_frame_target": next_frame_target,
            "weight": torch.tensor(
                float(episode["weight_mult"])
                * (
                    1.0
                    + (float(episode["score"]) / 100.0)
                    + max(0, progress) * 2.0
                ),
                dtype=torch.float32,
            ),
        }

    def _history_frames(self, frames: torch.Tensor, end_index: int) -> torch.Tensor:
        start_index = max(0, end_index - self.history + 1)
        history_frames = frames[start_index : end_index + 1]
        if history_frames.shape[0] < self.history:
            pad = history_frames[0:1].repeat(self.history - history_frames.shape[0], 1, 1)
            history_frames = torch.cat([pad, history_frames], dim=0)
        return history_frames

    @staticmethod
    def _encode_history(history_frames: torch.Tensor) -> torch.Tensor:
        clipped = history_frames.to(dtype=torch.long).clamp_(0, 15)
        encoded = F.one_hot(clipped, num_classes=16).permute(0, 3, 1, 2)
        return encoded.reshape(-1, encoded.shape[-2], encoded.shape[-1]).to(dtype=torch.float32)


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal ARC-AGI-3 object-centric policy.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to one collected episodes.jsonl.gz file, or multiple comma separated paths.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hardware-profile", type=str, default="a100")
    parser.add_argument("--games", type=str, default=None, help="Optional comma separated game ids to train/evaluate on, e.g. sp80 or sp80,ar25.")
    parser.add_argument("--split-mode", type=str, default="game", choices=["game", "episode"])
    parser.add_argument("--episode-val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--history", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--max-transitions-per-episode", type=int, default=None,
                        help="Cap transitions per episode (keeps the LAST N, where progress happens). "
                             "Default None = use all. Recommended ~1000 for bc_v2 corpus on 80 GB RAM "
                             "(memory ~ 4 KB * (N+1) * num_episodes).")
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=4)
    parser.add_argument("--online-val-every", type=int, default=None)
    parser.add_argument("--online-val-games", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--data-workers", type=int, default=None)
    parser.add_argument("--aux-archetype-weight", type=float, default=None,
                        help="Loss weight for archetype CE head (Plan A). 0 disables. Default: 0.15.")
    parser.add_argument("--aux-saliency-weight", type=float, default=None,
                        help="Loss weight for saliency BCE head (Plan A). 0 disables. Default: 0.10.")
    parser.add_argument("--aux-recon-weight", type=float, default=None,
                        help="Loss weight for next-frame recon CE head (Plan A). 0 disables. Default: 0.10.")
    parser.add_argument("--per-game-priors-path", type=str, default=None,
                        help="Path to per_game_priors.json for archetype labels. "
                             "Defaults to <project-root>/Local_Output/per_game_priors.json.")
    parser.add_argument("--color-permutation-prob", type=float, default=None,
                        help="Per-sample probability of randomly permuting colors 1..15 "
                             "(background fixed). Forces the model to learn color-invariant "
                             "features. 0 disables. Default: 0.5 — proxy for hidden-game palette shift.")
    return parser.parse_args()


def parse_data_paths(raw: str) -> List[Path]:
    paths = [Path(part.strip()).expanduser().resolve() for part in raw.split(",") if part.strip()]
    if not paths:
        raise RuntimeError("No valid --data paths were provided.")
    return paths


def parse_game_filter(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    games = sorted(set(part.strip() for part in raw.split(",") if part.strip()))
    return games or None


def episode_split_key(episode: Dict[str, Any]) -> str:
    for key in ("episode_id", "source_guid"):
        value = episode.get(key)
        if value:
            return str(value)
    # `transitions_len` (summary field from collect_gt_warmstart) lets us
    # build the fallback key from a metadata-only episode dict (no need to
    # parse the giant transitions array). Falls back to len(transitions) if
    # the summary field isn't present.
    trans_len = episode.get("transitions_len")
    if trans_len is None:
        trans_len = len(episode.get("transitions", []))
    fallback = "|".join(
        [
            str(episode.get("game_id", "")),
            str(episode.get("seed", "")),
            str(episode.get("actions_taken", "")),
            str(trans_len),
        ]
    )
    return hashlib.sha1(fallback.encode("utf-8")).hexdigest()


def split_episode_keys(keys: Sequence[str], holdout_fraction: float = 0.2) -> Tuple[List[str], List[str]]:
    unique_keys = sorted(set(str(key) for key in keys if str(key)))
    if not unique_keys:
        return [], []
    if len(unique_keys) == 1:
        return list(unique_keys), []

    ranked = []
    for key in unique_keys:
        digest = hashlib.sha1(("arcagi3-episode-split-" + key).encode("utf-8")).hexdigest()
        ranked.append((digest, key))
    ranked.sort()
    holdout = max(1, int(round(len(unique_keys) * holdout_fraction)))
    val_keys = sorted(key for _, key in ranked[:holdout])
    train_keys = sorted(key for _, key in ranked[holdout:])
    if not train_keys:
        train_keys = val_keys[1:] or val_keys
        val_keys = val_keys[:1]
    return train_keys, val_keys


def _is_cache_path(path: Path) -> bool:
    """Cache files end in .pkl.gz; raw episode .gz files end in .jsonl.gz."""
    name = path.name.lower()
    return name.endswith(".pkl.gz") or name.endswith(".cache.pkl.gz") or name.endswith(".pkl")


def _matches_allowed(game_id: str, allowed: set) -> bool:
    """Filter helper: episode's game_id is allowed if either the full id or
    its short prefix (before the first '-') is in the allowed set. Lets users
    pass --games sp80 and have it match game_id='sp80-589a99af'.
    """
    if not allowed:
        return True
    if game_id in allowed:
        return True
    short = game_id.split("-", 1)[0]
    return short in allowed


def _iter_pkl_cache(path: Path, allowed: set) -> Iterable[Dict[str, Any]]:
    """Yield episodes from a pre-built pickle cache (scripts/build_train_cache.py).

    Cache format: a CONCATENATED pickle stream — one `pickle.dump(episode, ...)`
    call per episode. Read with `while: try pickle.load except EOFError: break`.
    Loads incrementally without materializing the full list in memory, so
    cache size can exceed available RAM.

    Each cached episode dict has summary fields plus
    `frames: np.ndarray (N+1, 64, 64) uint8` and `transitions: List[Dict]`.
    On yield, the compact `frames` array is expanded back into per-transition
    `frame` / `next_frame` keys (zero-copy numpy views) so the rest of the
    pipeline (add_episode, transition_reward, etc.) sees the same shape it
    would from a raw .jsonl.gz parse.

    Falls back to legacy single-pickle-list format if pickle.load returns a
    list on the first call (covers caches built by the v1 build script).
    """
    import gzip as _gz
    import pickle as _pkl
    open_fn = (lambda p: _gz.open(p, "rb")) if path.name.lower().endswith(".gz") else (lambda p: open(p, "rb"))

    def _expand(ep: Dict[str, Any]) -> Dict[str, Any]:
        frames = ep.get("frames")
        if frames is None:
            return ep
        out_episode = {k: v for k, v in ep.items() if k != "frames"}
        # Pass the full (N+1, 64, 64) array as a single ndarray under a
        # private key so add_episode can torch.from_numpy() it directly
        # (zero-copy) instead of rebuilding the tensor from a list of views.
        out_episode["_cached_frames_array"] = frames
        new_transitions = []
        for i, t in enumerate(ep.get("transitions", [])):
            nt = dict(t)
            nt["frame"] = frames[i]
            nt["next_frame"] = frames[i + 1]
            new_transitions.append(nt)
        out_episode["transitions"] = new_transitions
        return out_episode

    with open_fn(path) as f:
        # Peek at first object — if it's a list, we have a legacy v1 cache.
        try:
            first = _pkl.load(f)
        except EOFError:
            return
        if isinstance(first, list):
            # Legacy: single big list of episodes
            for ep in first:
                game_id = str(ep.get("game_id", ""))
                if not _matches_allowed(game_id, allowed):
                    continue
                yield _expand(ep)
            return
        # Streaming: first object was already an episode
        ep = first
        while True:
            game_id = str(ep.get("game_id", ""))
            if _matches_allowed(game_id, allowed):
                yield _expand(ep)
            try:
                ep = _pkl.load(f)
            except EOFError:
                break


def iter_episodes_from_paths(paths: Sequence[Path | str], allowed_games: Optional[Sequence[str]] = None) -> Iterable[Dict[str, Any]]:
    allowed = set(str(game_id) for game_id in (allowed_games or []))
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("Collected data not found: %s" % path)
        if _is_cache_path(path):
            for episode in _iter_pkl_cache(path, allowed):
                yield episode
            continue
        for episode in iter_jsonl_gz(path):
            game_id = str(episode.get("game_id", ""))
            if not _matches_allowed(game_id, allowed):
                continue
            yield episode


def iter_episode_summaries_from_paths(
    paths: Sequence[Path | str],
    allowed_games: Optional[Sequence[str]] = None,
) -> Iterable[Dict[str, Any]]:
    """Fast metadata-only episode iterator for discovery / splitting.

    For .pkl.gz cache files: full pickle.load (cheap, ~30s for bc_v2 corpus),
    yields each episode with its summary fields.

    For .jsonl.gz raw files: skips parsing the giant `transitions` array.
    Slices the first 1024 chars of each line, finds `, "transitions":`,
    truncates, re-closes the dict — gives us a tiny JSON object with only the
    metadata. ~3x faster than full json.loads on bc_v2-scale corpora.
    """
    import gzip
    import json as _json
    allowed = set(str(game_id) for game_id in (allowed_games or []))
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("Collected data not found: %s" % path)
        if _is_cache_path(path):
            for ep in _iter_pkl_cache(path, allowed):
                yield ep
            continue
        with gzip.open(path, "rt") as f:
            for raw_line in f:
                head = raw_line[:1024]
                idx = head.find(', "transitions":')
                if idx > 0:
                    summary_text = head[:idx] + "}"
                else:
                    summary_text = raw_line.strip()
                try:
                    ep = _json.loads(summary_text)
                except _json.JSONDecodeError:
                    continue
                game_id = str(ep.get("game_id", ""))
                if not _matches_allowed(game_id, allowed):
                    continue
                yield ep


def split_episodes(episodes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    game_ids = [str(episode["game_id"]) for episode in episodes]
    train_games, val_games = split_games(game_ids)
    train_episodes = [episode for episode in episodes if episode["game_id"] in train_games]
    val_episodes = [episode for episode in episodes if episode["game_id"] in val_games]
    return train_episodes, val_episodes, train_games, val_games


def discover_game_split(
    paths: Sequence[Path | str],
    allowed_games: Optional[Sequence[str]] = None,
    progress_every: int = 25,
) -> Tuple[List[str], List[str], int]:
    print(
        "[data] discovering train/val game split from %d data file(s)"
        % len(paths),
        flush=True,
    )
    game_ids: List[str] = []
    unique_games = set()
    episode_count = 0
    for episode in iter_episode_summaries_from_paths(paths, allowed_games=allowed_games):
        game_id = str(episode["game_id"])
        game_ids.append(game_id)
        unique_games.add(game_id)
        episode_count += 1
        if progress_every > 0 and episode_count % progress_every == 0:
            print(
                "[data] split pass episodes=%d unique_games=%d"
                % (episode_count, len(unique_games)),
                flush=True,
            )
    train_games, val_games = split_games(game_ids)
    print(
        "[data] split pass complete episodes=%d unique_games=%d train_games=%d val_games=%d"
        % (episode_count, len(unique_games), len(train_games), len(val_games)),
        flush=True,
    )
    return train_games, val_games, episode_count


def discover_episode_split(
    paths: Sequence[Path | str],
    allowed_games: Optional[Sequence[str]] = None,
    holdout_fraction: float = 0.2,
    progress_every: int = 25,
) -> Tuple[List[str], List[str], set[str], set[str], int]:
    print(
        "[data] discovering train/val episode split from %d data file(s)"
        % len(paths),
        flush=True,
    )
    episode_keys: List[str] = []
    unique_games = set()
    episode_count = 0
    for episode in iter_episode_summaries_from_paths(paths, allowed_games=allowed_games):
        unique_games.add(str(episode["game_id"]))
        episode_keys.append(episode_split_key(episode))
        episode_count += 1
        if progress_every > 0 and episode_count % progress_every == 0:
            print(
                "[data] split pass episodes=%d unique_games=%d"
                % (episode_count, len(unique_games)),
                flush=True,
            )
    train_episode_keys, val_episode_keys = split_episode_keys(
        episode_keys,
        holdout_fraction=holdout_fraction,
    )
    train_games = sorted(unique_games)
    val_games = sorted(unique_games)
    print(
        "[data] split pass complete episodes=%d unique_games=%d train_games=%d val_games=%d train_episodes=%d val_episodes=%d"
        % (
            episode_count,
            len(unique_games),
            len(train_games),
            len(val_games),
            len(train_episode_keys),
            len(val_episode_keys),
        ),
        flush=True,
    )
    return train_games, val_games, set(train_episode_keys), set(val_episode_keys), episode_count


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_average(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def save_training_checkpoint(
    checkpoints_dir: Path,
    stem: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict[str, Any],
    epoch: int,
    best_score: float,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    primary_path = checkpoints_dir / ("%s.pth" % stem)
    legacy_path = checkpoints_dir / ("%s.pt" % stem)
    save_checkpoint(
        path=str(primary_path),
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=epoch,
        best_score=best_score,
        extra_state=extra_state,
    )
    shutil.copyfile(primary_path, legacy_path)
    return primary_path


def restore_training_state(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    payload = load_checkpoint(str(resume_path), device=device)
    # strict=False so pre-aux-head checkpoints can seed a new run with the new heads.
    model.load_state_dict(payload["model_state"], strict=False)
    optimizer_state = payload.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    last_epoch = int(payload.get("epoch", 0))
    epoch_complete = bool(payload.get("epoch_complete", True))
    best_score = float(payload.get("best_score", float("-inf")))
    best_epoch = int(payload.get("best_epoch", last_epoch if epoch_complete else max(last_epoch - 1, -1)))
    start_epoch = last_epoch + 1 if epoch_complete else max(1, last_epoch)
    return {
        "start_epoch": start_epoch,
        "best_metric": best_score,
        "best_epoch": best_epoch,
        "epoch_complete": epoch_complete,
        "payload": payload,
    }


def evaluate_public_score(
    checkpoint_path: str,
    project_root: Path,
    metadata_map: Dict[str, Dict[str, Any]],
    selected_games: Sequence[str],
    max_steps: int,
    stall_steps: int,
    reset_limit: int,
) -> float:
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(project_root / "tmp_eval_recordings"),
    )
    agent = PolicyGuidedAgent(
        checkpoint_path=checkpoint_path,
        max_steps=max_steps,
        stall_steps=stall_steps,
        reset_limit=reset_limit,
    )
    scores: List[float] = []
    for game_id in selected_games:
        print("[online-val] loading game=%s" % game_id, flush=True)
        env = arc.make(game_id)
        if env is None:
            print("[online-val] skipped game=%s env_unavailable" % game_id, flush=True)
            continue
        baseline_actions = metadata_map[game_id].get("baseline_actions", [])
        result = agent.play_env(env=env, game_id=game_id, baseline_actions=baseline_actions)
        score_info = rhae_score(
            baseline_actions=baseline_actions,
            completed_level_actions=episode_level_actions(result["transitions"]),
        )
        score = float(score_info["score"])
        scores.append(score)
        print("[online-val] game=%s score=%.6f" % (game_id, score), flush=True)
    return safe_mean(scores)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_accum: int,
    max_grad_norm: float = 0.0,
    epoch: int = 0,
    phase: str = "train",
    checkpoint_every_steps: int = 0,
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_every_batches: int = 0,
    aux_archetype_weight: float = 0.0,
    aux_saliency_weight: float = 0.0,
    aux_recon_weight: float = 0.0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_action_loss = 0.0
    total_coord_loss = 0.0
    total_value_loss = 0.0
    total_avail_loss = 0.0
    total_latent_loss = 0.0
    total_archetype_loss = 0.0
    total_saliency_loss = 0.0
    total_recon_loss = 0.0
    total_action_correct = 0.0
    total_action_count = 0.0
    total_coord_correct = 0.0
    total_coord_count = 0.0
    total_archetype_correct = 0.0
    total_archetype_count = 0.0
    optimizer_steps = 0
    seen_samples = 0

    autocast_enabled = device.type == "cuda"
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch_size = int(batch["obs"].shape[0])
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(
                obs=batch["obs"],
                scalar=batch["scalar"],
                action_index=batch["action_index"],
            )
            with torch.no_grad():
                next_latent = model.encode_state(batch["next_obs"], batch["next_scalar"])["pooled"]

            action_loss_per = F.cross_entropy(
                out["action_logits"],
                batch["action_index"],
                reduction="none",
            )
            coord_loss_x = F.cross_entropy(out["x_logits"], batch["x"], reduction="none")
            coord_loss_y = F.cross_entropy(out["y_logits"], batch["y"], reduction="none")
            coord_loss_per = (coord_loss_x + coord_loss_y) * batch["coord_mask"]
            value_loss_per = F.smooth_l1_loss(out["value"], batch["return_target"], reduction="none")
            avail_loss_per = F.binary_cross_entropy_with_logits(
                out["avail_logits"],
                batch["available_mask"],
                reduction="none",
            ).mean(dim=-1)
            latent_loss_per = F.mse_loss(
                out["pred_next_latent"],
                next_latent.detach(),
                reduction="none",
            ).mean(dim=-1)

            weights = batch["weight"]
            action_loss = masked_average(action_loss_per, weights)
            coord_loss = 0.5 * masked_average(coord_loss_per, weights)
            value_loss = 0.25 * masked_average(value_loss_per, weights)
            avail_loss = 0.2 * masked_average(avail_loss_per, weights)
            latent_loss = 0.1 * masked_average(latent_loss_per, weights)

            # Option D aux losses (Plan A heads): archetype CE + saliency BCE + next-frame recon CE.
            archetype_loss_per = F.cross_entropy(
                out["archetype_logits"],
                batch["archetype_label"],
                reduction="none",
            )
            saliency_loss_per = F.binary_cross_entropy_with_logits(
                out["saliency_logits"],
                batch["saliency_target"],
                reduction="none",
            ).mean(dim=(-1, -2))
            recon_loss_per = F.cross_entropy(
                out["next_frame_recon_logits"],
                batch["next_frame_target"],
                reduction="none",
            ).mean(dim=(-1, -2))
            archetype_loss = aux_archetype_weight * masked_average(archetype_loss_per, weights)
            saliency_loss = aux_saliency_weight * masked_average(saliency_loss_per, weights)
            recon_loss = aux_recon_weight * masked_average(recon_loss_per, weights)

            loss = (
                action_loss
                + coord_loss
                + value_loss
                + avail_loss
                + latent_loss
                + archetype_loss
                + saliency_loss
                + recon_loss
            )

        if optimizer is not None:
            (loss / float(max(1, grad_accum))).backward()
            if (step + 1) % max(1, grad_accum) == 0 or (step + 1) == len(loader):
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if checkpoint_callback is not None and checkpoint_every_steps > 0:
                    if optimizer_steps % checkpoint_every_steps == 0:
                        checkpoint_callback(
                            {
                                "epoch": epoch,
                                "epoch_complete": False,
                                "batch_in_epoch": step + 1,
                                "optimizer_steps_in_epoch": optimizer_steps,
                                "last_loss": float(loss.item()),
                            }
                        )

        seen_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_action_loss += float(action_loss.item()) * batch_size
        total_coord_loss += float(coord_loss.item()) * batch_size
        total_value_loss += float(value_loss.item()) * batch_size
        total_avail_loss += float(avail_loss.item()) * batch_size
        total_latent_loss += float(latent_loss.item()) * batch_size
        total_archetype_loss += float(archetype_loss.item()) * batch_size
        total_saliency_loss += float(saliency_loss.item()) * batch_size
        total_recon_loss += float(recon_loss.item()) * batch_size
        action_pred = out["action_logits"].argmax(dim=-1)
        total_action_correct += float((action_pred == batch["action_index"]).sum().item())
        total_action_count += float(batch["action_index"].numel())
        archetype_pred = out["archetype_logits"].argmax(dim=-1)
        total_archetype_correct += float((archetype_pred == batch["archetype_label"]).sum().item())
        total_archetype_count += float(batch["archetype_label"].numel())

        coord_mask = batch["coord_mask"] > 0
        if coord_mask.any():
            x_correct = out["x_logits"].argmax(dim=-1)[coord_mask] == batch["x"][coord_mask]
            y_correct = out["y_logits"].argmax(dim=-1)[coord_mask] == batch["y"][coord_mask]
            total_coord_correct += float((x_correct & y_correct).sum().item())
            total_coord_count += float(coord_mask.sum().item())

        if log_every_batches > 0 and (((step + 1) % log_every_batches == 0) or (step + 1 == len(loader))):
            running_loss = total_loss / max(seen_samples, 1)
            running_action_acc = total_action_correct / max(total_action_count, 1.0)
            running_coord_acc = total_coord_correct / max(total_coord_count, 1.0)
            print(
                "[%s] epoch=%d batch=%d/%d samples=%d/%d loss=%.4f action_acc=%.4f coord_acc=%.4f"
                % (
                    phase,
                    epoch,
                    step + 1,
                    len(loader),
                    seen_samples,
                    len(loader.dataset),
                    running_loss,
                    running_action_acc,
                    running_coord_acc,
                ),
                flush=True,
            )

    denom = max(len(loader.dataset), 1)
    return {
        "loss": total_loss / denom,
        "action_loss": total_action_loss / denom,
        "coord_loss": total_coord_loss / denom,
        "value_loss": total_value_loss / denom,
        "avail_loss": total_avail_loss / denom,
        "latent_loss": total_latent_loss / denom,
        "archetype_loss": total_archetype_loss / denom,
        "saliency_loss": total_saliency_loss / denom,
        "recon_loss": total_recon_loss / denom,
        "action_acc": total_action_correct / max(total_action_count, 1.0),
        "coord_acc": total_coord_correct / max(total_coord_count, 1.0),
        "archetype_acc": total_archetype_correct / max(total_archetype_count, 1.0),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "history": args.history,
        "model_dim": args.model_dim,
        "num_slots": args.num_slots,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "online_val_every": args.online_val_every,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "max_grad_norm": args.max_grad_norm,
        "log_every_batches": args.log_every_batches,
        "max_steps": args.max_steps,
        "stall_steps": args.stall_steps,
        "reset_limit": args.reset_limit,
        "data_workers": args.data_workers,
        "aux_archetype_weight": args.aux_archetype_weight,
        "aux_saliency_weight": args.aux_saliency_weight,
        "aux_recon_weight": args.aux_recon_weight,
        "color_permutation_prob": args.color_permutation_prob,
    }
    config = merge_config(args.hardware_profile, overrides)
    config["hardware_profile"] = args.hardware_profile
    config["seed"] = args.seed
    # Plan-D aux loss + augmentation defaults (apply only when not set by profile or CLI override).
    # Hidden-LB rationale: aux losses regularize encoder; color permutation forces palette
    # invariance so train→hidden-game generalization isn't bottlenecked by per-game color cues.
    config.setdefault("aux_archetype_weight", 0.15)
    config.setdefault("aux_saliency_weight", 0.10)
    config.setdefault("aux_recon_weight", 0.10)
    config.setdefault("color_permutation_prob", 0.5)
    data_paths = parse_data_paths(args.data)
    requested_games = parse_game_filter(args.games)
    config["data"] = [str(path) for path in data_paths]
    config["games"] = requested_games
    config["split_mode"] = args.split_mode
    config["episode_val_fraction"] = args.episode_val_fraction
    save_json(output_dir / "train_config.json", config)

    train_episode_key_set: set[str] = set()
    val_episode_key_set: set[str] = set()
    if args.split_mode == "episode":
        train_games, val_games, train_episode_key_set, val_episode_key_set, discovered_episodes = discover_episode_split(
            data_paths,
            allowed_games=requested_games,
            holdout_fraction=float(args.episode_val_fraction),
        )
    else:
        train_games, val_games, discovered_episodes = discover_game_split(
            data_paths,
            allowed_games=requested_games,
        )
    train_game_set = set(train_games)
    val_game_set = set(val_games)
    print(
        "[data] building train/val datasets from %d discovered episode(s)"
        % discovered_episodes,
        flush=True,
    )
    archetype_path = (
        Path(args.per_game_priors_path).resolve()
        if args.per_game_priors_path
        else (project_root / "Local_Output" / "per_game_priors.json")
    )
    archetype_map = load_per_game_archetypes(archetype_path)
    print(
        "[data] archetype labels loaded: %d games from %s"
        % (len(archetype_map), archetype_path),
        flush=True,
    )
    config["per_game_priors_path"] = str(archetype_path)
    train_dataset = EpisodeTransitionDataset(
        episodes=[],
        history=int(config["history"]),
        max_steps=int(config["max_steps"]),
        archetype_map=archetype_map,
        color_permutation_prob=float(config.get("color_permutation_prob", 0.0)),
        max_transitions_per_episode=args.max_transitions_per_episode,
    )
    # Val dataset uses the natural color distribution so val metrics are
    # comparable across runs — augmentation is a train-time-only regularizer.
    val_dataset = EpisodeTransitionDataset(
        episodes=[],
        history=int(config["history"]),
        max_steps=int(config["max_steps"]),
        archetype_map=archetype_map,
        color_permutation_prob=0.0,
        max_transitions_per_episode=args.max_transitions_per_episode,
    )
    loaded_episodes = 0
    for episode in iter_episodes_from_paths(data_paths, allowed_games=requested_games):
        game_id = str(episode["game_id"])
        if args.split_mode == "episode":
            split_key = episode_split_key(episode)
            if split_key in train_episode_key_set:
                train_dataset.add_episode(episode)
            elif split_key in val_episode_key_set:
                val_dataset.add_episode(episode)
        else:
            if game_id in train_game_set:
                train_dataset.add_episode(episode)
            elif game_id in val_game_set:
                val_dataset.add_episode(episode)
        loaded_episodes += 1
        if loaded_episodes % 25 == 0:
            print(
                "[data] build pass episodes=%d/%d train_episodes=%d val_episodes=%d train_samples=%d val_samples=%d"
                % (
                    loaded_episodes,
                    discovered_episodes,
                    train_dataset.num_episodes,
                    val_dataset.num_episodes,
                    len(train_dataset),
                    len(val_dataset),
                ),
                flush=True,
            )
    if train_dataset.num_episodes == 0:
        raise RuntimeError("No training episodes were found after the requested split.")
    if val_dataset.num_episodes == 0:
        fallback_episode = next(iter(iter_episodes_from_paths(data_paths, allowed_games=requested_games)), None)
        if fallback_episode is not None:
            val_dataset.add_episode(fallback_episode)
    num_train_episodes = train_dataset.num_episodes
    num_val_episodes = val_dataset.num_episodes
    print(
        "[data] build pass complete train_episodes=%d val_episodes=%d train_samples=%d val_samples=%d"
        % (
            num_train_episodes,
            num_val_episodes,
            len(train_dataset),
            len(val_dataset),
        ),
        flush=True,
    )
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_workers = config.get("data_workers")
    if data_workers is None:
        if device.type == "cuda":
            cpu_count = os.cpu_count() or 4
            data_workers = max(2, min(8, cpu_count // 2))
        else:
            data_workers = 0
    data_workers = max(0, int(data_workers))

    loader_kwargs: Dict[str, Any] = {
        "num_workers": data_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate,
    }
    if data_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    metrics_path = output_dir / "metrics.csv"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metadata_map = load_metadata_map(project_root / "environment_files")

    best_metric = float("-inf")
    best_epoch = -1
    best_action_metric = float("-inf")
    best_action_epoch = -1
    best_public_metric = float("-inf")
    best_public_epoch = -1
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError("Resume checkpoint not found: %s" % resume_path)
        resume_state = restore_training_state(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        start_epoch = int(resume_state["start_epoch"])
        best_metric = float(resume_state["best_metric"])
        best_epoch = int(resume_state["best_epoch"])
        payload = resume_state["payload"]
        best_action_metric = float(payload.get("best_action_metric", best_metric))
        best_action_epoch = int(payload.get("best_action_epoch", best_epoch))
        best_public_metric = float(payload.get("best_public_metric", float("-inf")))
        best_public_epoch = int(payload.get("best_public_epoch", -1))
        print(
            "Resumed from %s at epoch %d (%s checkpoint)"
            % (
                resume_path,
                start_epoch,
                "complete" if resume_state["epoch_complete"] else "partial",
            )
        )

    current_epoch = 0
    print(
        "Training setup: device=%s profile=%s epochs=%d batch_size=%d grad_accum=%d data_workers=%d train_games=%d val_games=%d train_episodes=%d val_episodes=%d train_samples=%d val_samples=%d"
        % (
            device,
            args.hardware_profile,
            int(config["epochs"]),
            int(config["batch_size"]),
            int(config["grad_accum"]),
            data_workers,
            len(train_games),
            len(val_games),
            num_train_episodes,
            num_val_episodes,
            len(train_dataset),
            len(val_dataset),
        ),
        flush=True,
    )
    print("Split mode: %s" % args.split_mode, flush=True)
    if requested_games:
        print("Requested games: %s" % ",".join(requested_games), flush=True)
    print("Train games: %s" % ",".join(train_games), flush=True)
    print("Val games: %s" % ",".join(val_games), flush=True)
    print("Data paths: %s" % ",".join(str(path) for path in data_paths), flush=True)

    def checkpoint_callback(extra_state: Dict[str, Any]) -> None:
        checkpoint_path = save_training_checkpoint(
            checkpoints_dir=checkpoints_dir,
            stem="last",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=int(extra_state.get("epoch", current_epoch)),
            best_score=best_metric if best_metric != float("-inf") else 0.0,
            extra_state={
                "best_epoch": best_epoch,
                "best_action_metric": best_action_metric,
                "best_action_epoch": best_action_epoch,
                "best_public_metric": best_public_metric,
                "best_public_epoch": best_public_epoch,
                **extra_state,
            },
        )
        print(
            "[checkpoint] epoch=%d batch_in_epoch=%s optimizer_steps_in_epoch=%s loss=%.4f path=%s"
            % (
                int(extra_state.get("epoch", current_epoch)),
                extra_state.get("batch_in_epoch", "-"),
                extra_state.get("optimizer_steps_in_epoch", "-"),
                float(extra_state.get("last_loss", 0.0)),
                checkpoint_path,
            ),
            flush=True,
        )

    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            current_epoch = epoch
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_accum=int(config["grad_accum"]),
                max_grad_norm=float(config.get("max_grad_norm", 0.0)),
                epoch=epoch,
                phase="train",
                checkpoint_every_steps=int(config.get("checkpoint_every_steps", 0)),
                checkpoint_callback=checkpoint_callback,
                log_every_batches=int(config.get("log_every_batches", 0)),
                aux_archetype_weight=float(config.get("aux_archetype_weight", 0.0)),
                aux_saliency_weight=float(config.get("aux_saliency_weight", 0.0)),
                aux_recon_weight=float(config.get("aux_recon_weight", 0.0)),
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                grad_accum=1,
                epoch=epoch,
                phase="val",
                log_every_batches=int(config.get("log_every_batches", 0)),
                aux_archetype_weight=float(config.get("aux_archetype_weight", 0.0)),
                aux_saliency_weight=float(config.get("aux_saliency_weight", 0.0)),
                aux_recon_weight=float(config.get("aux_recon_weight", 0.0)),
            )

            last_checkpoint = save_training_checkpoint(
                checkpoints_dir=checkpoints_dir,
                stem="last",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                best_score=best_metric if best_metric != float("-inf") else 0.0,
                extra_state={
                    "epoch_complete": True,
                    "best_epoch": best_epoch,
                    "best_action_metric": best_action_metric,
                    "best_action_epoch": best_action_epoch,
                    "best_public_metric": best_public_metric,
                    "best_public_epoch": best_public_epoch,
                },
            )

            public_val_score = None
            if val_games and epoch % int(config["online_val_every"]) == 0:
                selected_games = list(val_games)[: max(1, int(args.online_val_games))]
                print(
                    "[online-val] epoch=%d starting games=%s"
                    % (epoch, ",".join(selected_games)),
                    flush=True,
                )
                public_val_score = evaluate_public_score(
                    checkpoint_path=str(last_checkpoint),
                    project_root=project_root,
                    metadata_map=metadata_map,
                    selected_games=selected_games,
                    max_steps=int(config["max_steps"]),
                    stall_steps=int(config["stall_steps"]),
                    reset_limit=int(config["reset_limit"]),
                )
                print(
                    "[online-val] epoch=%d complete score=%.6f"
                    % (epoch, float(public_val_score)),
                    flush=True,
                )

            row = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_action_loss": round(train_metrics["action_loss"], 6),
                "train_coord_loss": round(train_metrics["coord_loss"], 6),
                "train_value_loss": round(train_metrics["value_loss"], 6),
                "train_avail_loss": round(train_metrics["avail_loss"], 6),
                "train_latent_loss": round(train_metrics["latent_loss"], 6),
                "train_archetype_loss": round(train_metrics["archetype_loss"], 6),
                "train_saliency_loss": round(train_metrics["saliency_loss"], 6),
                "train_recon_loss": round(train_metrics["recon_loss"], 6),
                "train_action_acc": round(train_metrics["action_acc"], 6),
                "train_coord_acc": round(train_metrics["coord_acc"], 6),
                "train_archetype_acc": round(train_metrics["archetype_acc"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_action_loss": round(val_metrics["action_loss"], 6),
                "val_coord_loss": round(val_metrics["coord_loss"], 6),
                "val_value_loss": round(val_metrics["value_loss"], 6),
                "val_avail_loss": round(val_metrics["avail_loss"], 6),
                "val_latent_loss": round(val_metrics["latent_loss"], 6),
                "val_archetype_loss": round(val_metrics["archetype_loss"], 6),
                "val_saliency_loss": round(val_metrics["saliency_loss"], 6),
                "val_recon_loss": round(val_metrics["recon_loss"], 6),
                "val_action_acc": round(val_metrics["action_acc"], 6),
                "val_coord_acc": round(val_metrics["coord_acc"], 6),
                "val_archetype_acc": round(val_metrics["archetype_acc"], 6),
                "public_val_score": None if public_val_score is None else round(public_val_score, 6),
            }
            append_metrics_row(metrics_path, row)
            print(
                "[epoch] %d/%d train_loss=%.6f action=%.6f coord=%.6f value=%.6f avail=%.6f latent=%.6f arche=%.6f sal=%.6f recon=%.6f train_action_acc=%.6f train_coord_acc=%.6f train_arche_acc=%.6f val_loss=%.6f action=%.6f coord=%.6f value=%.6f avail=%.6f latent=%.6f arche=%.6f sal=%.6f recon=%.6f val_action_acc=%.6f val_coord_acc=%.6f val_arche_acc=%.6f public_val_score=%s"
                % (
                    epoch,
                    int(config["epochs"]),
                    train_metrics["loss"],
                    train_metrics["action_loss"],
                    train_metrics["coord_loss"],
                    train_metrics["value_loss"],
                    train_metrics["avail_loss"],
                    train_metrics["latent_loss"],
                    train_metrics["archetype_loss"],
                    train_metrics["saliency_loss"],
                    train_metrics["recon_loss"],
                    train_metrics["action_acc"],
                    train_metrics["coord_acc"],
                    train_metrics["archetype_acc"],
                    val_metrics["loss"],
                    val_metrics["action_loss"],
                    val_metrics["coord_loss"],
                    val_metrics["value_loss"],
                    val_metrics["avail_loss"],
                    val_metrics["latent_loss"],
                    val_metrics["archetype_loss"],
                    val_metrics["saliency_loss"],
                    val_metrics["recon_loss"],
                    val_metrics["action_acc"],
                    val_metrics["coord_acc"],
                    val_metrics["archetype_acc"],
                    "n/a" if public_val_score is None else ("%.6f" % public_val_score),
                ),
                flush=True,
            )

            action_metric = float(val_metrics["action_acc"])
            if action_metric > best_action_metric:
                best_action_metric = action_metric
                best_action_epoch = epoch
                action_checkpoint = save_training_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    stem="best_action",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    best_score=best_action_metric,
                    extra_state={
                        "epoch_complete": True,
                        "best_epoch": best_epoch,
                        "best_action_metric": best_action_metric,
                        "best_action_epoch": best_action_epoch,
                        "best_public_metric": best_public_metric,
                        "best_public_epoch": best_public_epoch,
                        "selection_metric_name": "val_action_acc",
                        "selection_metric": best_action_metric,
                    },
                )
                print("[best-action] epoch=%d metric=%.6f path=%s" % (epoch, best_action_metric, action_checkpoint), flush=True)
                if best_public_metric == float("-inf"):
                    best_metric = best_action_metric
                    best_epoch = best_action_epoch
                    best_checkpoint = save_training_checkpoint(
                        checkpoints_dir=checkpoints_dir,
                        stem="best",
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        epoch=epoch,
                        best_score=best_metric,
                        extra_state={
                            "epoch_complete": True,
                            "best_epoch": best_epoch,
                            "best_action_metric": best_action_metric,
                            "best_action_epoch": best_action_epoch,
                            "best_public_metric": best_public_metric,
                            "best_public_epoch": best_public_epoch,
                            "selection_metric_name": "val_action_acc",
                            "selection_metric": best_metric,
                        },
                    )
                    print("[best] source=action epoch=%d metric=%.6f path=%s" % (epoch, best_metric, best_checkpoint), flush=True)

            if public_val_score is not None and float(public_val_score) > best_public_metric:
                best_public_metric = float(public_val_score)
                best_public_epoch = epoch
                best_metric = best_public_metric
                best_epoch = best_public_epoch
                public_checkpoint = save_training_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    stem="best_public",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    best_score=best_public_metric,
                    extra_state={
                        "epoch_complete": True,
                        "best_epoch": best_epoch,
                        "best_action_metric": best_action_metric,
                        "best_action_epoch": best_action_epoch,
                        "best_public_metric": best_public_metric,
                        "best_public_epoch": best_public_epoch,
                        "selection_metric_name": "public_val_score",
                        "selection_metric": best_public_metric,
                    },
                )
                print("[best-public] epoch=%d metric=%.6f path=%s" % (epoch, best_public_metric, public_checkpoint), flush=True)
                best_checkpoint = save_training_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    stem="best",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    best_score=best_metric,
                    extra_state={
                        "epoch_complete": True,
                        "best_epoch": best_epoch,
                        "best_action_metric": best_action_metric,
                        "best_action_epoch": best_action_epoch,
                        "best_public_metric": best_public_metric,
                        "best_public_epoch": best_public_epoch,
                        "selection_metric_name": "public_val_score",
                        "selection_metric": best_metric,
                    },
                )
                print("[best] source=public epoch=%d metric=%.6f path=%s" % (epoch, best_metric, best_checkpoint), flush=True)
    except KeyboardInterrupt:
        interrupt_checkpoint = save_training_checkpoint(
            checkpoints_dir=checkpoints_dir,
            stem="interrupt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=current_epoch,
            best_score=best_metric if best_metric != float("-inf") else 0.0,
            extra_state={
                "epoch_complete": False,
                "best_epoch": best_epoch,
                "best_action_metric": best_action_metric,
                "best_action_epoch": best_action_epoch,
                "best_public_metric": best_public_metric,
                "best_public_epoch": best_public_epoch,
                "status": "interrupted",
            },
        )
        save_training_checkpoint(
            checkpoints_dir=checkpoints_dir,
            stem="last",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=current_epoch,
            best_score=best_metric if best_metric != float("-inf") else 0.0,
            extra_state={
                "epoch_complete": False,
                "best_epoch": best_epoch,
                "best_action_metric": best_action_metric,
                "best_action_epoch": best_action_epoch,
                "best_public_metric": best_public_metric,
                "best_public_epoch": best_public_epoch,
                "status": "interrupted",
            },
        )
        summary = {
            "status": "interrupted",
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_action_epoch": best_action_epoch,
            "best_action_metric": best_action_metric,
            "best_public_epoch": best_public_epoch,
            "best_public_metric": best_public_metric,
            "resume_checkpoint": str(interrupt_checkpoint),
            "data_paths": [str(path) for path in data_paths],
            "train_games": train_games,
            "val_games": val_games,
            "num_train_episodes": num_train_episodes,
            "num_val_episodes": num_val_episodes,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
        }
        save_json(output_dir / "summary.json", summary)
        print("Training interrupted. Checkpoint saved to %s" % interrupt_checkpoint)
        return

    summary = {
        "status": "finished",
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_action_epoch": best_action_epoch,
        "best_action_metric": best_action_metric,
        "best_public_epoch": best_public_epoch,
        "best_public_metric": best_public_metric,
        "best_checkpoint": str(checkpoints_dir / "best.pth"),
        "best_action_checkpoint": str(checkpoints_dir / "best_action.pth"),
        "best_public_checkpoint": str(checkpoints_dir / "best_public.pth"),
        "data_paths": [str(path) for path in data_paths],
        "train_games": train_games,
        "val_games": val_games,
        "num_train_episodes": num_train_episodes,
        "num_val_episodes": num_val_episodes,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
    }
    save_json(output_dir / "summary.json", summary)
    print("Training finished. Best epoch: %d" % best_epoch)


if __name__ == "__main__":
    main()
