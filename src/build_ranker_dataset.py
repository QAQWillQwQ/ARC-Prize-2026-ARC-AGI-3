from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .common import ensure_dir, iter_jsonl_gz, save_json
from .ranker_features import (
    FEATURE_NAMES,
    action_matches,
    build_candidate_actions,
    candidate_feature_vector,
    source_phase_label,
    transition_utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate ranking examples from ARC trajectory logs."
    )
    parser.add_argument(
        "--episodes",
        type=str,
        required=True,
        help="One episodes.jsonl.gz file or a comma separated list of files.",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--metadata-output", type=str, default=None)
    parser.add_argument("--coord-budget", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--min-positive-utility", type=float, default=0.05)
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--click-tolerance", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def parse_paths(raw: str) -> List[Path]:
    paths = [Path(part.strip()).expanduser().resolve() for part in raw.split(",") if part.strip()]
    if not paths:
        raise RuntimeError("No --episodes path was provided.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing episode files: %s" % ", ".join(missing))
    return paths


def iter_episodes(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        yield from iter_jsonl_gz(path)


def build_example(
    episode: Dict[str, Any],
    transition: Dict[str, Any],
    previous_frame: Optional[Sequence[Sequence[int]]],
    coord_budget: int,
    max_steps: int,
    click_tolerance: int,
) -> Optional[Dict[str, Any]]:
    frame = transition.get("frame")
    if not frame:
        return None
    available_actions = [int(value) for value in transition.get("available_actions", [])]
    action_id = int(transition.get("action_id", 0))
    action_data = dict(transition.get("action_data") or {})
    candidates = build_candidate_actions(
        frame=frame,
        prev_frame=previous_frame,
        available_actions=available_actions,
        actual_action_id=action_id,
        actual_action_data=action_data,
        coord_budget=coord_budget,
    )
    if not candidates:
        return None

    utility = transition_utility(transition)
    candidate_rows: List[Dict[str, Any]] = []
    selected_index = -1
    selected_target = float(utility)
    for index, candidate in enumerate(candidates):
        chosen = action_matches(
            candidate=candidate,
            action_id=action_id,
            action_data=action_data,
            click_tolerance=click_tolerance,
        )
        if chosen and selected_index < 0:
            selected_index = index
        action_source = str(transition.get("action_source", ""))
        source_verified = bool(
            chosen
            and (
                action_source.startswith("source")
                or action_source in {"source_planner", "forge_bfs_replay"}
                or int(transition.get("delta_pixels", 0)) > 0
                or int(transition.get("levels_after", 0)) > int(transition.get("levels_before", 0))
            )
        )
        candidate_rows.append(
            {
                "action_id": int(candidate.action_id),
                "action_data": dict(candidate.action_data or {}),
                "source": str(candidate.source),
                "is_chosen": bool(chosen),
                "target": float(utility if chosen else 0.0),
                "features": candidate_feature_vector(
                    candidate=candidate,
                    frame=frame,
                    prev_frame=previous_frame,
                    available_actions=available_actions,
                    levels_before=int(transition.get("levels_before", 0)),
                    step_index=int(transition.get("step_index", 0)),
                    max_steps=max_steps,
                    source_verified=source_verified,
                ),
            }
        )

    if selected_index < 0:
        return None
    return {
        "episode_id": str(episode.get("episode_id", "")),
        "game_id": str(episode.get("game_id", "")),
        "seed": int(episode.get("seed", 0)),
        "step_index": int(transition.get("step_index", 0)),
        "levels_before": int(transition.get("levels_before", 0)),
        "levels_after": int(transition.get("levels_after", 0)),
        "delta_pixels": int(transition.get("delta_pixels", 0)),
        "state_after": str(transition.get("state_after", "")),
        "phase_label": source_phase_label(transition),
        "selected_index": int(selected_index),
        "selected_target": float(selected_target),
        "candidates": candidate_rows,
    }


def main() -> None:
    args = parse_args()
    episode_paths = parse_paths(args.episodes)
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_output).expanduser().resolve()
        if args.metadata_output
        else output_path.with_suffix("").with_suffix(".metadata.json")
    )
    ensure_dir(output_path.parent)

    stats = {
        "episodes": 0,
        "transitions": 0,
        "examples": 0,
        "positive_examples": 0,
        "negative_examples": 0,
        "skipped_low_utility": 0,
        "games": {},
        "phase_labels": {},
    }

    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for episode in iter_episodes(episode_paths):
            stats["episodes"] += 1
            game_id = str(episode.get("game_id", ""))
            stats["games"].setdefault(game_id, {"episodes": 0, "examples": 0})
            stats["games"][game_id]["episodes"] += 1
            previous_frame: Optional[Sequence[Sequence[int]]] = None
            for transition in episode.get("transitions", []):
                stats["transitions"] += 1
                example = build_example(
                    episode=episode,
                    transition=transition,
                    previous_frame=previous_frame,
                    coord_budget=int(args.coord_budget),
                    max_steps=int(args.max_steps),
                    click_tolerance=int(args.click_tolerance),
                )
                previous_frame = transition.get("next_frame") or transition.get("frame")
                if example is None:
                    continue
                phase = str(example["phase_label"])
                stats["phase_labels"][phase] = int(stats["phase_labels"].get(phase, 0)) + 1
                if float(example["selected_target"]) >= float(args.min_positive_utility):
                    stats["positive_examples"] += 1
                elif not args.include_negative:
                    stats["skipped_low_utility"] += 1
                    continue
                else:
                    stats["negative_examples"] += 1
                handle.write(json.dumps(example, ensure_ascii=True))
                handle.write("\n")
                stats["examples"] += 1
                stats["games"][game_id]["examples"] += 1
                if int(args.progress_every) > 0 and stats["examples"] % int(args.progress_every) == 0:
                    print(
                        "[ranker-data] examples=%d transitions=%d episodes=%d current_game=%s"
                        % (
                            int(stats["examples"]),
                            int(stats["transitions"]),
                            int(stats["episodes"]),
                            game_id,
                        ),
                        flush=True,
                    )
                if int(args.max_examples) > 0 and stats["examples"] >= int(args.max_examples):
                    break
            if int(args.max_examples) > 0 and stats["examples"] >= int(args.max_examples):
                break

    save_json(
        metadata_path,
        {
            "feature_names": FEATURE_NAMES,
            "source_episode_paths": [str(path) for path in episode_paths],
            "coord_budget": int(args.coord_budget),
            "max_steps": int(args.max_steps),
            "min_positive_utility": float(args.min_positive_utility),
            "include_negative": bool(args.include_negative),
            "click_tolerance": int(args.click_tolerance),
            "stats": stats,
        },
    )
    print("Saved ranker dataset to %s" % output_path, flush=True)
    print("Saved metadata to %s" % metadata_path, flush=True)
    print(
        "examples=%d positive=%d skipped_low_utility=%d games=%d"
        % (
            int(stats["examples"]),
            int(stats["positive_examples"]),
            int(stats["skipped_low_utility"]),
            len(stats["games"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
