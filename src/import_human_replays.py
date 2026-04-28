from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .common import (
    ACTION_IDS,
    ensure_dir,
    episode_level_actions,
    frame_delta,
    frame_hash,
    load_metadata_map,
    novelty_bonus,
    rhae_score,
    save_json,
    write_jsonl_gz,
)


@dataclass
class RecordingEntry:
    game_id: str
    source_game_id: str
    source_name: str
    source_guid: str
    rows: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ARC human replay logs into training episodes.jsonl.gz format.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--input", type=str, required=True, help="Path to the human replay zip file or extracted directory.")
    parser.add_argument("--output", type=str, required=True, help="Output episodes.jsonl.gz path.")
    parser.add_argument("--games", type=str, default=None, help="Optional comma separated game ids, e.g. sp80,lp85,ar25.")
    parser.add_argument("--min-levels", type=int, default=0, help="Keep only replays that reach at least this many levels.")
    parser.add_argument("--solved-only", action="store_true", help="Keep only human replays that end in WIN.")
    parser.add_argument("--progress-every", type=int, default=5, help="Print a progress line every N recordings.")
    parser.add_argument(
        "--top-k-per-game",
        type=int,
        default=None,
        help="If set, keep only the best K replays per game ranked by levels, score, then action count.",
    )
    return parser.parse_args()


def normalized_game_id(raw: str) -> str:
    return str(raw).split("-", 1)[0]


def recording_guid_from_name(name: str) -> str:
    return Path(name).name.replace(".recording.jsonl", "")


def iter_zip_recordings(path: Path, allowed_games: Optional[Set[str]] = None) -> Iterator[RecordingEntry]:
    with zipfile.ZipFile(path) as zf:
        names: List[str] = []
        for name in zf.namelist():
            if not name.startswith("public_games-dataset/") or not name.endswith(".recording.jsonl"):
                continue
            parts = PurePosixPath(name).parts
            if len(parts) >= 2 and allowed_games is not None and parts[1] not in allowed_games:
                continue
            names.append(name)
        names.sort()
        for name in names:
            with zf.open(name) as handle:
                rows = [json.loads(line) for line in handle.read().decode("utf-8").splitlines() if line.strip()]
            interactive = [row for row in rows if "action_input" in row.get("data", {}) and "frame" in row.get("data", {})]
            if not interactive:
                continue
            source_game_id = str(interactive[0]["data"].get("game_id", ""))
            yield RecordingEntry(
                game_id=normalized_game_id(source_game_id),
                source_game_id=source_game_id,
                source_name=name,
                source_guid=recording_guid_from_name(name),
                rows=rows,
            )


def iter_dir_recordings(path: Path, allowed_games: Optional[Set[str]] = None) -> Iterator[RecordingEntry]:
    for recording_path in sorted(path.rglob("*.recording.jsonl")):
        if "__MACOSX" in recording_path.parts:
            continue
        if allowed_games is not None:
            relative_parts = recording_path.relative_to(path).parts
            if len(relative_parts) >= 2 and relative_parts[0] == "public_games-dataset":
                if relative_parts[1] not in allowed_games:
                    continue
        rows = [json.loads(line) for line in recording_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        interactive = [row for row in rows if "action_input" in row.get("data", {}) and "frame" in row.get("data", {})]
        if not interactive:
            continue
        source_game_id = str(interactive[0]["data"].get("game_id", ""))
        yield RecordingEntry(
            game_id=normalized_game_id(source_game_id),
            source_game_id=source_game_id,
            source_name=str(recording_path),
            source_guid=recording_guid_from_name(str(recording_path)),
            rows=rows,
        )


def iter_recordings(path: Path, allowed_games: Optional[Set[str]] = None) -> Iterator[RecordingEntry]:
    if path.is_file() and path.suffix.lower() == ".zip":
        yield from iter_zip_recordings(path, allowed_games=allowed_games)
        return
    if path.is_dir():
        yield from iter_dir_recordings(path, allowed_games=allowed_games)
        return
    raise FileNotFoundError("Unsupported human replay input: %s" % path)


def summary_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for row in reversed(rows):
        data = row.get("data", {})
        if "won" in data and "levels_completed" in data:
            return data
    return {}


def interactive_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if "action_input" in row.get("data", {}) and "frame" in row.get("data", {})]


ACTION_NAME_TO_ID = {"RESET": 0, **{"ACTION%d" % index: index for index in range(1, 8)}}


def normalize_action_id(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return int(raw)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    text = str(raw).strip().upper()
    if not text:
        return None
    if text in ACTION_NAME_TO_ID:
        return ACTION_NAME_TO_ID[text]
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    match = re.fullmatch(r"ACTION[_\s-]*(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def normalize_available_actions(raw_values: Any) -> List[int]:
    if not isinstance(raw_values, list):
        return []
    normalized: List[int] = []
    for raw in raw_values:
        action_id = normalize_action_id(raw)
        if action_id is None or action_id in normalized:
            continue
        normalized.append(action_id)
    return normalized


def transition_action_data(action_id: int, raw_data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    if action_id != 6:
        return None
    if "x" not in raw_data or "y" not in raw_data:
        return None
    return {"x": int(raw_data["x"]), "y": int(raw_data["y"])}


def normalize_frame(frame: Any) -> Optional[List[List[int]]]:
    if not isinstance(frame, list) or not frame:
        return None
    if isinstance(frame[0], list) and frame[0] and isinstance(frame[0][0], list):
        frame = frame[0]
    if not isinstance(frame, list) or not frame or not isinstance(frame[0], list):
        return None
    return [[int(value) for value in row] for row in frame]


def build_episode(
    recording: RecordingEntry,
    metadata_map: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    rows = interactive_rows(recording.rows)
    if len(rows) < 2:
        return None

    transitions: List[Dict[str, Any]] = []
    seen_signatures: List[str] = []
    context_row = rows[0]
    reset_actions = 0

    for current_row in rows[1:]:
        current = current_row.get("data", {})
        previous = context_row.get("data", {})
        action_input = current.get("action_input", {}) or {}
        action_id = normalize_action_id(action_input.get("id"))

        if action_id == 0:
            reset_actions += 1
            context_row = current_row
            continue

        if action_id not in ACTION_IDS:
            context_row = current_row
            continue

        prev_frame = normalize_frame(previous.get("frame"))
        next_frame = normalize_frame(current.get("frame"))
        if prev_frame is None or next_frame is None:
            context_row = current_row
            continue

        signature = frame_hash(next_frame)
        novelty = novelty_bonus(signature, seen_signatures)
        delta_pixels = frame_delta(prev_frame, next_frame)
        raw_action_data = action_input.get("data", {}) or {}
        if not isinstance(raw_action_data, dict):
            raw_action_data = {}

        transitions.append(
            {
                "frame": prev_frame,
                "available_actions": normalize_available_actions(previous.get("available_actions") or []),
                "action_id": action_id,
                "action_data": transition_action_data(action_id, raw_action_data),
                "next_frame": next_frame,
                "levels_before": int(previous.get("levels_completed", 0)),
                "levels_after": int(current.get("levels_completed", 0)),
                "state_before": str(previous.get("state", "NOT_FINISHED")),
                "state_after": str(current.get("state", "NOT_FINISHED")),
                "delta_pixels": int(delta_pixels),
                "novelty": float(novelty),
                "source_timestamp": str(current_row.get("timestamp", "")),
            }
        )
        seen_signatures.append(signature)
        context_row = current_row

    if not transitions:
        return None

    summary = summary_row(recording.rows)
    final_data = context_row.get("data", {})
    baseline_actions = list(metadata_map.get(recording.game_id, {}).get("baseline_actions", []))
    completed_level_actions = episode_level_actions(transitions)
    score_info = rhae_score(
        baseline_actions=baseline_actions,
        completed_level_actions=completed_level_actions,
    )

    total_delta = int(sum(int(item["delta_pixels"]) for item in transitions))
    total_novelty = float(sum(float(item["novelty"]) for item in transitions))

    episode = {
        "episode_id": "human_%s_%s_levels%d_actions%d"
        % (
            recording.game_id,
            recording.source_guid,
            int(summary.get("levels_completed", final_data.get("levels_completed", 0))),
            int(summary.get("total_actions", len(transitions))),
        ),
        "game_id": recording.game_id,
        "seed": -1,
        "final_state": str(final_data.get("state", "NOT_FINISHED")),
        "levels_completed": int(final_data.get("levels_completed", 0)),
        "actions_taken": len(transitions),
        "score": float(score_info["score"]),
        "level_scores": list(score_info["level_scores"]),
        "total_delta_pixels": total_delta,
        "total_novelty": total_novelty,
        "transitions": transitions,
        "source_type": "human_public_demo",
        "source_game_id": recording.source_game_id,
        "source_guid": recording.source_guid,
        "source_name": recording.source_name,
        "source_summary": {
            "won": int(summary.get("won", 1 if final_data.get("state") == "WIN" else 0)),
            "levels_completed": int(summary.get("levels_completed", final_data.get("levels_completed", 0))),
            "total_actions": int(summary.get("total_actions", len(transitions))),
            "played": int(summary.get("played", 1)),
            "reset_actions": int(reset_actions),
        },
    }
    return episode


def ranking_key(episode: Dict[str, Any]) -> Tuple[int, float, int]:
    return (
        int(episode.get("levels_completed", 0)),
        float(episode.get("score", 0.0)),
        -int(episode.get("source_summary", {}).get("total_actions", episode.get("actions_taken", 0))),
    )


def keep_episode(
    episode: Dict[str, Any],
    allowed_games: Optional[Sequence[str]],
    min_levels: int,
    solved_only: bool,
) -> bool:
    if allowed_games is not None and str(episode["game_id"]) not in allowed_games:
        return False
    if int(episode.get("levels_completed", 0)) < int(min_levels):
        return False
    if solved_only and str(episode.get("final_state")) != "WIN":
        return False
    return True


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    allowed_games = None if not args.games else [item.strip() for item in args.games.split(",") if item.strip()]
    allowed_games_set = None if allowed_games is None else set(allowed_games)
    metadata_map = load_metadata_map(project_root / "environment_files")

    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()

    counts_by_game: Dict[str, int] = defaultdict(int)
    selected_by_game: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_seen = 0
    total_written = 0
    total_candidates = 0
    ordered: List[Dict[str, Any]] = []
    progress_every = max(1, int(args.progress_every))

    print(
        "[human-import] input=%s output=%s games=%s min_levels=%d solved_only=%s top_k_per_game=%s"
        % (
            input_path,
            output_path,
            ",".join(allowed_games) if allowed_games else "ALL",
            int(args.min_levels),
            bool(args.solved_only),
            args.top_k_per_game,
        ),
        flush=True,
    )

    def report_progress(force: bool = False) -> None:
        if total_seen == 0:
            return
        if not force and total_seen % progress_every != 0:
            return
        print(
            "[human-import] processed=%d candidate_episodes=%d written=%d"
            % (total_seen, total_candidates, total_written),
            flush=True,
        )

    if args.top_k_per_game is None:
        for recording in iter_recordings(input_path, allowed_games=allowed_games_set):
            total_seen += 1
            episode = build_episode(recording, metadata_map)
            if episode is None:
                report_progress()
                continue
            if not keep_episode(episode, allowed_games, args.min_levels, args.solved_only):
                report_progress()
                continue
            total_candidates += 1
            ordered.append(episode)
            counts_by_game[str(episode["game_id"])] += 1
            total_written += 1
            report_progress()
    else:
        limit = max(1, int(args.top_k_per_game))
        for recording in iter_recordings(input_path, allowed_games=allowed_games_set):
            total_seen += 1
            episode = build_episode(recording, metadata_map)
            if episode is None:
                report_progress()
                continue
            if not keep_episode(episode, allowed_games, args.min_levels, args.solved_only):
                report_progress()
                continue
            total_candidates += 1
            game_id = str(episode["game_id"])
            bucket = selected_by_game[game_id]
            bucket.append(episode)
            bucket.sort(key=ranking_key, reverse=True)
            del bucket[limit:]
            report_progress()

        ordered: List[Dict[str, Any]] = []
        for game_id in sorted(selected_by_game):
            bucket = selected_by_game[game_id]
            counts_by_game[game_id] = len(bucket)
            total_written += len(bucket)
            ordered.extend(bucket)

    write_jsonl_gz(output_path, ordered)
    report_progress(force=True)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "games": allowed_games,
        "min_levels": int(args.min_levels),
        "solved_only": bool(args.solved_only),
        "candidate_episodes": total_candidates,
        "top_k_per_game": args.top_k_per_game,
        "recordings_seen": total_seen,
        "episodes_written": total_written,
        "counts_by_game": dict(sorted(counts_by_game.items())),
    }
    save_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
