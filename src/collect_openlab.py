from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .collect import beam_collect_game, build_arcade, curriculum_key, format_duration
from .common import ensure_dir, load_metadata_map, merge_config, save_json, seed_everything, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenLab-safe parallel collector with isolated per-game output folders."
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hardware-profile", type=str, default="a100")
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--episodes-per-game", type=int, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--branch-factor", type=int, default=None)
    parser.add_argument("--coord-budget", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=144)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notebook", type=str, default=None, help="Optional notebook path used to record an md5 in run metadata.")
    return parser.parse_args()


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def file_md5(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return None
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl_gz(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(task["project_root"])
    episode_dir = Path(task["episode_dir"])
    ensure_dir(episode_dir)
    config = dict(task["config"])
    arc = build_arcade(
        project_root=project_root,
        recordings_root=episode_dir / "recordings",
        logger_name="arc_agi.collect_openlab.%d" % os.getpid(),
    )
    started = time.perf_counter()
    episode = beam_collect_game(
        arc=arc,
        game_id=str(task["game_id"]),
        baseline_actions=list(task["baseline_actions"]),
        checkpoint_path=task.get("checkpoint"),
        config=config,
        seed=int(task["seed"]),
    )
    elapsed = time.perf_counter() - started
    episode["episode_id"] = str(task["episode_id"])
    episode["worker_pid"] = os.getpid()
    episode["elapsed_seconds"] = round(float(elapsed), 6)
    save_json(episode_dir / "episode.json", episode)
    with gzip.open(episode_dir / "episode.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, ensure_ascii=True))
        handle.write("\n")
    save_json(
        episode_dir / "task_config.json",
        {
            "task": {
                key: value
                for key, value in task.items()
                if key not in {"baseline_actions", "config"}
            },
            "baseline_actions": list(task["baseline_actions"]),
            "config": config,
        },
    )
    return episode


def selected_games(metadata_map: Dict[str, Dict[str, Any]], raw_games: Optional[str]) -> List[str]:
    if raw_games:
        games = sorted(set(part.strip() for part in raw_games.split(",") if part.strip()))
    else:
        games = sorted(metadata_map.keys(), key=lambda game_id: curriculum_key(game_id, metadata_map[game_id]))
    missing = [game_id for game_id in games if game_id not in metadata_map]
    if missing:
        raise KeyError("Missing game metadata: %s" % ",".join(missing))
    return games


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)
    episodes_root = ensure_dir(output_root / "episodes")
    aggregate_path = output_root / "all_episodes.jsonl.gz"
    manifest_path = output_root / "manifest.jsonl"

    overrides = {
        "collect_episodes_per_game": args.episodes_per_game,
        "beam_width": args.beam_width,
        "branch_factor": args.branch_factor,
        "coord_budget": args.coord_budget,
        "max_steps": args.max_steps,
        "max_search_steps": args.max_steps,
        "stall_steps": args.stall_steps,
        "reset_limit": args.reset_limit,
    }
    config = merge_config(args.hardware_profile, overrides)
    config["hardware_profile"] = args.hardware_profile
    config["checkpoint"] = args.checkpoint
    config["workers"] = int(args.workers)
    config["created_at_utc"] = utc_timestamp()
    config["git_commit"] = git_commit(project_root)
    config["notebook_md5"] = file_md5(args.notebook)

    metadata_map = load_metadata_map(project_root / "environment_files")
    games = selected_games(metadata_map, args.games)
    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    if not seeds:
        seeds = [0]
    per_game = int(config["collect_episodes_per_game"])

    tasks: List[Dict[str, Any]] = []
    for game_id in games:
        for attempt in range(per_game):
            seed = seeds[attempt % len(seeds)] + (attempt // max(1, len(seeds))) * 1000
            episode_id = "%s_seed%d_try%d" % (game_id, seed, attempt)
            episode_dir = episodes_root / game_id / episode_id
            tasks.append(
                {
                    "project_root": str(project_root),
                    "episode_dir": str(episode_dir),
                    "game_id": game_id,
                    "seed": seed,
                    "episode_id": episode_id,
                    "attempt": attempt,
                    "baseline_actions": list(metadata_map[game_id].get("baseline_actions", [])),
                    "checkpoint": args.checkpoint,
                    "config": dict(config),
                }
            )

    save_json(
        output_root / "run_config.json",
        {
            "config": config,
            "games": games,
            "seeds": seeds,
            "num_tasks": len(tasks),
            "output_root": str(output_root),
            "aggregate_episodes": str(aggregate_path),
        },
    )

    print(
        "[openlab-collect] games=%d tasks=%d workers=%d output=%s"
        % (len(games), len(tasks), int(args.workers), output_root),
        flush=True,
    )
    start_time = time.perf_counter()
    completed = 0
    best_by_game: Dict[str, Dict[str, Any]] = {}
    def record_episode(task: Dict[str, Any], episode: Dict[str, Any]) -> None:
        nonlocal completed
        append_jsonl_gz(aggregate_path, episode)
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps({
                "episode_id": episode.get("episode_id"),
                "game_id": episode.get("game_id"),
                "seed": episode.get("seed"),
                "score": episode.get("score", 0.0),
                "levels_completed": episode.get("levels_completed", 0),
                "actions_taken": episode.get("actions_taken", 0),
                "final_state": episode.get("final_state", ""),
                "episode_json": str(Path(task["episode_dir"]) / "episode.json"),
                "episode_jsonl_gz": str(Path(task["episode_dir"]) / "episode.jsonl.gz"),
            }, ensure_ascii=True))
            manifest.write("\n")

        game_id = str(episode.get("game_id", task["game_id"]))
        previous_best = best_by_game.get(game_id)
        if previous_best is None or float(episode.get("score", 0.0)) > float(previous_best.get("score", 0.0)):
            best_by_game[game_id] = episode
        completed += 1
        elapsed = time.perf_counter() - start_time
        avg = elapsed / max(1, completed)
        eta = avg * max(0, len(tasks) - completed)
        print(
            "[openlab-collect] %d/%d game=%s levels=%s score=%.3f state=%s actions=%s avg=%s eta=%s"
            % (
                completed,
                len(tasks),
                game_id,
                episode.get("levels_completed", 0),
                float(episode.get("score", 0.0)),
                episode.get("final_state", ""),
                episode.get("actions_taken", 0),
                format_duration(avg),
                format_duration(eta),
            ),
            flush=True,
        )

    def error_episode(task: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
        episode = {
            "episode_id": str(task["episode_id"]),
            "game_id": str(task["game_id"]),
            "seed": int(task["seed"]),
            "error": repr(exc),
            "score": 0.0,
            "levels_completed": 0,
            "actions_taken": 0,
            "final_state": "ERROR",
        }
        error_dir = ensure_dir(Path(task["episode_dir"]))
        save_json(error_dir / "error.json", episode)
        return episode

    if int(args.workers) == 1:
        for task in tasks:
            try:
                episode = run_task(task)
            except Exception as exc:
                episode = error_episode(task, exc)
            record_episode(task, episode)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    episode = future.result()
                except Exception as exc:
                    episode = error_episode(task, exc)
                record_episode(task, episode)

    summary = {
        "num_tasks": len(tasks),
        "completed": completed,
        "elapsed_seconds": round(float(time.perf_counter() - start_time), 6),
        "aggregate_episodes": str(aggregate_path),
        "manifest": str(manifest_path),
        "best_by_game": {
            game_id: {
                "score": float(episode.get("score", 0.0)),
                "levels_completed": int(episode.get("levels_completed", 0)),
                "actions_taken": int(episode.get("actions_taken", 0)),
                "final_state": str(episode.get("final_state", "")),
                "episode_id": str(episode.get("episode_id", "")),
            }
            for game_id, episode in sorted(best_by_game.items())
        },
    }
    save_json(output_root / "summary.json", summary)
    print("[openlab-collect] saved aggregate=%s" % aggregate_path, flush=True)
    print("[openlab-collect] saved summary=%s" % (output_root / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
