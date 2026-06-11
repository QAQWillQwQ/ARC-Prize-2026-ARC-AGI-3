from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from .common import (
    ensure_dir,
    final_subframe,
    frame_delta,
    informative_subframe,
    load_metadata_map,
    rhae_score,
    save_json,
    utc_timestamp,
    visual_saliency_summary,
)
from .forge_v2_agent import MyAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenLab collector that runs the 0.31 FORGEHybrid V2 notebook agent backend."
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--heldout-games", type=str, default="")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--episodes-per-game", type=int, default=4)
    parser.add_argument(
        "--episodes-per-seed",
        type=int,
        default=None,
        help="If set, collect this many episodes for every game and every seed.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--reset-limit", type=int, default=3)
    parser.add_argument("--step-log-every", type=int, default=5)
    parser.add_argument("--episode-timeout", type=float, default=1800.0)
    parser.add_argument("--notebook", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)


def build_arcade(project_root: Path, recordings_root: Path, logger_name: str) -> Arcade:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(recordings_root),
        logger=logger,
    )


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


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return "%dh%02dm%02ds" % (hours, minutes, secs)
    if minutes > 0:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


def selected_games(metadata_map: Dict[str, Dict[str, Any]], raw_games: Optional[str]) -> List[str]:
    if raw_games:
        games = [part.strip() for part in raw_games.split(",") if part.strip()]
    else:
        games = sorted(metadata_map.keys())
    missing = [game_id for game_id in games if game_id not in metadata_map]
    if missing:
        raise KeyError("Missing game metadata: %s" % ",".join(missing))
    return games


def action_id(action: GameAction) -> int:
    return int(action.value if hasattr(action, "value") else action)


def action_payload(action: GameAction) -> Dict[str, Any]:
    data_obj = getattr(action, "action_data", None)
    if data_obj is None:
        return {}
    try:
        data = dict(data_obj.model_dump())
    except Exception:
        data = {}
    return {key: value for key, value in data.items() if value is not None}


def source_from_reason(reason: str) -> str:
    if reason.startswith("bfs:"):
        return "forge_bfs_replay"
    if reason.startswith("cnn:"):
        return "forge_cnn_fallback"
    if reason.startswith("undo"):
        return "forge_undo"
    if reason.startswith("reset"):
        return "forge_reset"
    if reason.startswith("err:"):
        return "forge_error_fallback"
    return "forge_unknown"


def agent_debug(agent: MyAgent) -> Dict[str, Any]:
    scanned = getattr(agent, "_scanned_actions", None)
    solution = getattr(agent, "_bfs_solution", None)
    return {
        "bfs_tried": bool(getattr(agent, "_bfs_tried", False)),
        "bfs_active": bool(solution),
        "bfs_step": int(getattr(agent, "_bfs_step", 0)),
        "bfs_solution_len": len(solution) if solution else 0,
        "bfs_solved_last": bool(getattr(agent, "_bfs_solved_last", False)),
        "scanned_action_count": len(scanned) if scanned is not None else None,
        "cnn_buffer_size": len(getattr(agent, "buf", [])),
        "epsilon": float(getattr(agent, "_eps", 0.0)),
        "fallback_steps_on_level": int(getattr(agent, "la", 0)),
    }


def level_action_counts(transitions: Sequence[Dict[str, Any]]) -> List[int]:
    counts: List[int] = []
    running = 0
    previous_level = 0
    for transition in transitions:
        running += 1
        levels_after = int(transition.get("levels_after", 0))
        if levels_after > previous_level:
            for _ in range(levels_after - previous_level):
                counts.append(running)
                running = 0
            previous_level = levels_after
    return counts


def write_jsonl_gz(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def append_episode_file(aggregate_path: Path, episode_path: Path) -> None:
    ensure_dir(aggregate_path.parent)
    with gzip.open(episode_path, "rt", encoding="utf-8") as src:
        with gzip.open(aggregate_path, "at", encoding="utf-8") as dst:
            shutil.copyfileobj(src, dst)


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    configure_logging()
    project_root = Path(task["project_root"])
    episode_dir = ensure_dir(Path(task["episode_dir"]))
    trace_path = episode_dir / "step_trace.jsonl"
    full_episode_path = episode_dir / "episode.jsonl.gz"

    arc = build_arcade(
        project_root=project_root,
        recordings_root=episode_dir / "recordings",
        logger_name="arc_agi.collect_forge.%d" % os.getpid(),
    )
    started = time.perf_counter()
    game_id = str(task["game_id"])
    seed = int(task["seed"])
    max_steps = int(task["max_steps"])
    reset_limit = int(task["reset_limit"])
    step_log_every = max(1, int(task["step_log_every"]))
    episode_timeout = float(task["episode_timeout"])
    baseline_actions = list(task["baseline_actions"])

    print(
        "[forge-task-start] game=%s seed=%d episode=%s pid=%d max_steps=%d"
        % (game_id, seed, task["episode_id"], os.getpid(), max_steps),
        flush=True,
    )

    env = arc.make(game_id, seed=seed)
    raw_obs = env.observation_space
    if raw_obs is None:
        raise RuntimeError("Environment returned no initial observation for %s" % game_id)

    agent = MyAgent(
        card_id="",
        game_id=raw_obs.game_id,
        agent_name="forge_v2_collect",
        ROOT_URL="",
        record=False,
        arc_env=env,
    )
    latest_frame = agent._convert_raw_frame_data(raw_obs)
    agent.frames = [latest_frame]

    transitions: List[Dict[str, Any]] = []
    reset_count = 0
    stop_reason = "max_steps"
    previous_frame = final_subframe(raw_obs.frame)

    for step_index in range(max_steps):
        if (time.perf_counter() - started) >= episode_timeout:
            stop_reason = "episode_timeout"
            break

        state_before = latest_frame.state.name if hasattr(latest_frame.state, "name") else str(latest_frame.state)
        if state_before == "WIN":
            stop_reason = "win"
            break
        if state_before == "GAME_OVER" and reset_count >= reset_limit:
            stop_reason = "reset_limit"
            break

        action = agent.choose_action(agent.frames, latest_frame)
        aid = action_id(action)
        payload = action_payload(action)
        reason = str(getattr(action, "reasoning", ""))
        if state_before == "GAME_OVER" and aid == 0:
            reset_count += 1

        next_raw = env.step(action, data=payload, reasoning={"agent_reason": reason})
        if next_raw is None:
            stop_reason = "env_returned_none"
            break

        next_frame = agent._convert_raw_frame_data(next_raw)
        current_frame = final_subframe(next_raw.frame)
        event_frame, event_frame_index, event_delta_pixels = informative_subframe(
            next_raw.frame,
            reference_frame=previous_frame,
        )
        stable_delta_pixels = frame_delta(previous_frame, current_frame)
        delta_pixels = max(stable_delta_pixels, event_delta_pixels)
        state_after = next_frame.state.name if hasattr(next_frame.state, "name") else str(next_frame.state)
        levels_before = int(latest_frame.levels_completed)
        levels_after = int(next_frame.levels_completed)

        debug = agent_debug(agent)
        transition = {
            "step_index": step_index,
            "frame": [row[:] for row in previous_frame],
            "available_actions": [
                int(a.value) if hasattr(a, "value") else int(a)
                for a in getattr(latest_frame, "available_actions", []) or []
            ],
            "action_id": aid,
            "action_data": payload,
            "action_source": source_from_reason(reason),
            "action_reason": reason,
            "action_metadata": {"forge_v2_agent_debug": debug},
            "frame_visual_summary": visual_saliency_summary(previous_frame),
            "next_frame_visual_summary": visual_saliency_summary(current_frame),
            "next_frame": [row[:] for row in current_frame],
            "event_frame": [row[:] for row in event_frame] if event_delta_pixels > stable_delta_pixels else None,
            "event_frame_index": int(event_frame_index),
            "stable_delta_pixels": int(stable_delta_pixels),
            "event_delta_pixels": int(event_delta_pixels),
            "delta_pixels": int(delta_pixels),
            "levels_before": levels_before,
            "levels_after": levels_after,
            "state_before": state_before,
            "state_after": state_after,
            "step_elapsed_seconds": round(float(time.perf_counter() - started), 6),
        }
        transitions.append(transition)

        trace_row = {
            "episode_id": task["episode_id"],
            "game_id": game_id,
            "seed": seed,
            "step": step_index,
            "action": aid,
            "data": payload,
            "reason": reason,
            "source": transition["action_source"],
            "levels": "%d->%d" % (levels_before, levels_after),
            "state": "%s->%s" % (state_before, state_after),
            "delta": int(delta_pixels),
            "scanned": debug["scanned_action_count"],
            "bfs": "%d/%d" % (debug["bfs_step"], debug["bfs_solution_len"]),
            "buffer": debug["cnn_buffer_size"],
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace_row, ensure_ascii=True))
            handle.write("\n")

        should_print = (
            step_index == 0
            or step_index % step_log_every == 0
            or levels_after > levels_before
            or state_after in {"WIN", "GAME_OVER"}
        )
        if should_print:
            print(
                "[forge-step] game=%s seed=%d step=%03d action=A%d data=%s reason=%s levels=%d->%d state=%s->%s delta=%d scanned=%s bfs=%s"
                % (
                    game_id,
                    seed,
                    step_index,
                    aid,
                    json.dumps(payload, sort_keys=True),
                    reason,
                    levels_before,
                    levels_after,
                    state_before,
                    state_after,
                    int(delta_pixels),
                    debug["scanned_action_count"],
                    trace_row["bfs"],
                ),
                flush=True,
            )

        agent.append_frame(next_frame)
        agent.action_counter += 1
        latest_frame = next_frame
        previous_frame = current_frame

    completed_level_actions = level_action_counts(transitions)
    score_info = rhae_score(
        baseline_actions=baseline_actions,
        completed_level_actions=completed_level_actions,
    )
    final_state = latest_frame.state.name if hasattr(latest_frame.state, "name") else str(latest_frame.state)
    total_delta = int(sum(int(t["delta_pixels"]) for t in transitions))
    unique_frames = len({hashlib.md5(json.dumps(t["next_frame"], separators=(",", ":")).encode()).hexdigest() for t in transitions})

    episode = {
        "episode_id": str(task["episode_id"]),
        "game_id": game_id,
        "full_game_id": str(getattr(latest_frame, "game_id", raw_obs.game_id)),
        "seed": seed,
        "agent_backend": "forge_hybrid_v2_0p31_notebook",
        "final_state": final_state,
        "stop_reason": stop_reason,
        "levels_completed": int(latest_frame.levels_completed),
        "actions_taken": len(transitions),
        "baseline_actions": baseline_actions,
        "completed_level_actions": completed_level_actions,
        "score": float(score_info["score"]),
        "level_scores": list(score_info["level_scores"]),
        "total_delta_pixels": total_delta,
        "unique_frames": unique_frames,
        "transitions": transitions,
    }
    write_jsonl_gz(full_episode_path, episode)

    summary = {
        "episode_id": episode["episode_id"],
        "game_id": game_id,
        "full_game_id": episode["full_game_id"],
        "seed": seed,
        "agent_backend": episode["agent_backend"],
        "final_state": final_state,
        "stop_reason": stop_reason,
        "levels_completed": episode["levels_completed"],
        "actions_taken": episode["actions_taken"],
        "score": episode["score"],
        "total_delta_pixels": total_delta,
        "unique_frames": unique_frames,
        "elapsed_seconds": round(float(time.perf_counter() - started), 6),
        "episode_jsonl_gz": str(full_episode_path),
        "step_trace": str(trace_path),
        "worker_pid": os.getpid(),
    }
    save_json(episode_dir / "episode.json", summary)
    save_json(
        episode_dir / "task_config.json",
        {
            "task": {k: v for k, v in task.items() if k not in {"baseline_actions"}},
            "baseline_actions": baseline_actions,
        },
    )
    print(
        "[forge-task-done] game=%s seed=%d levels=%d score=%.3f state=%s actions=%d stop=%s elapsed=%s"
        % (
            game_id,
            seed,
            summary["levels_completed"],
            summary["score"],
            final_state,
            summary["actions_taken"],
            stop_reason,
            format_duration(summary["elapsed_seconds"]),
        ),
        flush=True,
    )
    return summary


def error_summary(task: Dict[str, Any], exc: BaseException) -> Dict[str, Any]:
    episode_dir = ensure_dir(Path(task["episode_dir"]))
    payload = {
        "episode_id": str(task["episode_id"]),
        "game_id": str(task["game_id"]),
        "seed": int(task["seed"]),
        "agent_backend": "forge_hybrid_v2_0p31_notebook",
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "score": 0.0,
        "levels_completed": 0,
        "actions_taken": 0,
        "final_state": "ERROR",
        "stop_reason": "error",
        "episode_jsonl_gz": "",
        "step_trace": str(episode_dir / "step_trace.jsonl"),
    }
    save_json(episode_dir / "error.json", payload)
    save_json(episode_dir / "episode.json", payload)
    return payload


def build_tasks(args: argparse.Namespace, project_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    metadata_map = load_metadata_map(project_root / "environment_files")
    games = selected_games(metadata_map, args.games)
    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    if not seeds:
        seeds = [0]

    tasks: List[Dict[str, Any]] = []
    episodes_root = ensure_dir(output_root / "episodes")
    for game_id in games:
        attempt = 0
        if args.episodes_per_seed is not None:
            for seed in seeds:
                for repeat in range(max(1, int(args.episodes_per_seed))):
                    actual_seed = seed + repeat * 1000
                    episode_id = "%s_seed%d_try%d" % (game_id, actual_seed, attempt)
                    tasks.append(
                        {
                            "project_root": str(project_root),
                            "episode_dir": str(episodes_root / game_id / episode_id),
                            "game_id": game_id,
                            "seed": actual_seed,
                            "episode_id": episode_id,
                            "attempt": attempt,
                            "repeat": repeat,
                            "baseline_actions": list(metadata_map[game_id].get("baseline_actions", [])),
                            "max_steps": int(args.max_steps),
                            "reset_limit": int(args.reset_limit),
                            "step_log_every": int(args.step_log_every),
                            "episode_timeout": float(args.episode_timeout),
                        }
                    )
                    attempt += 1
        else:
            for attempt in range(max(1, int(args.episodes_per_game))):
                seed = seeds[attempt % len(seeds)] + (attempt // max(1, len(seeds))) * 1000
                episode_id = "%s_seed%d_try%d" % (game_id, seed, attempt)
                tasks.append(
                    {
                        "project_root": str(project_root),
                        "episode_dir": str(episodes_root / game_id / episode_id),
                        "game_id": game_id,
                        "seed": seed,
                        "episode_id": episode_id,
                        "attempt": attempt,
                        "repeat": 0,
                        "baseline_actions": list(metadata_map[game_id].get("baseline_actions", [])),
                        "max_steps": int(args.max_steps),
                        "reset_limit": int(args.reset_limit),
                        "step_log_every": int(args.step_log_every),
                        "episode_timeout": float(args.episode_timeout),
                    }
                )
    return tasks


def main() -> None:
    args = parse_args()
    configure_logging()

    project_root = Path(args.project_root).expanduser().resolve()
    output_root = ensure_dir(Path(args.output_root).expanduser().resolve())
    aggregate_path = output_root / "all_episodes.jsonl.gz"
    manifest_path = output_root / "manifest.jsonl"
    summary_path = output_root / "summary.json"
    tasks = build_tasks(args, project_root, output_root)

    config = {
        "collector": "collect_forge_openlab",
        "agent_backend": "forge_hybrid_v2_0p31_notebook",
        "project_root": str(project_root),
        "output_root": str(output_root),
        "games": args.games,
        "heldout_games": args.heldout_games,
        "seeds": args.seeds,
        "episodes_per_game": args.episodes_per_game,
        "episodes_per_seed": args.episodes_per_seed,
        "workers": int(args.workers),
        "max_steps": int(args.max_steps),
        "reset_limit": int(args.reset_limit),
        "step_log_every": int(args.step_log_every),
        "episode_timeout": float(args.episode_timeout),
        "created_at_utc": utc_timestamp(),
        "git_commit": git_commit(project_root),
        "notebook_md5": file_md5(args.notebook),
        "num_tasks": len(tasks),
    }
    save_json(output_root / "run_config.json", config)

    print(
        "[forge-openlab] games=%d tasks=%d workers=%d output=%s"
        % (
            len({task["game_id"] for task in tasks}),
            len(tasks),
            int(args.workers),
            output_root,
        ),
        flush=True,
    )

    start_time = time.perf_counter()
    completed = 0
    best_by_game: Dict[str, Dict[str, Any]] = {}

    def record(summary: Dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        episode_path = summary.get("episode_jsonl_gz")
        if episode_path:
            append_episode_file(aggregate_path, Path(str(episode_path)))
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=True))
            handle.write("\n")

        game_id = str(summary.get("game_id", ""))
        current_best = best_by_game.get(game_id)
        if current_best is None or float(summary.get("score", 0.0)) > float(current_best.get("score", 0.0)):
            best_by_game[game_id] = dict(summary)

        elapsed = time.perf_counter() - start_time
        avg = elapsed / max(1, completed)
        eta = avg * max(0, len(tasks) - completed)
        print(
            "[forge-openlab] %d/%d game=%s levels=%s score=%.3f state=%s actions=%s avg=%s eta=%s"
            % (
                completed,
                len(tasks),
                game_id,
                summary.get("levels_completed", 0),
                float(summary.get("score", 0.0)),
                summary.get("final_state", ""),
                summary.get("actions_taken", 0),
                format_duration(avg),
                format_duration(eta),
            ),
            flush=True,
        )

    workers = max(1, int(args.workers))
    if workers == 1:
        for task in tasks:
            try:
                record(run_task(task))
            except Exception as exc:
                record(error_summary(task, exc))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    record(future.result())
                except Exception as exc:
                    record(error_summary(task, exc))

    summary = {
        "num_tasks": len(tasks),
        "completed": completed,
        "elapsed_seconds": round(float(time.perf_counter() - start_time), 6),
        "aggregate_episodes": str(aggregate_path),
        "manifest": str(manifest_path),
        "best_by_game": {
            game_id: {
                "score": float(row.get("score", 0.0)),
                "levels_completed": int(row.get("levels_completed", 0)),
                "actions_taken": int(row.get("actions_taken", 0)),
                "final_state": str(row.get("final_state", "")),
                "episode_id": str(row.get("episode_id", "")),
                "episode_jsonl_gz": str(row.get("episode_jsonl_gz", "")),
                "step_trace": str(row.get("step_trace", "")),
            }
            for game_id, row in sorted(best_by_game.items())
        },
    }
    save_json(summary_path, summary)
    print("[forge-openlab] saved aggregate=%s" % aggregate_path, flush=True)
    print("[forge-openlab] saved summary=%s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
