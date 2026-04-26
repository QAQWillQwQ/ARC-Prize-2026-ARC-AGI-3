from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from .agent import PolicyGuidedAgent
from .common import (
    CandidateAction,
    append_jsonl_gz,
    final_subframe,
    frame_hash,
    load_metadata_map,
    merge_config,
    rhae_score,
    save_json,
    seed_everything,
    utc_timestamp,
)


@dataclass
class SearchItem:
    sequence: List[Dict[str, Any]]
    episode: Dict[str, Any]
    score: float
    signature: str
    current_frame: List[List[int]]
    available_actions: List[int]
    agent: PolicyGuidedAgent


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return "%dh%02dm%02ds" % (hours, minutes, secs)
    if minutes > 0:
        return "%dm%02ds" % (minutes, secs)
    return "%ds" % secs


def curriculum_key(game_id: str, metadata: Dict[str, Any]) -> tuple[float, float, int, str]:
    baseline = [int(value) for value in metadata.get("baseline_actions", [])]
    if baseline:
        avg_actions = float(sum(baseline)) / float(len(baseline))
        max_actions = float(max(baseline))
        num_levels = len(baseline)
    else:
        avg_actions = 1e9
        max_actions = 1e9
        num_levels = 0
    return (avg_actions, max_actions, num_levels, game_id)


def build_arcade_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    return logger


def serialize_candidate(candidate: CandidateAction) -> Dict[str, Any]:
    return {
        "action_id": int(candidate.action_id),
        "action_data": dict(candidate.action_data or {}),
        "score": float(candidate.score),
        "source": candidate.source,
    }


def build_agent(checkpoint_path: Optional[str], config: Dict[str, Any]) -> PolicyGuidedAgent:
    return PolicyGuidedAgent(
        checkpoint_path=checkpoint_path,
        history=int(config["history"]),
        max_steps=int(config.get("max_steps", 192)),
        stall_steps=int(config.get("stall_steps", 24)),
        reset_limit=int(config.get("reset_limit", 3)),
        coord_budget=int(config["coord_budget"]),
        random_seed=int(config.get("random_seed", 0)),
        game_prior=dict(config.get("game_prior") or {}),
    )


def build_arcade(project_root: Path, recordings_root: Path, logger_name: str) -> Arcade:
    return Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(recordings_root),
        logger=build_arcade_logger(logger_name),
    )


def replay_sequence(
    arc: Arcade,
    game_id: str,
    sequence: Sequence[Dict[str, Any]],
    baseline_actions: Sequence[int],
    checkpoint_path: Optional[str],
    config: Dict[str, Any],
    seed: int,
) -> SearchItem:
    config = dict(config)
    config["random_seed"] = int(seed)
    env = arc.make(game_id, seed=seed)
    if env is None:
        raise RuntimeError("Unable to create environment for %s" % game_id)
    raw_obs = env.observation_space
    if raw_obs is None:
        raise RuntimeError("Environment %s returned no initial observation" % game_id)

    current_frame = final_subframe(raw_obs.frame)
    agent = build_agent(checkpoint_path=checkpoint_path, config=config)
    agent.reset_memory(current_frame)

    transitions: List[Dict[str, Any]] = []
    previous_frame = current_frame
    for step_index, action_spec in enumerate(sequence):
        state_name = raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
        if state_name in ("WIN", "GAME_OVER"):
            break
        candidate = CandidateAction(
            action_id=int(action_spec["action_id"]),
            action_data=dict(action_spec.get("action_data") or {}),
            score=float(action_spec.get("score", 0.0)),
            source=str(action_spec.get("source", "beam")),
        )
        next_obs = env.step(GameAction.from_id(candidate.action_id), data=candidate.action_data or {})
        if next_obs is None:
            break
        next_frame = final_subframe(next_obs.frame)
        novelty = agent.update_memory(
            action=candidate,
            previous_frame=previous_frame,
            next_frame=next_frame,
            levels_before=int(raw_obs.levels_completed),
            levels_after=int(next_obs.levels_completed),
        )
        transitions.append(
            {
                "frame": [row[:] for row in previous_frame],
                "available_actions": list(getattr(raw_obs, "available_actions", [])),
                "action_id": candidate.action_id,
                "action_data": dict(candidate.action_data or {}),
                "next_frame": [row[:] for row in next_frame],
                "levels_before": int(raw_obs.levels_completed),
                "levels_after": int(next_obs.levels_completed),
                "state_before": state_name,
                "state_after": next_obs.state.name if hasattr(next_obs.state, "name") else str(next_obs.state),
                "delta_pixels": sum(
                    1
                    for y in range(len(previous_frame))
                    for x in range(len(previous_frame[y]))
                    if int(previous_frame[y][x]) != int(next_frame[y][x])
                ),
                "novelty": novelty,
                "step_index": step_index,
            }
        )
        raw_obs = next_obs
        previous_frame = next_frame

    completed_level_actions: List[int] = []
    running_actions = 0
    previous_levels = 0
    for transition in transitions:
        running_actions += 1
        levels_after = int(transition["levels_after"])
        if levels_after > previous_levels:
            completed_level_actions.append(running_actions)
            running_actions = 0
            previous_levels = levels_after
    score_info = rhae_score(baseline_actions=baseline_actions, completed_level_actions=completed_level_actions)
    state_name = raw_obs.state.name if hasattr(raw_obs.state, "name") else str(raw_obs.state)
    total_novelty = float(sum(float(transition["novelty"]) for transition in transitions))
    total_delta = int(sum(int(transition["delta_pixels"]) for transition in transitions))
    unique_frames = len(
        {
            frame_hash(transition["next_frame"])
            for transition in transitions
        }
    )
    episode = {
        "game_id": game_id,
        "seed": seed,
        "sequence": list(sequence),
        "final_state": state_name,
        "levels_completed": int(raw_obs.levels_completed),
        "actions_taken": len(transitions),
        "baseline_actions": list(baseline_actions),
        "score": float(score_info["score"]),
        "level_scores": list(score_info["level_scores"]),
        "total_novelty": total_novelty,
        "total_delta_pixels": total_delta,
        "unique_frames": unique_frames,
        "transitions": transitions,
    }
    signature = "%s:%s:%s" % (
        game_id,
        int(raw_obs.levels_completed),
        frame_hash(previous_frame),
    )
    priority = (
        (int(raw_obs.levels_completed) * 1000.0)
        + float(score_info["score"]) * 4.0
        + (500.0 if state_name == "WIN" else 0.0)
        + min(total_delta / 32.0, 60.0)
        + (total_novelty * 8.0)
        + (unique_frames * 1.5)
        + (10.0 if state_name == "NOT_FINISHED" and len(transitions) > 0 else 0.0)
        - (50.0 if state_name == "GAME_OVER" else 0.0)
        - (len(transitions) * 0.15)
    )
    return SearchItem(
        sequence=list(sequence),
        episode=episode,
        score=priority,
        signature=signature,
        current_frame=previous_frame,
        available_actions=list(getattr(raw_obs, "available_actions", [])),
        agent=agent,
    )


def beam_collect_game(
    arc: Arcade,
    game_id: str,
    baseline_actions: Sequence[int],
    checkpoint_path: Optional[str],
    config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    root = replay_sequence(
        arc=arc,
        game_id=game_id,
        sequence=[],
        baseline_actions=baseline_actions,
        checkpoint_path=checkpoint_path,
        config=config,
        seed=seed,
    )
    beam = [root]
    finished: List[SearchItem] = []

    for _ in range(int(config.get("max_search_steps", config.get("max_steps", 96)))):
        expanded: List[SearchItem] = []
        for item in beam:
            if item.episode["final_state"] in ("WIN", "GAME_OVER"):
                finished.append(item)
                continue

            prev_frame = item.agent.history_frames[-2] if len(item.agent.history_frames) > 1 else None
            ranked = item.agent.rank_candidates(
                frame=item.current_frame,
                prev_frame=prev_frame,
                available_actions=item.available_actions,
                levels_completed=int(item.episode["levels_completed"]),
                step_index=len(item.sequence),
            )
            for candidate in ranked[: int(config["branch_factor"])]:
                new_sequence = list(item.sequence)
                new_sequence.append(serialize_candidate(candidate))
                child = replay_sequence(
                    arc=arc,
                    game_id=game_id,
                    sequence=new_sequence,
                    baseline_actions=baseline_actions,
                    checkpoint_path=checkpoint_path,
                    config=config,
                    seed=seed,
                )
                expanded.append(child)

        if not expanded:
            break

        dedup: Dict[str, SearchItem] = {}
        for item in expanded:
            current = dedup.get(item.signature)
            if current is None or item.score > current.score:
                dedup[item.signature] = item
        beam = sorted(dedup.values(), key=lambda item: item.score, reverse=True)[: int(config["beam_width"])]

        if any(item.episode["final_state"] == "WIN" for item in beam):
            finished.extend(beam)
            break

    candidates = finished + beam
    if not candidates:
        return root.episode
    best = max(candidates, key=lambda item: item.score)
    best.episode["search_priority"] = best.score
    return best.episode


def collect_episode_task(task: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(task["project_root"])
    output_root = Path(task["output_root"])
    arc = build_arcade(
        project_root=project_root,
        recordings_root=output_root / "recordings" / ("worker_%d" % os.getpid()),
        logger_name="arc_agi.collect.worker.%d" % os.getpid(),
    )
    episode = beam_collect_game(
        arc=arc,
        game_id=str(task["game_id"]),
        baseline_actions=list(task["baseline_actions"]),
        checkpoint_path=task.get("checkpoint_path"),
        config=dict(task["config"]),
        seed=int(task["seed"]),
    )
    episode["episode_id"] = str(task["episode_id"])
    return episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ARC-AGI-3 public trajectories with structured search.")
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
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--stall-steps", type=int, default=24)
    parser.add_argument("--reset-limit", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize_progress(
    episode: Dict[str, Any],
    completed_episodes: int,
    total_episodes: int,
    start_time: float,
) -> str:
    elapsed = time.perf_counter() - start_time
    avg_seconds = elapsed / completed_episodes if completed_episodes > 0 else 0.0
    remaining_episodes = max(0, total_episodes - completed_episodes)
    eta_seconds = avg_seconds * remaining_episodes if completed_episodes > 0 else 0.0
    episodes_per_minute = (completed_episodes / elapsed) * 60.0 if elapsed > 0 else 0.0
    return (
        "saved %s state=%s levels=%d actions=%d score=%.3f | done %d/%d | %.2f ep/min | avg %s/ep | eta %s"
        % (
            episode["episode_id"],
            episode["final_state"],
            int(episode["levels_completed"]),
            int(episode["actions_taken"]),
            float(episode["score"]),
            completed_episodes,
            total_episodes,
            episodes_per_minute,
            format_duration(avg_seconds),
            format_duration(eta_seconds),
        )
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / "collected"
    episodes_path = run_dir / "episodes.jsonl.gz"

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

    save_json(output_root / "collect_config.json", config)

    metadata_map = load_metadata_map(project_root / "environment_files")
    requested_games = (
        sorted(metadata_map.keys())
        if not args.games
        else sorted(set(game.strip() for game in args.games.split(",") if game.strip()))
    )
    if not args.games:
        requested_games = sorted(requested_games, key=lambda game_id: curriculum_key(game_id, metadata_map[game_id]))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    total_episodes = len(requested_games) * int(config["collect_episodes_per_game"])
    start_time = time.perf_counter()
    completed_episodes = 0
    tasks: List[Dict[str, Any]] = []
    per_game = int(config["collect_episodes_per_game"])
    for generated in range(per_game):
        for game_index, game_id in enumerate(requested_games, start=1):
            if game_id not in metadata_map:
                raise KeyError("Game %s not found in environment_files metadata" % game_id)
            baseline_actions = metadata_map[game_id].get("baseline_actions", [])
            seed = seeds[generated % len(seeds)] + (generated // max(1, len(seeds))) * 1000
            tasks.append(
                {
                    "project_root": str(project_root),
                    "output_root": str(output_root),
                    "game_id": game_id,
                    "game_index": game_index,
                    "episode_index": generated + 1,
                    "episode_id": "%s_seed%d_try%d" % (game_id, seed, generated),
                    "seed": seed,
                    "baseline_actions": list(baseline_actions),
                    "checkpoint_path": args.checkpoint,
                    "config": dict(config),
                }
            )

    workers = max(1, int(args.workers))
    if workers == 1:
        arc = build_arcade(
            project_root=project_root,
            recordings_root=output_root / "recordings",
            logger_name="arc_agi.collect",
        )
        for task in tasks:
            print(
                "Collecting %s (%d/%d) episode %d/%d seed=%d | overall %d/%d | elapsed %s"
                % (
                    task["game_id"],
                    int(task["game_index"]),
                    len(requested_games),
                    int(task["episode_index"]),
                    int(config["collect_episodes_per_game"]),
                    int(task["seed"]),
                    completed_episodes + 1,
                    total_episodes,
                    format_duration(time.perf_counter() - start_time),
                ),
                flush=True,
            )
            episode = beam_collect_game(
                arc=arc,
                game_id=str(task["game_id"]),
                baseline_actions=list(task["baseline_actions"]),
                checkpoint_path=args.checkpoint,
                config=config,
                seed=int(task["seed"]),
            )
            episode["episode_id"] = str(task["episode_id"])
            append_jsonl_gz(episodes_path, episode)
            completed_episodes += 1
            print(summarize_progress(episode, completed_episodes, total_episodes, start_time), flush=True)
    else:
        print(
            "Running parallel collect with %d workers across %d episodes"
            % (workers, total_episodes),
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(collect_episode_task, task) for task in tasks]
            for future in as_completed(futures):
                episode = future.result()
                append_jsonl_gz(episodes_path, episode)
                completed_episodes += 1
                print(summarize_progress(episode, completed_episodes, total_episodes, start_time), flush=True)

    total_elapsed = time.perf_counter() - start_time
    print("Saved episodes to %s" % episodes_path)
    print(
        "Collection finished: %d episodes in %s (avg %s/ep)"
        % (
            completed_episodes,
            format_duration(total_elapsed),
            format_duration(total_elapsed / max(completed_episodes, 1)),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
