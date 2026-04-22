from __future__ import annotations

import argparse
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
        - (len(transitions) * 0.5)
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    config["created_at_utc"] = utc_timestamp()

    save_json(output_root / "collect_config.json", config)

    metadata_map = load_metadata_map(project_root / "environment_files")
    requested_games = (
        sorted(metadata_map.keys())
        if not args.games
        else sorted(set(game.strip() for game in args.games.split(",") if game.strip()))
    )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(project_root / "environment_files"),
        recordings_dir=str(output_root / "recordings"),
    )

    for game_id in requested_games:
        if game_id not in metadata_map:
            raise KeyError("Game %s not found in environment_files metadata" % game_id)
        baseline_actions = metadata_map[game_id].get("baseline_actions", [])
        per_game = int(config["collect_episodes_per_game"])
        generated = 0
        while generated < per_game:
            seed = seeds[generated % len(seeds)] + (generated // max(1, len(seeds))) * 1000
            episode = beam_collect_game(
                arc=arc,
                game_id=game_id,
                baseline_actions=baseline_actions,
                checkpoint_path=args.checkpoint,
                config=config,
                seed=seed,
            )
            episode["episode_id"] = "%s_seed%d_try%d" % (game_id, seed, generated)
            append_jsonl_gz(episodes_path, episode)
            generated += 1

    print("Saved episodes to %s" % episodes_path)


if __name__ == "__main__":
    main()
