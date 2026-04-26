from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arcengine import GameAction

from .collect import build_arcade, collect_episode_task, curriculum_key, format_duration
from .common import (
    append_jsonl_gz,
    final_subframe,
    frame_delta,
    load_metadata_map,
    merge_config,
    salient_points,
    save_json,
    seed_everything,
    utc_timestamp,
)


@dataclass
class StageSpec:
    name: str
    selector: str
    episodes_per_game: int
    beam_width: int
    branch_factor: int
    coord_budget: int
    max_steps: int
    max_games: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ARC-AGI-3 public trajectories with staged curriculum search.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hardware-profile", type=str, default="a100")
    parser.add_argument("--games", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_stage_specs(checkpoint_path: Optional[str]) -> List[StageSpec]:
    if checkpoint_path:
        return [
            StageSpec("policy_warmup", "warmup", 8, 6, 10, 24, 112, max_games=10),
            StageSpec("policy_breadth", "all", 8, 6, 10, 24, 112),
            StageSpec("policy_focus", "focus", 16, 8, 12, 28, 128, max_games=10),
            StageSpec("policy_rescue", "rescue", 10, 8, 14, 32, 144, max_games=8),
            StageSpec("policy_exploit", "exploit", 20, 8, 12, 28, 144, max_games=6),
        ]
    return [
        StageSpec("warmup_safe", "warmup", 8, 6, 8, 20, 96, max_games=10),
        StageSpec("breadth_all", "all", 8, 6, 10, 24, 112),
        StageSpec("focus_progress", "focus", 16, 8, 12, 28, 128, max_games=10),
        StageSpec("rescue_hard", "rescue", 10, 8, 14, 32, 144, max_games=8),
        StageSpec("exploit_best", "exploit", 20, 8, 12, 28, 144, max_games=6),
    ]


def stage_config(base_profile: str, stage: StageSpec, checkpoint_path: Optional[str], workers: int) -> Dict[str, Any]:
    overrides = {
        "collect_episodes_per_game": stage.episodes_per_game,
        "beam_width": stage.beam_width,
        "branch_factor": stage.branch_factor,
        "coord_budget": stage.coord_budget,
        "max_steps": stage.max_steps,
        "max_search_steps": stage.max_steps,
        "checkpoint": checkpoint_path,
        "workers": workers,
    }
    config = merge_config(base_profile, overrides)
    config["hardware_profile"] = base_profile
    config["checkpoint"] = checkpoint_path
    config["workers"] = workers
    return config


def ensure_game_stats(game_stats: Dict[str, Dict[str, Any]], game_id: str) -> Dict[str, Any]:
    if game_id not in game_stats:
        game_stats[game_id] = {
            "episodes": 0,
            "nonzero": 0,
            "wins": 0,
            "best_score": 0.0,
            "max_levels": 0,
            "best_total_delta_pixels": 0,
            "best_unique_frames": 0,
            "best_total_novelty": 0.0,
        }
    return game_stats[game_id]


def update_game_stats(game_stats: Dict[str, Dict[str, Any]], episode: Dict[str, Any]) -> None:
    game_id = str(episode["game_id"])
    stats = ensure_game_stats(game_stats, game_id)
    score = float(episode.get("score", 0.0))
    levels = int(episode.get("levels_completed", 0))
    total_delta = int(episode.get("total_delta_pixels", 0))
    unique_frames = int(episode.get("unique_frames", 0))
    total_novelty = float(episode.get("total_novelty", 0.0))

    stats["episodes"] += 1
    stats["nonzero"] += int(score > 0.0)
    stats["wins"] += int(str(episode.get("final_state")) == "WIN")
    stats["best_score"] = max(stats["best_score"], score)
    stats["max_levels"] = max(stats["max_levels"], levels)
    stats["best_total_delta_pixels"] = max(stats["best_total_delta_pixels"], total_delta)
    stats["best_unique_frames"] = max(stats["best_unique_frames"], unique_frames)
    stats["best_total_novelty"] = max(stats["best_total_novelty"], total_novelty)


def probe_transition_score(
    previous_frame: Sequence[Sequence[int]],
    next_frame: Sequence[Sequence[int]],
    state_name: str,
    levels_before: int,
    levels_after: int,
) -> Tuple[float, int]:
    level_gain = max(0, int(levels_after) - int(levels_before))
    delta_pixels = frame_delta(previous_frame, next_frame)
    score = (level_gain * 18.0) + min(delta_pixels / 20.0, 6.0)
    if delta_pixels > 0:
        score += 0.75
    else:
        score -= 0.5
    if state_name == "WIN":
        score += 30.0
    elif state_name == "GAME_OVER":
        score -= 8.0
    elif state_name == "NOT_FINISHED":
        score += 0.75
    return score, delta_pixels


def run_probe_stage(
    project_root: Path,
    output_root: Path,
    ordered_games: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    probe_priors: Dict[str, Dict[str, Any]] = {}
    probe_seeds = list(seeds[: min(3, len(seeds))]) or [0]
    arc = build_arcade(
        project_root=project_root,
        recordings_root=output_root / "recordings" / "probe",
        logger_name="arc_agi.collect.probe",
    )

    for game_id in ordered_games:
        action_totals: Dict[int, float] = {}
        action_counts: Dict[int, int] = {}
        action_gameovers: Dict[int, int] = {}
        coord_totals: Dict[Tuple[int, int], float] = {}
        coord_counts: Dict[Tuple[int, int], int] = {}
        total_trials = 0
        total_gameovers = 0

        for seed in probe_seeds:
            env = arc.make(game_id, seed=int(seed))
            if env is None or env.observation_space is None:
                continue
            raw_obs = env.observation_space
            initial_frame = final_subframe(raw_obs.frame)
            available_actions = [int(action) for action in getattr(raw_obs, "available_actions", [])]
            levels_before = int(raw_obs.levels_completed)

            def record_trial(action_id: int, action_data: Optional[Dict[str, int]]) -> None:
                nonlocal total_trials, total_gameovers
                trial_env = arc.make(game_id, seed=int(seed))
                if trial_env is None or trial_env.observation_space is None:
                    return
                trial_obs = trial_env.observation_space
                next_obs = trial_env.step(GameAction.from_id(action_id), data=action_data or {})
                if next_obs is None:
                    return
                next_frame = final_subframe(next_obs.frame)
                state_name = next_obs.state.name if hasattr(next_obs.state, "name") else str(next_obs.state)
                score, _ = probe_transition_score(
                    previous_frame=initial_frame,
                    next_frame=next_frame,
                    state_name=state_name,
                    levels_before=levels_before,
                    levels_after=int(next_obs.levels_completed),
                )
                action_totals[action_id] = action_totals.get(action_id, 0.0) + score
                action_counts[action_id] = action_counts.get(action_id, 0) + 1
                action_gameovers[action_id] = action_gameovers.get(action_id, 0) + int(state_name == "GAME_OVER")
                total_trials += 1
                total_gameovers += int(state_name == "GAME_OVER")
                if action_id == 6 and action_data is not None:
                    point = (int(action_data["x"]), int(action_data["y"]))
                    coord_totals[point] = coord_totals.get(point, 0.0) + score
                    coord_counts[point] = coord_counts.get(point, 0) + 1

            for action_id in available_actions:
                if action_id == 6:
                    for x, y in salient_points(initial_frame, prev_frame=None, budget=12):
                        record_trial(action_id=6, action_data={"x": int(x), "y": int(y)})
                else:
                    record_trial(action_id=action_id, action_data=None)

        action_scores = {
            str(action_id): (float(action_totals[action_id]) / float(action_counts[action_id]))
            for action_id in sorted(action_counts)
            if action_counts[action_id] > 0
        }
        risky_actions = [
            int(action_id)
            for action_id in sorted(action_counts)
            if action_counts[action_id] > 0
            and (float(action_gameovers.get(action_id, 0)) / float(action_counts[action_id])) >= 0.5
            and (float(action_totals[action_id]) / float(action_counts[action_id])) <= 0.0
        ]
        coord_rank = sorted(
            coord_totals.items(),
            key=lambda item: (float(item[1]) / float(coord_counts[item[0]]), float(item[1])),
            reverse=True,
        )
        coord_hint_points = [
            [int(point[0]), int(point[1])]
            for point, total in coord_rank
            if coord_counts[point] > 0 and (float(total) / float(coord_counts[point])) > 0.0
        ][:8]
        probe_signal = max((float(score) for score in action_scores.values()), default=0.0)
        if coord_rank:
            best_point, total = coord_rank[0]
            probe_signal = max(probe_signal, float(total) / float(coord_counts[best_point]))

        prior = {
            "action_scores": action_scores,
            "risky_actions": risky_actions,
            "coord_hint_points": coord_hint_points,
            "probe_signal": probe_signal,
            "game_over_ratio": (float(total_gameovers) / float(total_trials)) if total_trials > 0 else 1.0,
            "positive_actions": sum(1 for score in action_scores.values() if float(score) > 0.0),
            "probe_trials": total_trials,
            "probe_seeds": probe_seeds,
        }
        probe_priors[game_id] = prior
        print(
            "Probe %s signal=%.3f positive_actions=%d risky=%s coord_hints=%d"
            % (
                game_id,
                float(prior["probe_signal"]),
                int(prior["positive_actions"]),
                ",".join(str(action_id) for action_id in risky_actions) or "-",
                len(coord_hint_points),
            ),
            flush=True,
        )

    save_json(output_root / "probe_priors.json", {"updated_at_utc": utc_timestamp(), "game_priors": probe_priors})
    return probe_priors


def progress_scalar(game_id: str, game_stats: Dict[str, Dict[str, Any]]) -> float:
    stats = ensure_game_stats(game_stats, game_id)
    return (
        float(stats["max_levels"]) * 100.0
        + float(stats["best_score"]) * 24.0
        + float(stats["nonzero"]) * 4.0
        + float(stats["best_total_delta_pixels"]) / 24.0
        + float(stats["best_unique_frames"]) * 2.0
        + float(stats["best_total_novelty"]) * 16.0
        + float(stats["wins"]) * 1000.0
    )


def probe_scalar(game_id: str, game_priors: Dict[str, Dict[str, Any]]) -> float:
    prior = game_priors.get(game_id, {})
    return (
        float(prior.get("probe_signal", 0.0)) * 8.0
        + float(prior.get("positive_actions", 0)) * 1.5
        - float(prior.get("game_over_ratio", 1.0)) * 12.0
    )


def hybrid_priority(
    game_id: str,
    metadata_map: Dict[str, Dict[str, Any]],
    game_stats: Dict[str, Dict[str, Any]],
    game_priors: Dict[str, Dict[str, Any]],
) -> Tuple[float, float, float, float]:
    difficulty = curriculum_key(game_id, metadata_map[game_id])
    return (
        progress_scalar(game_id, game_stats),
        probe_scalar(game_id, game_priors),
        -difficulty[0],
        -difficulty[1],
    )


def select_games(
    ordered_games: Sequence[str],
    metadata_map: Dict[str, Dict[str, Any]],
    game_stats: Dict[str, Dict[str, Any]],
    game_priors: Dict[str, Dict[str, Any]],
    stage: StageSpec,
) -> List[str]:
    ranked_all = sorted(
        ordered_games,
        key=lambda game_id: hybrid_priority(game_id, metadata_map, game_stats, game_priors),
        reverse=True,
    )

    if stage.selector == "all":
        return ranked_all

    if stage.selector == "warmup":
        ranked_warmup = sorted(
            ordered_games,
            key=lambda game_id: (
                probe_scalar(game_id, game_priors),
                -curriculum_key(game_id, metadata_map[game_id])[0],
                -curriculum_key(game_id, metadata_map[game_id])[1],
            ),
            reverse=True,
        )
        return ranked_warmup[: stage.max_games]

    if stage.selector == "focus":
        progressed = [
            game_id
            for game_id in ranked_all
            if ensure_game_stats(game_stats, game_id)["best_score"] > 0.0
            or ensure_game_stats(game_stats, game_id)["max_levels"] > 0
        ]
        base = progressed or ranked_all
        return base[: stage.max_games]

    if stage.selector == "exploit":
        progressed = [
            game_id
            for game_id in ranked_all
            if ensure_game_stats(game_stats, game_id)["best_score"] > 0.0
            or ensure_game_stats(game_stats, game_id)["max_levels"] > 0
        ]
        if progressed:
            return progressed[: stage.max_games]
        return ranked_all[: stage.max_games]

    if stage.selector == "rescue":
        unresolved = [
            game_id
            for game_id in ordered_games
            if ensure_game_stats(game_stats, game_id)["best_score"] <= 0.0
            and ensure_game_stats(game_stats, game_id)["max_levels"] == 0
        ]
        ranked_rescue = sorted(
            unresolved or ordered_games,
            key=lambda game_id: (
                probe_scalar(game_id, game_priors),
                -float(game_priors.get(game_id, {}).get("game_over_ratio", 1.0)),
                -curriculum_key(game_id, metadata_map[game_id])[0],
            ),
            reverse=True,
        )
        return ranked_rescue[: stage.max_games]

    raise KeyError("Unknown stage selector: %s" % stage.selector)


def compute_stage_targets(
    selected_games: Sequence[str],
    metadata_map: Dict[str, Dict[str, Any]],
    game_stats: Dict[str, Dict[str, Any]],
    game_priors: Dict[str, Dict[str, Any]],
    stage: StageSpec,
) -> Dict[str, int]:
    targets = {game_id: int(stage.episodes_per_game) for game_id in selected_games}
    if not selected_games:
        return targets

    ranked = sorted(
        selected_games,
        key=lambda game_id: hybrid_priority(game_id, metadata_map, game_stats, game_priors),
        reverse=True,
    )

    if stage.selector == "focus":
        bonus = max(2, stage.episodes_per_game // 2)
        for game_id in ranked[: max(1, len(ranked) // 3)]:
            targets[game_id] += bonus
    elif stage.selector == "exploit":
        bonus = max(4, stage.episodes_per_game)
        for game_id in ranked[: max(1, len(ranked) // 2)]:
            targets[game_id] += bonus
    elif stage.selector == "rescue":
        floor = max(4, stage.episodes_per_game // 2)
        for game_id in selected_games:
            prior = game_priors.get(game_id, {})
            if float(prior.get("probe_signal", 0.0)) <= 0.0 and float(prior.get("game_over_ratio", 1.0)) >= 0.75:
                targets[game_id] = floor

    return targets


def save_stage_state(
    path: Path,
    stage_results: List[Dict[str, Any]],
    game_stats: Dict[str, Dict[str, Any]],
    game_priors: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "updated_at_utc": utc_timestamp(),
        "stage_results": stage_results,
        "game_stats": game_stats,
        "game_priors": game_priors,
    }
    save_json(path, payload)


def build_stage_tasks(
    project_root: Path,
    output_root: Path,
    selected_games: Sequence[str],
    metadata_map: Dict[str, Dict[str, Any]],
    game_priors: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    stage_index: int,
    stage: StageSpec,
    seeds: Sequence[int],
    episode_targets: Dict[str, int],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    max_target = max((episode_targets.get(game_id, 0) for game_id in selected_games), default=0)
    for generated in range(max_target):
        for game_index, game_id in enumerate(selected_games, start=1):
            if generated >= int(episode_targets.get(game_id, 0)):
                continue
            baseline_actions = metadata_map[game_id].get("baseline_actions", [])
            base_seed = int(seeds[(generated + game_index + stage_index) % len(seeds)])
            seed = base_seed + (generated // max(1, len(seeds))) * 1000 + stage_index * 100000
            per_game_config = dict(config)
            per_game_config["game_prior"] = dict(game_priors.get(game_id, {}))
            tasks.append(
                {
                    "project_root": str(project_root),
                    "output_root": str(output_root),
                    "game_id": game_id,
                    "game_index": game_index,
                    "episode_index": generated + 1,
                    "episode_id": "%s_%s_seed%d_try%d" % (stage.name, game_id, seed, generated),
                    "seed": seed,
                    "baseline_actions": list(baseline_actions),
                    "checkpoint_path": config.get("checkpoint"),
                    "config": per_game_config,
                }
            )
    return tasks


def summarize_stage_progress(
    episode: Dict[str, Any],
    stage_name: str,
    stage_completed: int,
    stage_total: int,
    overall_completed: int,
    overall_total: int,
    stage_start_time: float,
    overall_start_time: float,
) -> str:
    stage_elapsed = time.perf_counter() - stage_start_time
    overall_elapsed = time.perf_counter() - overall_start_time
    stage_avg = stage_elapsed / stage_completed if stage_completed else 0.0
    overall_avg = overall_elapsed / overall_completed if overall_completed else 0.0
    stage_eta = stage_avg * max(0, stage_total - stage_completed) if stage_completed else 0.0
    overall_eta = overall_avg * max(0, overall_total - overall_completed) if overall_completed else 0.0
    return (
        "[%s] saved %s state=%s levels=%d actions=%d score=%.3f | stage %d/%d eta %s | overall %d/%d eta %s"
        % (
            stage_name,
            episode["episode_id"],
            episode["final_state"],
            int(episode["levels_completed"]),
            int(episode["actions_taken"]),
            float(episode["score"]),
            stage_completed,
            stage_total,
            format_duration(stage_eta),
            overall_completed,
            overall_total,
            format_duration(overall_eta),
        )
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / "collected"
    episodes_path = run_dir / "episodes.jsonl.gz"
    metadata_map = load_metadata_map(project_root / "environment_files")
    requested_games = (
        sorted(metadata_map.keys(), key=lambda game_id: curriculum_key(game_id, metadata_map[game_id]))
        if not args.games
        else sorted(set(game.strip() for game in args.games.split(",") if game.strip()))
    )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    workers = max(1, int(args.workers))
    stages = build_stage_specs(args.checkpoint)

    save_json(
        output_root / "collect_staged_config.json",
        {
            "created_at_utc": utc_timestamp(),
            "hardware_profile": args.hardware_profile,
            "checkpoint": args.checkpoint,
            "workers": workers,
            "requested_games": requested_games,
            "seeds": seeds,
            "stages": [stage.__dict__ for stage in stages],
        },
    )

    overall_start_time = time.perf_counter()
    overall_completed = 0
    planned_total = 0
    game_stats: Dict[str, Dict[str, Any]] = {}
    stage_results: List[Dict[str, Any]] = []

    print("Starting probe stage for %d games across seeds %s" % (len(requested_games), ",".join(str(seed) for seed in seeds[:3] or seeds)), flush=True)
    game_priors = run_probe_stage(
        project_root=project_root,
        output_root=output_root,
        ordered_games=requested_games,
        seeds=seeds,
    )

    for stage_index, stage in enumerate(stages, start=1):
        selected_games = select_games(requested_games, metadata_map, game_stats, game_priors, stage)
        config = stage_config(args.hardware_profile, stage, args.checkpoint, workers)
        config["created_at_utc"] = utc_timestamp()
        config["stage_name"] = stage.name
        config["stage_index"] = stage_index

        episode_targets = compute_stage_targets(selected_games, metadata_map, game_stats, game_priors, stage)
        tasks = build_stage_tasks(
            project_root=project_root,
            output_root=output_root,
            selected_games=selected_games,
            metadata_map=metadata_map,
            game_priors=game_priors,
            config=config,
            stage_index=stage_index,
            stage=stage,
            seeds=seeds,
            episode_targets=episode_targets,
        )
        stage_total = len(tasks)
        planned_total += stage_total
        stage_completed = 0
        stage_start_time = time.perf_counter()

        print(
            "Starting stage %d/%d %s with %d games, beam=%d branch=%d coord=%d max_steps=%d"
            % (
                stage_index,
                len(stages),
                stage.name,
                len(selected_games),
                stage.beam_width,
                stage.branch_factor,
                stage.coord_budget,
                stage.max_steps,
            ),
            flush=True,
        )
        print("Selected games: %s" % ",".join(selected_games), flush=True)
        print(
            "Episode targets: %s"
            % ",".join("%s=%d" % (game_id, int(episode_targets[game_id])) for game_id in selected_games),
            flush=True,
        )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(collect_episode_task, task) for task in tasks]
            for future in as_completed(futures):
                episode = future.result()
                episode["stage_name"] = stage.name
                episode["stage_index"] = stage_index
                append_jsonl_gz(episodes_path, episode)
                update_game_stats(game_stats, episode)
                stage_completed += 1
                overall_completed += 1
                print(
                    summarize_stage_progress(
                        episode=episode,
                        stage_name=stage.name,
                        stage_completed=stage_completed,
                        stage_total=stage_total,
                        overall_completed=overall_completed,
                        overall_total=planned_total,
                        stage_start_time=stage_start_time,
                        overall_start_time=overall_start_time,
                    ),
                    flush=True,
                )

        stage_elapsed = time.perf_counter() - stage_start_time
        stage_results.append(
            {
                "name": stage.name,
                "games": list(selected_games),
                "episode_targets": dict(episode_targets),
                "beam_width": stage.beam_width,
                "branch_factor": stage.branch_factor,
                "coord_budget": stage.coord_budget,
                "max_steps": stage.max_steps,
                "completed": stage_completed,
                "elapsed_seconds": stage_elapsed,
            }
        )
        save_stage_state(output_root / "stage_state.json", stage_results, game_stats, game_priors)
        print("Finished stage %s in %s (%d episodes)" % (stage.name, format_duration(stage_elapsed), stage_completed), flush=True)

    total_elapsed = time.perf_counter() - overall_start_time
    print("Saved episodes to %s" % episodes_path, flush=True)
    print("Staged collection finished: %d episodes in %s" % (overall_completed, format_duration(total_elapsed)), flush=True)


if __name__ == "__main__":
    main()
