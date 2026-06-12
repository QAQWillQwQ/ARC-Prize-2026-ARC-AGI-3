from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import sys
import time
import traceback
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARC_PALETTE = np.array(
    [
        (0, 0, 0),
        (0, 116, 217),
        (255, 65, 54),
        (46, 204, 64),
        (255, 220, 0),
        (170, 170, 170),
        (240, 18, 190),
        (255, 133, 27),
        (127, 219, 255),
        (135, 12, 37),
        (177, 13, 201),
        (1, 255, 112),
        (133, 20, 75),
        (57, 204, 204),
        (255, 255, 255),
        (127, 127, 127),
    ],
    dtype=np.uint8,
)


SMOKE_GAMES = {"g50t", "lp85", "r11l", "su15", "vc33"}


def default_games(project_root: Path) -> list[str]:
    return sorted(p.name for p in (project_root / "environment_files").iterdir() if p.is_dir())


def latest_jihang_dir(project_root: Path) -> Path:
    candidates = sorted(
        (project_root / "Local_Output" / "Eval").glob("Jihang_forge_v3_score0p27_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if (path / "my_agent.py").is_file() and (path / "agents" / "agent.py").is_file():
            return path
    raise FileNotFoundError("No Jihang run directory with my_agent.py and agents shim was found.")


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.split(".")[-1]


def action_id(value: Any) -> int:
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except Exception:
        return -1


def frame_array(frame_data: Any) -> np.ndarray:
    frame = getattr(frame_data, "frame", frame_data)
    arr = np.array(frame, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape={arr.shape}")
    return arr


def action_data_dict(action: Any) -> dict[str, Any]:
    data = getattr(action, "action_data", None)
    if data is None:
        return {}
    try:
        return dict(data.model_dump())
    except Exception:
        try:
            return dict(data)
        except Exception:
            return {}


def small_reasoning(reasoning: Any) -> str:
    if isinstance(reasoning, dict):
        pieces = []
        for key in ("mode", "act", "eps", "buf"):
            if key in reasoning:
                pieces.append(f"{key}={reasoning[key]}")
        return " ".join(pieces)[:110]
    return str(reasoning)[:110] if reasoning is not None else ""


def safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return str(value)


def text_line(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text[:170], fill=fill, font=ImageFont.load_default())


def draw_grid(frame: np.ndarray, cell_size: int) -> Image.Image:
    indexed = np.clip(frame, 0, len(ARC_PALETTE) - 1)
    rgb = ARC_PALETTE[indexed]
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((frame.shape[1] * cell_size, frame.shape[0] * cell_size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(0, w + 1, cell_size * 8):
        draw.line([(x, 0), (x, h)], fill=(45, 45, 45), width=1)
    for y in range(0, h + 1, cell_size * 8):
        draw.line([(0, y), (w, y)], fill=(45, 45, 45), width=1)
    return img


def draw_click_marker(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int] | None,
    cell_size: int,
    header_h: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if point is None:
        return
    x, y = point
    cx = x * cell_size + cell_size // 2
    cy = header_h + y * cell_size + cell_size // 2
    r = max(5, cell_size + 2)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=3)
    draw.line((cx - r, cy, cx + r, cy), fill=color, width=2)
    draw.line((cx, cy - r, cx, cy + r), fill=color, width=2)
    draw.text((cx + r + 2, max(header_h, cy - r)), label, fill=color, font=ImageFont.load_default())


def changed_box(prev_frame: np.ndarray, next_frame: np.ndarray) -> tuple[int, int, int, int] | None:
    diff = prev_frame != next_frame
    if not np.any(diff):
        return None
    ys, xs = np.where(diff)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def same_click_payload(actual: Any, intended: Any) -> bool:
    """Compare click coordinates while ignoring harmless metadata fields."""
    actual = actual or {}
    intended = intended or {}
    if not intended:
        return True
    try:
        return int(actual.get("x", -999)) == int(intended.get("x", -998)) and int(actual.get("y", -999)) == int(intended.get("y", -998))
    except Exception:
        return False


def render_panel(step: dict[str, Any], total_steps: int, cell_size: int) -> Image.Image:
    frame = np.array(step["after_frame"], dtype=np.int64)
    before = np.array(step["before_frame"], dtype=np.int64)
    grid = draw_grid(frame, cell_size=cell_size)
    header_h = 96
    out = Image.new("RGB", (grid.width, grid.height + header_h), color=(18, 18, 18))
    out.paste(grid, (0, header_h))
    draw = ImageDraw.Draw(out)

    actual_point = None
    intended_point = None
    if step["action_id"] == 6:
        actual = step.get("actual_data") or {}
        intended = step.get("intended_data") or {}
        if "x" in actual and "y" in actual:
            actual_point = (int(actual["x"]), int(actual["y"]))
        if "x" in intended and "y" in intended:
            intended_point = (int(intended["x"]), int(intended["y"]))

    box = changed_box(before, frame)
    if box is not None:
        x0, y0, x1, y1 = box
        draw.rectangle(
            (
                x0 * cell_size,
                header_h + y0 * cell_size,
                (x1 + 1) * cell_size - 1,
                header_h + (y1 + 1) * cell_size - 1,
            ),
            outline=(220, 40, 255),
            width=2,
        )

    draw_click_marker(draw, actual_point, cell_size, header_h, (255, 60, 60), "actual")
    if intended_point is not None and intended_point != actual_point:
        draw_click_marker(draw, intended_point, cell_size, header_h, (80, 220, 255), "intended")

    data_text = ""
    if step["action_id"] == 6:
        data_text = f" actual={step.get('actual_data')} intended={step.get('intended_data')}"
    text_line(
        draw,
        (8, 6),
        f"{step['game_id']} | step {step['step']:03d}/{total_steps} | {step['action_name']}{data_text}",
        (245, 245, 245),
    )
    text_line(
        draw,
        (8, 30),
        f"levels {step['levels_before']} -> {step['levels_after']} | state {step['state_before']} -> {step['state_after']} | delta={step['delta_pixels']}",
        (210, 210, 210),
    )
    text_line(draw, (8, 54), f"reason: {step.get('reasoning_short', '')}", (180, 220, 255))
    if step.get("note"):
        text_line(draw, (8, 76), f"note: {step['note']}", (255, 210, 140))
    return out


def select_gif_steps(steps: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    if len(steps) <= max_frames:
        return steps
    important = {0, len(steps) - 1}
    for idx, step in enumerate(steps):
        if step["levels_after"] != step["levels_before"]:
            important.add(idx)
        if step["state_after"] in {"GAME_OVER", "WIN"}:
            important.add(idx)
        if step["delta_pixels"] >= 128:
            important.add(idx)
        if step["action_id"] == 6 and not same_click_payload(step.get("actual_data"), step.get("intended_data")):
            important.add(idx)
    remaining = max(1, max_frames - len(important))
    for raw in np.linspace(0, len(steps) - 1, remaining):
        important.add(int(round(float(raw))))
    return [steps[idx] for idx in sorted(important)[:max_frames]]


def filename_stem(game_id: str, levels: int, actions: int) -> str:
    return f"L{levels}_{game_id}_A{actions}"


def summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(step["action_name"] for step in steps)
    click_steps = [step for step in steps if step["action_id"] == 6]
    actual_clicks = Counter(
        (int((step.get("actual_data") or {}).get("x", -1)), int((step.get("actual_data") or {}).get("y", -1)))
        for step in click_steps
    )
    intended_clicks = Counter(
        (int((step.get("intended_data") or {}).get("x", -1)), int((step.get("intended_data") or {}).get("y", -1)))
        for step in click_steps
    )
    zero_delta = sum(1 for step in steps if int(step["delta_pixels"]) == 0)
    progress_steps = [
        step["step"] for step in steps if int(step["levels_after"]) > int(step["levels_before"])
    ]
    state_counts = Counter(step["state_after"] for step in steps)
    click_mismatch = sum(
        1 for step in click_steps if not same_click_payload(step.get("actual_data"), step.get("intended_data"))
    )
    return {
        "num_steps": len(steps),
        "action_counts": dict(action_counts),
        "zero_delta_steps": zero_delta,
        "zero_delta_ratio": round(zero_delta / max(1, len(steps)), 4),
        "progress_steps": progress_steps,
        "state_counts": dict(state_counts),
        "num_click_steps": len(click_steps),
        "click_data_mismatch_steps": click_mismatch,
        "top_actual_clicks": [
            {"x": xy[0], "y": xy[1], "count": count}
            for xy, count in actual_clicks.most_common(8)
        ],
        "top_intended_clicks": [
            {"x": xy[0], "y": xy[1], "count": count}
            for xy, count in intended_clicks.most_common(8)
        ],
    }


def run_one(
    project_root: str,
    agent_dir: str,
    output_dir: str,
    game_id: str,
    max_actions: int,
    max_gif_frames: int,
    gif_ms: int,
    cell_size: int,
    q: Any,
) -> None:
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        project = Path(project_root)
        out = Path(output_dir)
        agent_path = Path(agent_dir)
        os.environ["ARC_LOCAL_ENV_BASE"] = str(project / "environment_files")
        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

        sys.path.insert(0, str(agent_path))
        sys.path.insert(1, str(project / "ARC-AGI-3-Agents"))

        from arc_agi import Arcade, OperationMode
        from arcengine import GameState
        from my_agent import MyAgent

        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(project / "environment_files"),
            recordings_dir=str(out / "recordings" / game_id),
        )
        env_info = next(item for item in arcade.get_environments() if item.game_id.split("-", 1)[0] == game_id)
        card_id = arcade.open_scorecard(tags=["jihang_forge_v3_visual_replay", game_id])
        env = arcade.make(env_info.game_id, scorecard_id=card_id)
        if env is None:
            raise RuntimeError(f"Unable to make env for {game_id}")

        agent = MyAgent(
            card_id=card_id,
            game_id=env_info.game_id,
            agent_name="jihang_forge_v3_visual_replay",
            ROOT_URL="offline",
            record=False,
            arc_env=env,
            tags=["jihang_forge_v3_visual_replay"],
        )
        agent.MAX_ACTIONS = int(max_actions)
        agent.action_counter = 0
        agent.timer = time.time()

        steps: list[dict[str, Any]] = []
        started = time.time()
        while (
            not agent.is_done(agent.frames, agent.frames[-1])
            and agent.action_counter <= agent.MAX_ACTIONS
        ):
            latest = agent._convert_raw_frame_data(env.observation_space)
            before_frame = frame_array(latest)
            before_state = enum_name(latest.state)
            before_levels = int(getattr(latest, "levels_completed", 0))
            action = agent.choose_action(agent.frames, latest)
            actual_data = action_data_dict(action)
            intended_data = dict(getattr(agent, "_last_action_data", {}) or {})
            reasoning = safe_json(getattr(action, "reasoning", ""))
            note = ""
            if action_id(action) == 6 and intended_data and not same_click_payload(actual_data, intended_data):
                note = "click data mismatch: env will use actual data"

            frame = agent.take_action(action)
            if frame is not None:
                after_frame = frame_array(frame)
                after_state = enum_name(frame.state)
                after_levels = int(getattr(frame, "levels_completed", before_levels))
                agent.append_frame(frame)
            else:
                after_frame = before_frame.copy()
                after_state = "NO_FRAME"
                after_levels = before_levels

            steps.append(
                {
                    "game_id": game_id,
                    "full_game_id": env_info.game_id,
                    "step": int(agent.action_counter + 1),
                    "action_id": int(action_id(action)),
                    "action_name": enum_name(action),
                    "actual_data": safe_json(actual_data),
                    "intended_data": safe_json(intended_data),
                    "reasoning": reasoning,
                    "reasoning_short": small_reasoning(reasoning),
                    "levels_before": before_levels,
                    "levels_after": after_levels,
                    "state_before": before_state,
                    "state_after": after_state,
                    "delta_pixels": int(np.sum(before_frame != after_frame)),
                    "before_hash": hashlib.md5(before_frame.tobytes()).hexdigest()[:16],
                    "after_hash": hashlib.md5(after_frame.tobytes()).hexdigest()[:16],
                    "note": note,
                    "before_frame": before_frame.tolist(),
                    "after_frame": after_frame.tolist(),
                }
            )
            agent.action_counter += 1

            if after_state == enum_name(GameState.WIN):
                break

        scorecard = arcade.close_scorecard(card_id)
        payload = scorecard.model_dump() if scorecard is not None else {}
        env_rows = payload.get("environments", [])
        env_score = env_rows[0] if env_rows else {}
        run_row = (env_score.get("runs", [{}])[0] or {}) if env_score else {}

        levels = int(env_score.get("levels_completed", steps[-1]["levels_after"] if steps else 0))
        actions = int(env_score.get("actions", len(steps)))
        state = enum_name(run_row.get("state", steps[-1]["state_after"] if steps else "UNKNOWN"))
        score = float(env_score.get("score", 0.0))
        stem = filename_stem(game_id, levels, actions)
        gif_path = out / f"{stem}.gif"
        trace_path = out / f"{stem}_trace.json"
        summary_path = out / f"{stem}_summary.json"

        gif_steps = select_gif_steps(steps, max_frames=max_gif_frames)
        panels = [render_panel(step, total_steps=len(steps), cell_size=cell_size) for step in gif_steps]
        if panels:
            panels[0].save(
                gif_path,
                save_all=True,
                append_images=panels[1:],
                duration=max(80, int(gif_ms)),
                loop=0,
                optimize=True,
            )

        compact_steps = []
        for step in steps:
            row = dict(step)
            row.pop("before_frame", None)
            row.pop("after_frame", None)
            compact_steps.append(row)

        summary = {
            "game_id": game_id,
            "full_game_id": env_info.game_id,
            "status": "ok",
            "elapsed_seconds": round(time.time() - started, 3),
            "score": score,
            "levels_completed": levels,
            "actions": actions,
            "state": state,
            "max_actions": max_actions,
            "gif": str(gif_path),
            "trace": str(trace_path),
            "scorecard": payload,
            "behavior": summarize_steps(steps),
        }
        trace_path.write_text(json.dumps(compact_steps, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        q.put(summary)
    except Exception as exc:
        q.put(
            {
                "game_id": game_id,
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Jihang forge v3 and render one annotated GIF per game.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--agent-run-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", default=None)
    parser.add_argument("--max-actions", type=int, default=240)
    parser.add_argument("--smoke-max-actions", type=int, default=160)
    parser.add_argument("--per-game-timeout", type=int, default=900)
    parser.add_argument("--max-gif-frames", type=int, default=170)
    parser.add_argument("--gif-ms", type=int, default=180)
    parser.add_argument("--cell-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(args.project_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    agent_dir = Path(args.agent_run_dir).resolve() if args.agent_run_dir else latest_jihang_dir(project)
    selected = default_games(project) if not args.games else [item.strip() for item in args.games.split(",") if item.strip()]

    print(f"[gif] output_dir={out}", flush=True)
    print(f"[gif] agent_run_dir={agent_dir}", flush=True)
    print(f"[gif] games={','.join(selected)}", flush=True)
    print(
        f"[gif] max_actions={args.max_actions} smoke_max_actions={args.smoke_max_actions} timeout={args.per_game_timeout}s",
        flush=True,
    )

    ctx = get_context("spawn")
    results: list[dict[str, Any]] = []
    started_all = time.time()
    for index, game_id in enumerate(selected, start=1):
        max_actions = args.smoke_max_actions if game_id in SMOKE_GAMES else args.max_actions
        print(f"\n[{index:02d}/{len(selected):02d}] replay start game={game_id} max_actions={max_actions}", flush=True)
        q = ctx.Queue()
        proc = ctx.Process(
            target=run_one,
            args=(
                str(project),
                str(agent_dir),
                str(out),
                game_id,
                max_actions,
                int(args.max_gif_frames),
                int(args.gif_ms),
                int(args.cell_size),
                q,
            ),
        )
        started = time.time()
        proc.start()
        proc.join(args.per_game_timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(10)
            row = {
                "game_id": game_id,
                "status": "timeout",
                "elapsed_seconds": round(time.time() - started, 3),
                "score": 0.0,
                "levels_completed": 0,
                "actions": max_actions,
                "state": "TIMEOUT",
            }
        else:
            try:
                row = q.get_nowait()
            except queue.Empty:
                row = {
                    "game_id": game_id,
                    "status": "no_result",
                    "elapsed_seconds": round(time.time() - started, 3),
                    "score": 0.0,
                    "levels_completed": 0,
                    "actions": 0,
                    "state": "NO_RESULT",
                }
        results.append(row)
        short = {
            "game_id": row.get("game_id"),
            "status": row.get("status"),
            "score": row.get("score", 0.0),
            "levels_completed": row.get("levels_completed", 0),
            "actions": row.get("actions", 0),
            "state": row.get("state", ""),
            "elapsed_seconds": row.get("elapsed_seconds", 0.0),
            "gif": row.get("gif", ""),
        }
        print(f"[{index:02d}/{len(selected):02d}] replay done {json.dumps(short, ensure_ascii=False)}", flush=True)
        (out / "visual_replay_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_results = [row for row in results if row.get("status") == "ok"]
    summary = {
        "run_name": out.name,
        "num_games": len(results),
        "ok_games": len(ok_results),
        "mean_score": sum(float(row.get("score", 0.0)) for row in ok_results) / max(1, len(ok_results)),
        "mean_levels_completed": sum(float(row.get("levels_completed", 0)) for row in ok_results) / max(1, len(ok_results)),
        "solved_games": [row["game_id"] for row in ok_results if int(row.get("levels_completed", 0)) > 0],
        "elapsed_seconds": round(time.time() - started_all, 3),
        "output_dir": str(out),
        "agent_run_dir": str(agent_dir),
    }
    (out / "visual_replay_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[summary]", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
