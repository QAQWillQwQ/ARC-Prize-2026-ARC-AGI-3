from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Pillow is required for inspect_collect.py. Activate the project .venv first."
    ) from exc


ARC_PALETTE: List[Tuple[int, int, int]] = [
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
]

MOTION_PURPLE: Tuple[int, int, int] = (220, 40, 255)
MOTION_PURPLE_SOFT: Tuple[int, int, int] = (170, 90, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and visualize collected ARC episodes.")
    parser.add_argument("--data", type=str, required=True, help="Path to episodes.jsonl.gz")
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument(
        "--match",
        type=str,
        default="any",
        choices=["any", "progress", "nonzero", "game_over", "not_finished"],
        help="Filter mode when selecting an episode by game/index.",
    )
    parser.add_argument("--index", type=int, default=0, help="Index within filtered episodes.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--case-name", type=str, default=None, help="Optional friendly output folder name.")
    parser.add_argument("--cell-size", type=int, default=10)
    parser.add_argument("--gif-ms", type=int, default=350)
    parser.add_argument("--gif-only", action="store_true", help="Only write episode.gif and summary json.")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def iter_episodes(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def episode_matches(episode: Dict[str, Any], game_id: Optional[str], match: str) -> bool:
    if game_id is not None and str(episode.get("game_id")) != game_id:
        return False
    if match == "any":
        return True
    if match == "progress":
        return float(episode.get("score", 0.0)) > 0.0 or int(episode.get("levels_completed", 0)) > 0
    if match == "nonzero":
        return float(episode.get("score", 0.0)) > 0.0
    if match == "game_over":
        return str(episode.get("final_state")) == "GAME_OVER"
    if match == "not_finished":
        return str(episode.get("final_state")) == "NOT_FINISHED"
    return False


def select_episode(
    path: Path,
    episode_id: Optional[str],
    game_id: Optional[str],
    match: str,
    index: int,
) -> Dict[str, Any]:
    if episode_id is not None:
        for episode in iter_episodes(path):
            if str(episode.get("episode_id")) == episode_id:
                return episode
        raise KeyError("Episode id %s not found in %s" % (episode_id, path))

    matches: List[Dict[str, Any]] = []
    for episode in iter_episodes(path):
        if episode_matches(episode, game_id=game_id, match=match):
            matches.append(episode)
    if not matches:
        raise RuntimeError("No episodes matched game_id=%s match=%s" % (game_id, match))
    if index < 0 or index >= len(matches):
        raise IndexError("Index %d is out of range for %d matched episodes" % (index, len(matches)))
    return matches[index]


def render_grid(frame: Sequence[Sequence[int]], cell_size: int) -> Image.Image:
    height = len(frame)
    width = len(frame[0]) if frame else 0
    image = Image.new("RGB", (width * cell_size, height * cell_size), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)
    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            color = ARC_PALETTE[int(value) % len(ARC_PALETTE)]
            draw.rectangle(
                [
                    x * cell_size,
                    y * cell_size,
                    (x + 1) * cell_size - 1,
                    (y + 1) * cell_size - 1,
                ],
                fill=color,
                outline=(35, 35, 35),
            )
    return image


def changed_points(prev_frame: Sequence[Sequence[int]], next_frame: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for y in range(min(len(prev_frame), len(next_frame))):
        for x in range(min(len(prev_frame[y]), len(next_frame[y]))):
            if int(prev_frame[y][x]) != int(next_frame[y][x]):
                points.append((x, y))
    return points


def infer_motion_centroid(points: Sequence[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    x_total = sum(point[0] for point in points)
    y_total = sum(point[1] for point in points)
    return (x_total / len(points), y_total / len(points))


def frame_signature(frame: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(frame, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def score_tag(score: Any) -> str:
    text = ("%.3f" % float(score)).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def safe_name(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    cleaned = "".join(char if char in allowed else "_" for char in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "episode"


def episode_file_stem(episode: Dict[str, Any]) -> str:
    parts = [
        str(episode.get("episode_id") or "episode"),
        str(episode.get("final_state") or "UNKNOWN"),
        "lvl%d" % int(episode.get("levels_completed", 0)),
        "score%s" % score_tag(episode.get("score", 0.0)),
        "act%d" % int(episode.get("actions_taken", 0)),
    ]
    return safe_name("__".join(parts))


def render_diff_overlay(
    next_frame: Sequence[Sequence[int]],
    prev_frame: Sequence[Sequence[int]],
    cell_size: int,
) -> Image.Image:
    image = render_grid(next_frame, cell_size)
    draw = ImageDraw.Draw(image)
    for x, y in changed_points(prev_frame, next_frame):
        draw.rectangle(
            [
                x * cell_size,
                y * cell_size,
                (x + 1) * cell_size - 1,
                (y + 1) * cell_size - 1,
            ],
            outline=(255, 255, 255),
            width=max(1, cell_size // 5),
        )
    return image


def render_motion_overlay(
    next_frame: Sequence[Sequence[int]],
    prev_frame: Sequence[Sequence[int]],
    cell_size: int,
    prior_centroids: Sequence[Tuple[float, float]],
) -> Image.Image:
    image = render_grid(next_frame, cell_size)
    draw = ImageDraw.Draw(image)
    points = changed_points(prev_frame, next_frame)
    if not points:
        return image

    x_values = [x for x, _ in points]
    y_values = [y for _, y in points]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    for x, y in points:
        draw.rectangle(
            [
                x * cell_size + 1,
                y * cell_size + 1,
                (x + 1) * cell_size - 2,
                (y + 1) * cell_size - 2,
            ],
            outline=MOTION_PURPLE_SOFT,
            width=max(1, cell_size // 6),
        )

    draw.rectangle(
        [
            min_x * cell_size,
            min_y * cell_size,
            (max_x + 1) * cell_size - 1,
            (max_y + 1) * cell_size - 1,
        ],
        outline=MOTION_PURPLE,
        width=max(2, cell_size // 4),
    )

    centroid = infer_motion_centroid(points)
    if centroid is not None:
        trace = list(prior_centroids) + [centroid]
        if len(trace) >= 2:
            draw.line(
                [(x * cell_size + cell_size / 2.0, y * cell_size + cell_size / 2.0) for x, y in trace],
                fill=MOTION_PURPLE,
                width=max(2, cell_size // 3),
            )
        cx = centroid[0] * cell_size + cell_size / 2.0
        cy = centroid[1] * cell_size + cell_size / 2.0
        radius = max(4, cell_size // 2)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(255, 255, 255),
            width=max(2, cell_size // 4),
            fill=MOTION_PURPLE,
        )
    return image


def panel_with_caption(image: Image.Image, caption: str) -> Image.Image:
    caption_height = 46
    out = Image.new("RGB", (image.width, image.height + caption_height), color=(245, 245, 245))
    out.paste(image, (0, caption_height))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), caption, fill=(0, 0, 0))
    return out


def render_transition_panel(
    transition: Dict[str, Any],
    cell_size: int,
    prior_centroids: Sequence[Tuple[float, float]],
) -> Image.Image:
    before = render_grid(transition["frame"], cell_size)
    after = render_grid(transition["next_frame"], cell_size)
    diff = render_diff_overlay(transition["next_frame"], transition["frame"], cell_size)
    motion = render_motion_overlay(
        transition["next_frame"],
        transition["frame"],
        cell_size,
        prior_centroids=prior_centroids,
    )

    before = panel_with_caption(before, "1. BEFORE\nframe at current step")
    after = panel_with_caption(after, "2. AFTER\nnext frame after chosen action")
    diff = panel_with_caption(diff, "3. CHANGED CELLS\nwhite boxes = pixels that changed")
    motion = panel_with_caption(motion, "4. INFERRED MOTION\nbright purple = estimated moving region/path")

    spacer = 16
    width = before.width + after.width + diff.width + motion.width + spacer * 3
    height = max(before.height, after.height, diff.height, motion.height)
    panel = Image.new("RGB", (width, height), color=(255, 255, 255))
    panel.paste(before, (0, 0))
    panel.paste(after, (before.width + spacer, 0))
    panel.paste(diff, (before.width + spacer + after.width + spacer, 0))
    panel.paste(
        motion,
        (before.width + spacer + after.width + spacer + diff.width + spacer, 0),
    )

    draw = ImageDraw.Draw(panel)
    meta = (
        "step=%d action=%d data=%s delta=%d novelty=%.3f levels %d->%d state %s->%s"
        % (
            int(transition.get("step_index", 0)),
            int(transition.get("action_id", 0)),
            json.dumps(transition.get("action_data", {}), ensure_ascii=True, separators=(",", ":")),
            int(transition.get("delta_pixels", 0)),
            float(transition.get("novelty", 0.0)),
            int(transition.get("levels_before", 0)),
            int(transition.get("levels_after", 0)),
            str(transition.get("state_before", "")),
            str(transition.get("state_after", "")),
        )
    )
    draw.rectangle([0, 0, width, 22], fill=(230, 230, 230))
    draw.text((6, 4), meta, fill=(0, 0, 0))
    return panel


def write_html_summary(
    output_dir: Path,
    episode: Dict[str, Any],
    frame_paths: Sequence[Path],
    gif_name: str,
    html_name: str,
) -> None:
    rows: List[str] = []
    transitions = list(episode.get("transitions", []))
    for transition, image_path in zip(transitions, frame_paths):
        rows.append(
            "<tr>"
            "<td>%d</td><td>%d</td><td>%s</td><td>%d</td><td>%.3f</td><td>%d→%d</td><td>%s→%s</td>"
            "<td><img src='%s' style='max-width:1200px;border:1px solid #ccc;'></td>"
            "</tr>"
            % (
                int(transition.get("step_index", 0)),
                int(transition.get("action_id", 0)),
                html.escape(json.dumps(transition.get("action_data", {}), ensure_ascii=True)),
                int(transition.get("delta_pixels", 0)),
                float(transition.get("novelty", 0.0)),
                int(transition.get("levels_before", 0)),
                int(transition.get("levels_after", 0)),
                html.escape(str(transition.get("state_before", ""))),
                html.escape(str(transition.get("state_after", ""))),
                html.escape(image_path.name),
            )
        )

    payload = {
        key: episode.get(key)
        for key in ["episode_id", "game_id", "seed", "final_state", "levels_completed", "actions_taken", "score"]
    }
    html_text = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Collect Episode Inspection</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%%; }
    th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
    th { background: #f4f4f4; }
    code { background: #f6f8fa; padding: 2px 4px; }
  </style>
</head>
<body>
  <h1>Collect Episode Inspection</h1>
  <pre>%s</pre>
  <p>GIF: <a href="%s">%s</a></p>
  <table>
    <thead>
      <tr>
        <th>step</th><th>action</th><th>data</th><th>delta</th><th>novelty</th><th>levels</th><th>state</th><th>visual</th>
      </tr>
    </thead>
    <tbody>
      %s
    </tbody>
  </table>
</body>
</html>
""" % (
        html.escape(json.dumps(payload, indent=2, ensure_ascii=True)),
        html.escape(gif_name),
        html.escape(gif_name),
        "\n".join(rows),
    )
    (output_dir / html_name).write_text(html_text, encoding="utf-8")


def write_trace_csv(output_dir: Path, episode: Dict[str, Any], trace_name: str) -> None:
    transitions = list(episode.get("transitions", []))
    rows: List[Dict[str, Any]] = []
    seen_after: Dict[str, int] = {}
    prior_after_signature: Optional[str] = None

    for transition in transitions:
        before_frame = transition["frame"]
        after_frame = transition["next_frame"]
        points = changed_points(before_frame, after_frame)
        centroid = infer_motion_centroid(points)
        before_signature = frame_signature(before_frame)
        after_signature = frame_signature(after_frame)
        seen_after_count = seen_after.get(after_signature, 0)
        seen_after[after_signature] = seen_after_count + 1
        rows.append(
            {
                "step_index": int(transition.get("step_index", 0)),
                "action_id": int(transition.get("action_id", 0)),
                "action_data": json.dumps(transition.get("action_data", {}), ensure_ascii=True, separators=(",", ":")),
                "delta_pixels": int(transition.get("delta_pixels", 0)),
                "novelty": float(transition.get("novelty", 0.0)),
                "levels_before": int(transition.get("levels_before", 0)),
                "levels_after": int(transition.get("levels_after", 0)),
                "state_before": str(transition.get("state_before", "")),
                "state_after": str(transition.get("state_after", "")),
                "changed_cells": len(points),
                "motion_centroid_x": "" if centroid is None else "%.3f" % centroid[0],
                "motion_centroid_y": "" if centroid is None else "%.3f" % centroid[1],
                "frame_before_sig": before_signature,
                "frame_after_sig": after_signature,
                "after_seen_count_before_step": seen_after_count,
                "returned_to_immediate_previous_frame": int(prior_after_signature == after_signature),
            }
        )
        prior_after_signature = after_signature

    fieldnames = [
        "step_index",
        "action_id",
        "action_data",
        "delta_pixels",
        "novelty",
        "levels_before",
        "levels_after",
        "state_before",
        "state_after",
        "changed_cells",
        "motion_centroid_x",
        "motion_centroid_y",
        "frame_before_sig",
        "frame_after_sig",
        "after_seen_count_before_step",
        "returned_to_immediate_previous_frame",
    ]
    with (output_dir / trace_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_action_sequence(output_dir: Path, episode: Dict[str, Any], actions_name: str) -> None:
    transitions = list(episode.get("transitions", []))
    action_ids = [str(int(transition.get("action_id", 0))) for transition in transitions]
    lines = [
        "episode_id=%s" % str(episode.get("episode_id", "")),
        "game_id=%s" % str(episode.get("game_id", "")),
        "final_state=%s" % str(episode.get("final_state", "")),
        "levels_completed=%d" % int(episode.get("levels_completed", 0)),
        "score=%.3f" % float(episode.get("score", 0.0)),
        "actions_taken=%d" % int(episode.get("actions_taken", 0)),
        "action_sequence=%s" % ",".join(action_ids),
    ]
    (output_dir / actions_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_episode_visuals(
    episode: Dict[str, Any],
    output_dir: Path,
    cell_size: int,
    gif_ms: int,
    gif_only: bool,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transitions = list(episode.get("transitions", []))
    if not transitions:
        raise RuntimeError("Episode %s has no transitions" % episode.get("episode_id"))

    stem = episode_file_stem(episode)
    gif_name = "%s.gif" % stem
    html_name = "%s.html" % stem
    summary_name = "%s_summary.json" % stem
    trace_name = "%s_trace.csv" % stem
    actions_name = "%s_actions.txt" % stem

    frames: List[Image.Image] = []
    frame_paths: List[Path] = []
    centroids: List[Tuple[float, float]] = []
    for idx, transition in enumerate(transitions):
        panel = render_transition_panel(transition, cell_size=cell_size, prior_centroids=centroids[-8:])
        if not gif_only:
            frame_path = output_dir / ("step_%03d.png" % idx)
            panel.save(frame_path)
            frame_paths.append(frame_path)
        frames.append(panel)
        centroid = infer_motion_centroid(changed_points(transition["frame"], transition["next_frame"]))
        if centroid is not None:
            centroids.append(centroid)

    gif_path = output_dir / gif_name
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(80, gif_ms),
        loop=0,
    )
    if not gif_only:
        write_html_summary(
            output_dir=output_dir,
            episode=episode,
            frame_paths=frame_paths,
            gif_name=gif_name,
            html_name=html_name,
        )
    write_trace_csv(output_dir=output_dir, episode=episode, trace_name=trace_name)
    write_action_sequence(output_dir=output_dir, episode=episode, actions_name=actions_name)
    (output_dir / summary_name).write_text(json.dumps(episode, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "gif": gif_path,
        "html": output_dir / html_name,
        "summary": output_dir / summary_name,
        "trace": output_dir / trace_name,
        "actions": output_dir / actions_name,
    }


def print_episode_summary(episode: Dict[str, Any]) -> None:
    transitions = list(episode.get("transitions", []))
    print("episode_id =", episode.get("episode_id"))
    print("game_id =", episode.get("game_id"))
    print("seed =", episode.get("seed"))
    print("final_state =", episode.get("final_state"))
    print("levels_completed =", episode.get("levels_completed"))
    print("score =", episode.get("score"))
    print("actions_taken =", episode.get("actions_taken"))
    print("num_transitions =", len(transitions))
    if transitions:
        deltas = [int(transition.get("delta_pixels", 0)) for transition in transitions]
        novelties = [float(transition.get("novelty", 0.0)) for transition in transitions]
        print("avg_delta_pixels = %.2f" % (sum(deltas) / len(deltas)))
        print("avg_novelty = %.4f" % (sum(novelties) / len(novelties)))
        print("max_delta_pixels =", max(deltas))


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    episode = select_episode(
        path=data_path,
        episode_id=args.episode_id,
        game_id=args.game_id,
        match=args.match,
        index=int(args.index),
    )
    print_episode_summary(episode)
    if args.summary_only:
        return
    episode_dir = output_dir / str(args.case_name or episode.get("episode_id") or "episode")
    saved_paths = save_episode_visuals(
        episode=episode,
        output_dir=episode_dir,
        cell_size=max(4, int(args.cell_size)),
        gif_ms=int(args.gif_ms),
        gif_only=bool(args.gif_only),
    )
    print("saved_visuals =", episode_dir)
    print("gif =", saved_paths["gif"])
    print("trace =", saved_paths["trace"])
    print("actions =", saved_paths["actions"])
    print("summary =", saved_paths["summary"])
    if not args.gif_only:
        print("html =", saved_paths["html"])


if __name__ == "__main__":
    main()
