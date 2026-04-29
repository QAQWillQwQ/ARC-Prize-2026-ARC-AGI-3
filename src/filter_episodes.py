from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import iter_jsonl_gz, save_json, write_jsonl_gz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter training episodes from one or more .jsonl.gz files.")
    parser.add_argument("--input", type=str, required=True, help="One or more comma separated input .jsonl.gz paths.")
    parser.add_argument("--output", type=str, required=True, help="Output .jsonl.gz path.")
    parser.add_argument("--games", type=str, default=None, help="Optional comma separated game ids to keep.")
    parser.add_argument("--min-levels", type=int, default=0, help="Keep only episodes with levels_completed >= this threshold.")
    parser.add_argument("--min-score", type=float, default=None, help="Optional minimum episode score.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of episodes to keep after filtering.")
    return parser.parse_args()


def parse_paths(raw: str) -> List[Path]:
    paths = [Path(part.strip()).expanduser().resolve() for part in raw.split(",") if part.strip()]
    if not paths:
        raise RuntimeError("No valid --input paths were provided.")
    return paths


def parse_games(raw: Optional[str]) -> Optional[set[str]]:
    if raw is None:
        return None
    games = {part.strip() for part in raw.split(",") if part.strip()}
    return games or None


def iter_episodes(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Input episodes file not found: %s" % path)
        yield from iter_jsonl_gz(path)


def main() -> None:
    args = parse_args()
    input_paths = parse_paths(args.input)
    output_path = Path(args.output).expanduser().resolve()
    requested_games = parse_games(args.games)
    min_levels = int(args.min_levels)
    min_score = None if args.min_score is None else float(args.min_score)
    limit = None if args.limit is None else max(0, int(args.limit))

    kept_rows: List[Dict[str, Any]] = []
    seen = 0
    kept = 0
    kept_games: Dict[str, int] = {}
    for episode in iter_episodes(input_paths):
        seen += 1
        game_id = str(episode.get("game_id", ""))
        if requested_games and game_id not in requested_games:
            continue
        if int(episode.get("levels_completed", 0)) < min_levels:
            continue
        if min_score is not None and float(episode.get("score", 0.0)) < min_score:
            continue
        kept_rows.append(episode)
        kept += 1
        kept_games[game_id] = kept_games.get(game_id, 0) + 1
        if limit is not None and kept >= limit:
            break
        if kept % 25 == 0:
            print(
                "[filter-episodes] kept=%d seen=%d games=%s"
                % (kept, seen, ",".join("%s:%d" % (k, kept_games[k]) for k in sorted(kept_games))),
                flush=True,
            )

    if not kept_rows:
        raise RuntimeError("No episodes matched the requested filter.")

    write_jsonl_gz(output_path, kept_rows)
    summary = {
        "input": [str(path) for path in input_paths],
        "output": str(output_path),
        "requested_games": None if requested_games is None else sorted(requested_games),
        "min_levels": min_levels,
        "min_score": min_score,
        "limit": limit,
        "episodes_seen": seen,
        "episodes_written": kept,
        "games": kept_games,
    }
    save_json(output_path.with_suffix(output_path.suffix + ".summary.json"), summary)
    print(
        "[filter-episodes] wrote=%d seen=%d output=%s"
        % (kept, seen, output_path),
        flush=True,
    )


if __name__ == "__main__":
    main()
