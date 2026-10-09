"""Offline replay/audit of a runner-produced run directory.

Rebuilds each episode from the saved manifest, replays the recorded actions, and
rejects the run on the first mismatch: altered actions, altered requests,
altered after-states, wrong terminal reasons/counts, or a shortest-path length
that does not match the manifest. Exit code 0 means the run is fully
reproducible from its manifest; non-zero names the first failure.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from .maze import Maze, bfs_path


class AuditFailure(Exception):
    pass


def _fail(message):
    raise AuditFailure(message)


def audit(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "maze-manifest.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    attempt_cap = manifest["attempt_cap"]
    layouts = {m["maze_id"]: m for m in manifest["mazes"]}

    for maze_id, layout in layouts.items():
        walls = {tuple(w) for w in layout["walls"]}
        if [tuple(w) for w in sorted(walls)] != [tuple(w) for w in layout["walls"]]:
            _fail(f"{maze_id}: walls are not in canonical sorted order")
        path = bfs_path(layout["start"], layout["goal"], layout["walls"],
                        layout["width"], layout["height"])
        if path is None:
            _fail(f"{maze_id}: goal unreachable from start")
        if len(path) - 1 != layout["shortest_path_length"]:
            _fail(f"{maze_id}: recorded shortest_path_length {layout['shortest_path_length']} "
                  f"!= recomputed {len(path) - 1}")

    rows_by_episode = {}
    for line in (run_dir / "decisions.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (row["maze_id"], row["policy"], row["sequence"])
        rows_by_episode.setdefault(key, []).append(row)

    summaries = {}
    for row in csv.DictReader((run_dir / "rounds.csv").read_text().splitlines()):
        summaries[(row["maze_id"], row["policy"], int(row["sequence"]))] = row

    if set(rows_by_episode) != set(summaries):
        _fail("decisions.jsonl episodes do not match rounds.csv episodes")

    for (maze_id, policy, sequence), rows in sorted(rows_by_episode.items()):
        layout = layouts.get(maze_id)
        if layout is None:
            _fail(f"episode {sequence}: unknown maze_id {maze_id}")
        maze = Maze(layout, attempt_cap)
        for row in rows:
            if row["attempt"] != maze.attempts + 1:
                _fail(f"episode {sequence}: attempt out of order at {row['attempt']}")
            if row["request"] != maze.request():
                _fail(f"episode {sequence}: request mismatch at attempt {row['attempt']} "
                      f"(altered state or layout)")
            maze.apply(row["action"])
            if row["after"] != maze.snapshot():
                _fail(f"episode {sequence}: after-state mismatch at attempt {row['attempt']} "
                      f"(altered action or outcome)")
        if not maze.done:
            _fail(f"episode {sequence}: log ends before termination")
        summary = summaries[(maze_id, policy, sequence)]
        for field in ("reached", "attempts", "invalid_moves"):
            logged = summary[field]
            logged = logged == "True" if field == "reached" else int(logged)
            actual = getattr(maze, field)
            if logged != actual:
                _fail(f"episode {sequence}: rounds.csv {field}={summary[field]} != replayed {actual}")
        if summary["end_reason"] != maze.terminal_reason():
            _fail(f"episode {sequence}: rounds.csv end_reason={summary['end_reason']} "
                  f"!= replayed {maze.terminal_reason()}")

    episodes = len(rows_by_episode)
    reached = sum(1 for s in summaries.values() if s["reached"] == "True")
    return f"OK: {episodes} episodes replayed exactly ({reached} reached goal)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        print(audit(args.run_dir))
    except AuditFailure as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
