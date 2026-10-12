"""Offline replay/audit of a runner-produced run directory.

Rebuilds each episode from the saved manifest, replays the recorded actions, and
rejects the run on the first mismatch. Checks, in order:

1. Manifest integrity: sha256 of maze-manifest.json matches config.json.
2. Manifest fidelity: every layout regenerates exactly from its recorded
   seed/parameters via the versioned generator (catches hand-edited layouts),
   walls are in canonical order, the goal is reachable, and the recorded
   shortest-path length matches recomputation.
3. Episode replay: each recorded request equals the recomputed request for the
   live state, each recorded action applies cleanly, and each after-state
   equals the replayed snapshot. Rejects altered actions/requests/outcomes,
   out-of-order or extra rows, and logs that end before termination.
4. Summary reconciliation: rounds.csv reached/attempts/invalid_moves,
   end_reason, and path_inefficiency all match the replay; episode sets match
   across files; policies match the configured set.

Exit code 0 means the run is fully reproducible from its manifest; non-zero
names the first failure.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from .maze import Maze, bfs_path, generate_layout


class AuditFailure(Exception):
    pass


def _fail(message):
    raise AuditFailure(message)


def _check_manifest(run_dir):
    manifest_bytes = (run_dir / "maze-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config = json.loads((run_dir / "config.json").read_text())
    expected_sha = config.get("manifest_sha256")
    if not expected_sha:
        _fail("config.json is missing manifest_sha256")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha:
        _fail("maze-manifest.json does not match the sha256 recorded in config.json")
    return manifest, config


def _check_layouts(manifest):
    layouts = {}
    for entry in manifest["mazes"]:
        maze_id = entry["maze_id"]
        if maze_id in layouts:
            _fail(f"duplicate maze_id: {maze_id}")
        layouts[maze_id] = entry
        regen = generate_layout(entry["seed"], entry["width"], entry["height"],
                                entry["loop_fraction"], entry["target_distance"])
        for key in ("walls", "start", "goal", "shortest_path_length"):
            if regen[key] != entry[key]:
                _fail(f"{maze_id}: manifest {key} does not match regeneration "
                      f"from seed {entry['seed']!r}")
        if regen["generator"] != entry["generator"]:
            _fail(f"{maze_id}: generator version mismatch")
        walls = {tuple(w) for w in entry["walls"]}
        if [list(w) for w in sorted(walls)] != entry["walls"]:
            _fail(f"{maze_id}: walls are not in canonical sorted order")
        path = bfs_path(entry["start"], entry["goal"], entry["walls"],
                        entry["width"], entry["height"])
        if path is None:
            _fail(f"{maze_id}: goal unreachable from start")
        if len(path) - 1 != entry["shortest_path_length"]:
            _fail(f"{maze_id}: recorded shortest_path_length {entry['shortest_path_length']} "
                  f"!= recomputed {len(path) - 1}")
    return layouts


def _load_episodes(run_dir):
    rows_by_episode = {}
    for line in (run_dir / "decisions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["maze_id"], row["policy"], row["sequence"])
        rows_by_episode.setdefault(key, []).append(row)
    summaries = {}
    for row in csv.DictReader((run_dir / "rounds.csv").read_text().splitlines()):
        key = (row["maze_id"], row["policy"], int(row["sequence"]))
        if key in summaries:
            _fail(f"rounds.csv has a duplicate episode: {key}")
        summaries[key] = row
    if set(rows_by_episode) != set(summaries):
        _fail("decisions.jsonl episodes do not match rounds.csv episodes")
    return rows_by_episode, summaries


def _replay_episode(maze_id, policy, sequence, rows, summary, layout, attempt_cap):
    maze = Maze(layout, attempt_cap)
    for row in rows:
        if row["attempt"] != maze.attempts + 1:
            _fail(f"episode {sequence}: attempt out of order at {row['attempt']} "
                  f"(missing, duplicate, or extra row)")
        if row["request"] != maze.request():
            _fail(f"episode {sequence}: request mismatch at attempt {row['attempt']} "
                  f"(altered state or layout)")
        try:
            maze.apply(row["action"])
        except ValueError as exc:
            _fail(f"episode {sequence}: recorded action {row['action']!r} rejected "
                  f"at attempt {row['attempt']}: {exc}")
        if row["after"] != maze.snapshot():
            _fail(f"episode {sequence}: after-state mismatch at attempt {row['attempt']} "
                  f"(altered action or outcome)")
    if not maze.done:
        _fail(f"episode {sequence}: log ends before termination")
    if (summary["reached"] == "True") != maze.reached:
        _fail(f"episode {sequence}: rounds.csv reached={summary['reached']} != replayed {maze.reached}")
    for field in ("attempts", "invalid_moves"):
        if int(summary[field]) != getattr(maze, field):
            _fail(f"episode {sequence}: rounds.csv {field}={summary[field]} "
                  f"!= replayed {getattr(maze, field)}")
    if summary["end_reason"] != maze.terminal_reason():
        _fail(f"episode {sequence}: rounds.csv end_reason={summary['end_reason']} "
              f"!= replayed {maze.terminal_reason()}")
    shortest = layout["shortest_path_length"]
    expected_ineff = str(round(maze.attempts / shortest, 3)) if maze.reached else ""
    if str(summary["path_inefficiency"]) != expected_ineff:
        _fail(f"episode {sequence}: rounds.csv path_inefficiency={summary['path_inefficiency']} "
              f"!= recomputed {expected_ineff}")
    return maze.reached


def audit(run_dir):
    run_dir = Path(run_dir)
    manifest, config = _check_manifest(run_dir)
    layouts = _check_layouts(manifest)
    attempt_cap = manifest["attempt_cap"]
    rows_by_episode, summaries = _load_episodes(run_dir)
    configured_policies = set(config.get("policies", []))
    seen_policies = {policy for (_, policy, _) in rows_by_episode}
    if seen_policies != configured_policies:
        _fail(f"episodes use policies {sorted(seen_policies)} but config lists "
              f"{sorted(configured_policies)}")
    reached = 0
    for (maze_id, policy, sequence), rows in sorted(rows_by_episode.items()):
        layout = layouts.get(maze_id)
        if layout is None:
            _fail(f"episode {sequence}: unknown maze_id {maze_id}")
        if _replay_episode(maze_id, policy, sequence, rows, summaries[(maze_id, policy, sequence)],
                           layout, attempt_cap):
            reached += 1
    episodes = len(rows_by_episode)
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
