"""Offline paired runner: scripted policies on a saved manifest. No models.

Each (maze, policy) pair gets a fresh Maze instance from the same saved layout,
so paired conditions share exactly the maze/start/goal/attempt cap while their
states evolve independently. Writes decisions.jsonl + rounds.csv mirroring the
Snake benchmark's evidence shape (request/action/after provenance).
"""
import argparse
import csv
import json
from pathlib import Path
from .maze import Maze
from .policies import make_policy, POLICIES


def run_episode(layout, policy, attempt_cap):
    maze = Maze(layout, attempt_cap)
    policy.reset()
    rows = []
    while not maze.done:
        request = maze.request()
        action = policy(maze)
        maze.apply(action)
        rows.append({"maze_id": layout["maze_id"], "policy": policy.name,
                     "attempt": maze.attempts, "action": action,
                     "request": request, "after": maze.snapshot()})
    return maze, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICIES), default=["greedy"])
    parser.add_argument("--policy-seed", default="default")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output must be empty: completed evidence is never overwritten")
    manifest = json.loads(args.manifest.read_text())
    attempt_cap = manifest["attempt_cap"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "maze-manifest.json").write_bytes(args.manifest.read_bytes())
    (args.out / "config.json").write_text(json.dumps(
        {"policies": args.policies, "policy_seed": args.policy_seed,
         "driver": "clef-maze-runner-v1", "note": "scripted policies only; not model results"},
        indent=2) + "\n")
    rounds = []
    sequence = 0
    with (args.out / "decisions.jsonl").open("w") as log:
        for layout in manifest["mazes"]:
            for policy_name in args.policies:
                sequence += 1
                policy = make_policy(policy_name, args.policy_seed)
                maze, rows = run_episode(layout, policy, attempt_cap)
                for row in rows:
                    log.write(json.dumps({"sequence": sequence, **row}) + "\n")
                shortest = layout["shortest_path_length"]
                rounds.append({
                    "sequence": sequence,
                    "maze_id": layout["maze_id"],
                    "policy": policy_name,
                    "reached": maze.reached,
                    "attempts": maze.attempts,
                    "invalid_moves": maze.invalid_moves,
                    "shortest_path_length": shortest,
                    "path_inefficiency": round(maze.attempts / shortest, 3) if maze.reached else "",
                    "end_reason": maze.terminal_reason(),
                })
                log.flush()
    with (args.out / "rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rounds[0]))
        writer.writeheader()
        writer.writerows(rounds)
    done = sum(1 for r in rounds if r["reached"])
    print(f"Completed {len(rounds)} episodes ({done} reached goal) in {args.out}")


if __name__ == "__main__":
    main()
