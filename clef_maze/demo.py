"""Lightweight visual maze viewer and manual/scripted demo. No model inference.

Renders mazes as ASCII art with the recorded path. Scripted-policy runs are
labeled as such; manual play reads WASD keys from stdin. This demonstrates rules
and recorded paths only -- never model experiment results.
"""
import argparse
import json
from pathlib import Path
from .maze import Maze
from .policies import make_policy, POLICIES

BANNER = "CLEF-MAZE DEMO -- scripted/manual demonstration only, not a model result"


def render(layout, positions=(), current=None):
    """ASCII rendering: '#' walls, 'S' start, 'G' goal, '.' path, '@' position."""
    width, height = layout["width"], layout["height"]
    walls = {tuple(w) for w in layout["walls"]}
    rows, cols = 2 * height + 1, 2 * width + 1
    canvas = [[" "] * cols for _ in range(rows)]
    for c in range(cols):
        canvas[0][c] = canvas[rows - 1][c] = "#"
    for r in range(rows):
        canvas[r][0] = canvas[r][cols - 1] = "#"
    for x1, y1, x2, y2 in walls:
        if x1 == x2:
            canvas[y1 + y2 + 1][2 * x1 + 1] = "#"
        else:
            canvas[2 * y1 + 1][x1 + x2 + 1] = "#"
    for jx in range(1, width):
        for jy in range(1, height):
            r, c = 2 * jy, 2 * jx
            if (canvas[r - 1][c] == "#" or canvas[r + 1][c] == "#"
                    or canvas[r][c - 1] == "#" or canvas[r][c + 1] == "#"):
                canvas[r][c] = "#"
    path_cells = {(p[0], p[1]) for p in positions}
    for x, y in path_cells:
        canvas[2 * y + 1][2 * x + 1] = "."
    sx, sy = layout["start"]
    gx, gy = layout["goal"]
    canvas[2 * sy + 1][2 * sx + 1] = "S"
    canvas[2 * gy + 1][2 * gx + 1] = "G"
    if current is not None:
        canvas[2 * current[1] + 1][2 * current[0] + 1] = "@"
    return "\n".join("".join(row) for row in canvas)


def run_policy_demo(layout, attempt_cap, policy_name, policy_seed, show_frames):
    policy = make_policy(policy_name, policy_seed)
    maze = Maze(layout, attempt_cap)
    policy.reset()
    print(BANNER)
    print(f"maze={layout['maze_id']} policy={policy_name} (scripted) cap={attempt_cap}")
    positions = [maze.position[:]]
    if show_frames:
        print(render(layout, positions, maze.position))
    while not maze.done:
        maze.apply(policy(maze))
        positions.append(maze.position[:])
        if show_frames:
            print(f"--- attempt {maze.attempts} ---")
            print(render(layout, positions, maze.position))
    print(render(layout, positions, maze.position))
    print(f"terminal={maze.terminal_reason()} attempts={maze.attempts} "
          f"invalid={maze.invalid_moves} shortest={layout['shortest_path_length']}")


def run_manual(layout, attempt_cap):
    maze = Maze(layout, attempt_cap)
    keys = {"w": "up", "s": "down", "a": "left", "d": "right"}
    print(BANNER)
    print("Manual play: w/a/s/d to move, q to quit.")
    positions = [maze.position[:]]
    while not maze.done:
        print(render(layout, positions, maze.position))
        print(f"attempts {maze.attempts}/{attempt_cap} invalid {maze.invalid_moves}")
        key = input("> ").strip().lower()
        if key == "q":
            break
        if key not in keys:
            print("Use w/a/s/d or q.")
            continue
        maze.apply(keys[key])
        positions.append(maze.position[:])
    print(render(layout, positions, maze.position))
    print(f"terminal={maze.terminal_reason()} attempts={maze.attempts} invalid={maze.invalid_moves}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--maze-id", required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="greedy")
    parser.add_argument("--policy-seed", default="default")
    parser.add_argument("--frames", action="store_true", help="Render every step")
    parser.add_argument("--manual", action="store_true", help="Play manually with WASD")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    layouts = {m["maze_id"]: m for m in manifest["mazes"]}
    if args.maze_id not in layouts:
        parser.error(f"Unknown maze_id: {args.maze_id}")
    layout = layouts[args.maze_id]
    if args.manual:
        run_manual(layout, manifest["attempt_cap"])
    else:
        run_policy_demo(layout, manifest["attempt_cap"], args.policy, args.policy_seed, args.frames)


if __name__ == "__main__":
    main()
