"""Generate a saved maze manifest: deterministic layouts for paired comparison.

Manifests store exact layouts so comparisons never rely on regenerating seeds.
Refuses to overwrite an existing file: saved manifests are immutable.
"""
import argparse
import json
from pathlib import Path
from .maze import generate_layout, GENERATOR_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--loop-fraction", type=float, default=0.08)
    parser.add_argument("--target-distance", default="far",
                        help="'far' or an integer shortest-path distance")
    parser.add_argument("--attempt-cap", type=int, default=200)
    parser.add_argument("--id-prefix", default="maze")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite existing output: saved manifests are immutable")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    target = args.target_distance if args.target_distance == "far" else int(args.target_distance)
    mazes = []
    for i, seed in enumerate(args.seeds, 1):
        layout = generate_layout(seed, args.width, args.height, args.loop_fraction, target)
        layout["maze_id"] = f"{args.id_prefix}-{i:03d}"
        mazes.append(layout)
    manifest = {
        "generator": f"clef-maze {GENERATOR_VERSION}",
        "attempt_cap": args.attempt_cap,
        "difficulty": {"width": args.width, "height": args.height,
                       "loop_fraction": args.loop_fraction,
                       "target_distance": args.target_distance},
        "mazes": mazes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(mazes)} mazes to {args.out}")


if __name__ == "__main__":
    main()
