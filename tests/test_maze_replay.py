import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from clef_maze.generate import main as _generate  # noqa: F401 (import exercises module load)
from clef_maze.maze import generate_layout
from clef_maze.replay import audit, AuditFailure
from clef_maze.runner import run_episode
from clef_maze.policies import make_policy


def write_manifest(path):
    mazes = []
    for i, seed in enumerate(("rep-a", "rep-b"), 1):
        layout = generate_layout(seed, 5, 5, 0.1, "far")
        layout["maze_id"] = f"maze-{i:03d}"
        mazes.append(layout)
    manifest = {"generator": "clef-maze maze-v1", "attempt_cap": 60,
                "difficulty": {"width": 5, "height": 5, "loop_fraction": 0.1,
                               "target_distance": "far"},
                "mazes": mazes}
    Path(path).write_text(json.dumps(manifest) + "\n")
    return manifest


def write_run(run_dir, manifest):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "maze-manifest.json").write_text(json.dumps(manifest) + "\n")
    (run_dir / "config.json").write_text(json.dumps({"policies": ["optimal"]}) + "\n")
    rounds = []
    with (run_dir / "decisions.jsonl").open("w") as log:
        for layout in manifest["mazes"]:
            policy = make_policy("optimal")
            maze, rows = run_episode(layout, policy, manifest["attempt_cap"])
            for row in rows:
                log.write(json.dumps({"sequence": 1, **row}) + "\n")
            rounds.append({"sequence": 1, "maze_id": layout["maze_id"], "policy": "optimal",
                           "reached": str(maze.reached), "attempts": maze.attempts,
                           "invalid_moves": maze.invalid_moves,
                           "shortest_path_length": layout["shortest_path_length"],
                           "path_inefficiency": round(maze.attempts / layout["shortest_path_length"], 3),
                           "end_reason": maze.terminal_reason()})
    with (run_dir / "rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rounds[0]))
        writer.writeheader()
        writer.writerows(rounds)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="maze-replay-"))
        self.manifest = write_manifest(self.tmp / "manifest.json")
        self.run_dir = self.tmp / "run"
        write_run(self.run_dir, self.manifest)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_run_audits(self):
        self.assertTrue(audit(self.run_dir).startswith("OK:"))

    def test_altered_action_rejected(self):
        lines = (self.run_dir / "decisions.jsonl").read_text().splitlines()
        row = json.loads(lines[2])
        row["action"] = "up" if row["action"] != "up" else "down"
        lines[2] = json.dumps(row)
        (self.run_dir / "decisions.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_altered_after_state_rejected(self):
        lines = (self.run_dir / "decisions.jsonl").read_text().splitlines()
        row = json.loads(lines[1])
        row["after"]["attempts"] += 5
        lines[1] = json.dumps(row)
        (self.run_dir / "decisions.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_altered_terminal_reason_rejected(self):
        text = (self.run_dir / "rounds.csv").read_text().replace("goal", "capped")
        (self.run_dir / "rounds.csv").write_text(text)
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_wrong_shortest_path_rejected(self):
        manifest = json.loads((self.run_dir / "maze-manifest.json").read_text())
        manifest["mazes"][0]["shortest_path_length"] += 1
        (self.run_dir / "maze-manifest.json").write_text(json.dumps(manifest) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)


if __name__ == "__main__":
    unittest.main()
