import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
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
    raw = json.dumps(manifest) + "\n"
    Path(path).write_text(raw)
    return manifest, hashlib.sha256(raw.encode()).hexdigest()


def write_run(run_dir, manifest, manifest_sha):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "maze-manifest.json").write_text(json.dumps(manifest) + "\n")
    (run_dir / "config.json").write_text(json.dumps(
        {"policies": ["optimal"], "manifest_sha256": manifest_sha}) + "\n")
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


def resign(run_dir, manifest):
    """Rewrite a (possibly tampered) manifest and update its recorded sha."""
    raw = json.dumps(manifest) + "\n"
    (run_dir / "maze-manifest.json").write_text(raw)
    config = json.loads((run_dir / "config.json").read_text())
    config["manifest_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    (run_dir / "config.json").write_text(json.dumps(config) + "\n")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="maze-replay-"))
        self.manifest, sha = write_manifest(self.tmp / "manifest.json")
        self.run_dir = self.tmp / "run"
        write_run(self.run_dir, self.manifest, sha)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _manifest(self):
        return json.loads((self.run_dir / "maze-manifest.json").read_text())

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

    def test_altered_inefficiency_rejected(self):
        text = (self.run_dir / "rounds.csv").read_text().replace("1.0", "0.5", 1)
        (self.run_dir / "rounds.csv").write_text(text)
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_manifest_tampering_rejected_by_sha(self):
        manifest = self._manifest()
        manifest["mazes"][0]["shortest_path_length"] += 1
        (self.run_dir / "maze-manifest.json").write_text(json.dumps(manifest) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_hand_edited_walls_rejected_by_regeneration(self):
        manifest = self._manifest()
        walls = manifest["mazes"][0]["walls"]
        walls.append(walls[0])  # duplicate a wall segment: still sorted, still canonical-ish
        manifest["mazes"][0]["walls"] = sorted(walls)
        resign(self.run_dir, manifest)
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_swapped_seed_rejected_by_regeneration(self):
        manifest = self._manifest()
        manifest["mazes"][0]["seed"] = "rep-b"
        resign(self.run_dir, manifest)
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_extra_row_after_termination_rejected(self):
        lines = (self.run_dir / "decisions.jsonl").read_text().splitlines()
        row = json.loads(lines[-1])
        row["attempt"] += 1
        lines.append(json.dumps(row))
        (self.run_dir / "decisions.jsonl").write_text("\n".join(lines) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_unconfigured_policy_rejected(self):
        config = json.loads((self.run_dir / "config.json").read_text())
        config["policies"] = ["greedy"]
        (self.run_dir / "config.json").write_text(json.dumps(config) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)

    def test_missing_manifest_sha_rejected(self):
        config = json.loads((self.run_dir / "config.json").read_text())
        del config["manifest_sha256"]
        (self.run_dir / "config.json").write_text(json.dumps(config) + "\n")
        with self.assertRaises(AuditFailure):
            audit(self.run_dir)


if __name__ == "__main__":
    unittest.main()
