"""Verify the preparation freeze offline; never contact a model or create seeds."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import re

from clef_snake.game import Game
from .adapter import build_request

PREFIX = "experiments/qwen3_snake_v1/"
ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_design(protocol, manifest, schedule, published_seeds):
    seeds, preflight = manifest["seeds"], manifest["preflight_seeds"]
    all_seeds = seeds + preflight
    require(len(seeds) == 30 and len(preflight) == 6, "Expected 30 + 6 saved seeds")
    require(len(set(all_seeds)) == 36, "Repeated main/preflight seed")
    require(all(isinstance(s, str) and re.fullmatch(r"[0-9a-f]{32}", s)
                for s in all_seeds), "Invalid seed encoding")
    require(not set(all_seeds).intersection(published_seeds), "Discovery seed reused")
    require(protocol["status"] == "prepared_unexecuted" and
            protocol["execution_authorized"] is False and
            protocol["results_collected"] is False, "Preparation is not execution")
    models = [v["id"] for v in protocol["variants"]]
    require(models == ["bf16", "q4-k-m", "q2-k"] == schedule["models"],
            "Variant family differs from freeze")
    rounds, execution = schedule["rounds"], schedule["execution_order"]
    require(len(rounds) == 30 and len(execution) == 90, "Incomplete schedule")
    orders = Counter(tuple(row["order"]) for row in rounds)
    require(orders == Counter({p: 5 for p in itertools.permutations(models)}),
            "Within-seed orders are not balanced")
    expected = []
    for number, (seed, row) in enumerate(zip(seeds, rounds), 1):
        require(row["round"] == number and row["seed"] == seed, "Seed order changed")
        for model in row["order"]:
            expected.append({"position": len(expected) + 1, "round": number,
                             "seed": seed, "model": model})
    require(execution == expected, "Execution entries differ from saved order")
    expected_preflight = [{"position": i * 3 + j + 1, "seed": seed, "model": model}
                          for i, seed in enumerate(preflight)
                          for j, model in enumerate(models)]
    require(schedule["preflight"]["order"] == expected_preflight,
            "Preflight is not the separate 18-request panel")
    request = build_request(Game(preflight[0]))
    frozen = protocol["request"]
    require(request["messages"][0]["content"] == frozen["system_prompt"],
            "Adapter prompt differs from protocol")
    require(request["response_format"] == frozen["response_format"],
            "Adapter schema differs from protocol")
    require({k: v for k, v in request.items()
             if k not in ("model", "messages", "response_format")} == frozen["decoding"],
            "Adapter decoding differs from protocol")
    require(protocol["game"]["successful_move_cap"] == 500 and
            protocol["design"]["maximum_main_requests"] == 45000,
            "Fixed campaign bounds changed")


def verify(root=ROOT):
    root = Path(root)
    read = lambda name: json.loads((root / PREFIX / name).read_text())
    freeze = read("freeze.json")
    required = {PREFIX + name for name in ["adapter.py", "verify.py", "storage.py",
        "protocol.json", "seed-manifest.json", "schedule.json", "artifact-plan.json",
        "storage-plan.json", "ANALYSIS_PLAN.md", "README.md", "CANDIDATES.md"]}
    required.update(["clef_snake/game.py", "tests/test_qwen_adapter.py",
                     "tests/test_qwen_storage.py", "tests/test_qwen_preparation.py"])
    require(required.issubset(freeze["files"]), "Freeze omits required inputs")
    for name, digest in freeze["files"].items():
        parts = PurePosixPath(name)
        require(not parts.is_absolute() and ".." not in parts.parts, "Unsafe freeze path")
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Missing/redirected frozen file: " + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                "Frozen file changed: " + name)
    published = set()
    for path in (root / "data").glob("*/seed-manifest.json"):
        published.update(json.loads(path.read_text())["seeds"])
    protocol, manifest, schedule = read("protocol.json"), read("seed-manifest.json"), read("schedule.json")
    validate_design(protocol, manifest, schedule, published)
    artifact, storage = read("artifact-plan.json"), read("storage-plan.json")
    require(artifact["selected_source_commit"] == protocol["source"]["revision"],
            "Source pins disagree")
    require(artifact["runtime"]["commit"] == protocol["runtime"]["revision"],
            "Runtime pins disagree")
    source_bytes = sum(row["bytes"] for row in artifact["artifacts"])
    require(source_bytes == artifact["source_snapshot_bytes_exact"] ==
            storage["allocations_bytes"]["source_snapshot_exact"], "Source byte totals disagree")
    require(sum(storage["allocations_bytes"].values()) ==
            storage["persistent_phase_required_free_bytes"] <= storage["launch_gate_free_bytes"],
            "Storage reservations exceed the launch gate")
    require(storage["required_mount_path"] == "/Volumes/Models" and
            storage["internal"] is False and storage["execution_authorized"] is False,
            "External-only preparation policy changed")
    return {"status": "verified_preparation_only", "frozen_files": len(freeze["files"]),
            "main_seeds": 30, "preflight_seeds": 6, "planned_games": 90,
            "model_calls_made_by_verifier": 0, "execution_authorized": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
