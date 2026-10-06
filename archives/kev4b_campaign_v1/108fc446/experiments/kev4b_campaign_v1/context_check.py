"""Synthetic Snake context checks; default mode never imports a tokenizer.

The explicit --tokenize mode uses pinned local tokenizer assets and the unchanged
Kev serializer/encoder. It does not construct a checkpoint, model, or MLX object.
Synthetic coverage is not an exhaustive maximum over all reachable Snake states.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
CODE_REVISION = "5e42a7a03f28134853dd3ff77461457e921e5ec1"
GAME_SHA256 = "f9dba4b830850f9eba36efcc73f2d2992e484c26e419cfc0921dc43079720f08"
SOURCE_HASHES = {
    "kev/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "kev/api.py": "a6f53af5354f8ea17915abc78360fdf291a5108ce39a53a2d9135d48c603ae58",
    "kev/model.py": "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c",
}
TOKENIZER_HASHES = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "KEV_PREFIX_CACHE": "0", "KEV_DATE_FACTS": "0", "KEV_TRUNCATE_STATES": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
}
LENGTHS = (4, 16, 32, 64, 96, 128, 143, 144)
HEAD_OFFSETS = (0, 35, 72, 107)
ACTIONS = ("up", "down", "left", "right")
MAX_TOKENS = 8192


def wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def hamiltonian_cycle():
    """144 distinct cells, including an adjacent final-to-first edge."""
    cells = [(x, 0) for x in range(12)]
    for y in range(1, 12):
        cells.extend((x, y) for x in (range(11, 0, -1) if y % 2 else range(1, 12)))
    cells.extend((0, y) for y in range(11, 0, -1))
    return cells


def _fixture(case_id, body, food, *, legacy_duplicate=False):
    from clef_snake.game import DIRECTIONS, Game
    # No initial food generation, main seeds, transition loop, or model policy.
    game = Game.__new__(Game)
    game.body = [list(cell) for cell in body]
    game.food = list(food)
    delta = tuple(a - b for a, b in zip(game.body[0], game.body[1]))
    game.direction = next(name for name, value in DIRECTIONS.items() if value == delta)
    request = deepcopy(game.request())
    request["model"] = "kev-latest"
    return {"case_id": case_id, "body_length": len(body),
            "legacy_tail_food_duplicate": legacy_duplicate,
            "origin": "synthetic_context_stress_not_a_played_game",
            "request": request}


def synthetic_fixtures():
    """113 deterministic cases; includes the unchanged full-board edge case.

    A full 144-cell body can enter its tail at fallback food [0,0]. Since the
    move eats, the old tail is not popped: the following request has 145 entries
    with [0,0] at both ends and no safe move. Do not 'repair' that legacy state.
    """
    cycle = hamiltonian_cycle()
    cases = []
    for length in LENGTHS:
        for head in HEAD_OFFSETS:
            for orientation in (1, -1):
                body = [cycle[(head + orientation * i) % 144] for i in range(length)]
                occupied = set(body)
                remaining = [cell for cell in cycle if cell not in occupied]
                foods = list(dict.fromkeys((remaining[0], remaining[-1]))) if remaining else [(0, 0)]
                for food_index, food in enumerate(foods):
                    name = f"length-{length:03d}-head-{head:03d}-order-{orientation:+d}-food-{food_index}"
                    cases.append(_fixture(name, body, food))
    before = cycle[1:] + cycle[:1]  # head [1,0], neck [2,0], tail [0,0]
    cases.append(_fixture("legacy-length-145-tail-food", [(0, 0)] + before, (0, 0),
                          legacy_duplicate=True))
    return cases


def verify_environment(environ=None, *, bytecode_disabled=None):
    env = os.environ if environ is None else environ
    disabled = sys.dont_write_bytecode if bytecode_disabled is None else bytecode_disabled
    bad = [name for name, value in OFFLINE_ENV.items() if env.get(name) != value]
    if bad or not disabled:
        raise ValueError("Tokenization requires offline flags and disabled bytecode: " + ", ".join(bad))


def _verified_files(directory, expected):
    actual = {}
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing pinned file: {name}")
        if not path.resolve().is_relative_to(Path("/Volumes/Models").resolve()):
            raise ValueError(f"Pinned file resolves outside Models: {name}")
        actual[name] = sha(path.read_bytes())
        if actual[name] != digest:
            raise ValueError(f"Pinned file hash mismatch: {name}")
    return actual


def verify_local_inputs(base_snapshot, kev_source):
    verify_environment()
    base, source = Path(base_snapshot).resolve(strict=True), Path(kev_source).resolve(strict=True)
    external = Path("/Volumes/Models").resolve(strict=True)
    if not external.is_mount():
        raise ValueError("External Models volume must be mounted")
    for path in (base, source):
        if not path.is_relative_to(external):
            raise ValueError("Inputs must remain on the external Models volume")
    if base.name != BASE_REVISION or base.parent.name != "snapshots" or base.parent.parent.name != "models--Qwen--Qwen3.5-4B-Base":
        raise ValueError("Expected the exact pinned base tokenizer snapshot, not the Kev tokenizer copy")
    for name in ("special_tokens_map.json", "added_tokens.json", "tokenizer.model"):
        if (base / name).exists():
            raise ValueError(f"Unexpected tokenizer override file: {name}")
    revision = subprocess.check_output(["git", "--no-optional-locks", "-C", str(source),
                                        "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "--no-optional-locks", "-C", str(source),
                                     "status", "--porcelain", "--untracked-files=all"], text=True)
    if revision != CODE_REVISION or dirty:
        raise ValueError("Kev source must be clean at the pinned revision")
    from clef_snake import game
    game_hash = sha(Path(game.__file__).read_bytes())
    if game_hash != GAME_SHA256:
        raise ValueError("Snake serializer source changed")
    return base, source, {"base_repository": "Qwen/Qwen3.5-4B-Base", "base_revision": BASE_REVISION,
                          "tokenizer_files_sha256": _verified_files(base, TOKENIZER_HASHES),
                          "kev_code_revision": revision, "kev_files_sha256": _verified_files(source, SOURCE_HASHES),
                          "game_source_sha256": game_hash}


@contextmanager
def tokenizer_only_guard():
    """Fail closed on network, model imports/calls, or weight-file opens.

    The audit hook is active only in this context. It is a second guard around
    a code path that never requests weights; it is not an OS sandbox.
    """
    counters = {"network_attempts": 0, "weight_file_open_attempts": 0,
                "model_load_attempts": 0, "forward_call_attempts": 0}
    active = [True]
    forbidden = ("kev.checkpoint", "kev.mlx_model", "mlx")
    if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden):
        raise ValueError("Run tokenizer validation in a fresh process without model/MLX modules")

    def audit(event, args):
        if not active[0]:
            return
        if event.startswith("socket."):
            counters["network_attempts"] += 1
            raise RuntimeError("Network access forbidden during tokenizer validation")
        if event == "import" and any(args[0] == x or args[0].startswith(x + ".") for x in forbidden):
            counters["model_load_attempts"] += 1
            raise RuntimeError("Model/MLX import forbidden during tokenizer validation")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0]).lower()
            if name.endswith((".safetensors", ".pt", ".pth", ".gguf", ".bin")):
                counters["weight_file_open_attempts"] += 1
                raise RuntimeError("Weight-file access forbidden during tokenizer validation")

    def profile(frame, event, arg):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        function = frame.f_code.co_name
        instance = frame.f_locals.get("self")
        cls = type(instance).__name__ if instance is not None else ""
        load = (module == "transformers.modeling_utils" and function in ("from_pretrained", "__init__"))
        load |= module == "kev.model" and cls in ("DecisionModel", "PointerHead") and function == "__init__"
        forward = module.startswith(("kev.", "transformers.")) and function == "forward"
        if load or forward:
            counters["model_load_attempts" if load else "forward_call_attempts"] += 1
            raise RuntimeError("Model construction/forward forbidden during tokenizer validation")

    old_profile = sys.getprofile()
    sys.addaudithook(audit)
    sys.setprofile(profile)
    try:
        yield counters
    finally:
        active[0] = False
        sys.setprofile(old_profile)


def summarize_encoding(case, enc, state_ids, rows, metadata, markers):
    if len(rows) != 1 or metadata[0]["keys"] != list(ACTIONS):
        raise ValueError("Native question/option order changed")
    if enc["state_truncated"] or enc["option_isolation"]:
        raise ValueError("Truncation or option isolation is not allowed")
    if not 0 < len(enc["ids"]) <= MAX_TOKENS or len(enc["ids"]) != len(state_ids) + len(rows[0]["ids"]):
        raise ValueError("Complete native row exceeds the bound or is inconsistent")
    if enc["state_tokens"] != len(state_ids) or len(enc["opt_idx"][0]) != 4:
        raise ValueError("Invalid native state/option accounting")
    if enc["ids"][0] != markers[0] or enc["ids"][len(state_ids)] != markers[1]:
        raise ValueError("Native state/question markers differ")
    if any(enc["ids"][index] != markers[3] for index in enc["opt_idx"][0]):
        raise ValueError("Native option-end markers differ")
    if enc["ids"][enc["decide_idx"][0]] != markers[4]:
        raise ValueError("Native decision marker differs")
    return {"case_id": case["case_id"], "body_length": case["body_length"],
            "legacy_tail_food_duplicate": case["legacy_tail_food_duplicate"],
            "request_sha256": sha(wire(case["request"])), "native_encoding_sha256": sha(wire(enc)),
            "state_tokens": len(state_ids), "branch_tokens": len(rows[0]["ids"]),
            "complete_row_tokens": len(enc["ids"]), "option_positions": enc["opt_idx"][0],
            "decision_position": enc["decide_idx"][0], "state_truncated": enc["state_truncated"],
            "would_require_internal_passes": (len(state_ids) + 1023) // 1024 + 1}


def tokenize_fixtures(base_snapshot, kev_source):
    started = time.monotonic()
    base, source, provenance = verify_local_inputs(base_snapshot, kev_source)
    cases = synthetic_fixtures()
    if any(name == "kev" or name.startswith("kev.") for name in sys.modules):
        raise ValueError("Run in a fresh process without previously imported Kev modules")
    sys.path.insert(0, str(source))
    try:
        with tokenizer_only_guard() as counters:
            from transformers import AutoTokenizer
            from kev import api, model
            for module, name in ((api, "kev/api.py"), (model, "kev/model.py")):
                if Path(module.__file__).resolve() != source / name:
                    raise ValueError("Imported source differs from verified pinned source")
            # This is the tokenizer call used by load_tokenizer, made explicitly
            # local-only here. No checkpoint loader or model is constructed.
            tok = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
            marker_ids = [tok.convert_tokens_to_ids(token) for token in model.SPECIAL]
            added = json.loads((base / "tokenizer.json").read_text())["added_tokens"]
            expected = {entry["content"]: entry["id"] for entry in added}
            if marker_ids != [expected[token] for token in model.SPECIAL] or len(set(marker_ids)) != 5:
                raise ValueError("Native marker IDs do not match the verified tokenizer")
            records = []
            for case in cases:
                req = api.SystemOneRequest.model_validate(deepcopy(case["request"]))
                rec, metadata = api.to_record(req)
                enc = model.encode(tok, rec, max_state=MAX_TOKENS, max_branch=MAX_TOKENS,
                                   strict=True, option_isolation=False)
                state_ids, _, rows = model.rows_of(enc)
                entry = summarize_encoding(case, enc, state_ids, rows, metadata, marker_ids)
                entry["rendered_state_utf8_bytes"] = len(rec["state"].encode("utf-8"))
                entry["native_record_sha256"] = sha(wire(rec))
                records.append(entry)
            tokenizer_class = type(tok).__name__
    finally:
        sys.path.remove(str(source))
    if any(counters.values()):
        raise ValueError("Forbidden operation was attempted")
    return {"schema_version": 1, "mode": "native_tokenizer_only", "status": "passed",
            "completed_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
            "provenance": provenance, "utility_sha256": sha(Path(__file__).read_bytes()),
            "versions": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "torch", "pydantic")},
            "tokenizer_class": tokenizer_class, "native_markers": dict(zip(model.SPECIAL, marker_ids)),
            "fixtures_sha256": sha(wire(cases)), "case_count": len(records), "cases": records,
            "maximum_complete_row_tokens": max(x["complete_row_tokens"] for x in records),
            "maximum_state_tokens": max(x["state_tokens"] for x in records),
            "maximum_projected_internal_passes": max(x["would_require_internal_passes"] for x in records),
            "truncated_requests": 0, "model_loads": 0, "forward_calls": 0, "games": 0,
            "guard_counters": counters,
            "limits": {"complete_row_tokens": MAX_TOKENS},
            "limitations": ["Synthetic stress coverage, not an exhaustive maximum over all game states.",
                            "No model load, inference, long-state latency, or long-state memory validation.",
                            "Legacy full-board [0,0] food fallback is preserved, including the 145-entry edge state."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenize", action="store_true", help="Explicit CPU tokenizer-only validation; never inference")
    parser.add_argument("--base-snapshot")
    parser.add_argument("--kev-source")
    parser.add_argument("--report", type=Path, help="Create a new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.tokenize:
        if not args.base_snapshot or not args.kev_source:
            parser.error("--tokenize requires --base-snapshot and --kev-source")
        result = tokenize_fixtures(args.base_snapshot, args.kev_source)
    else:
        if args.base_snapshot or args.kev_source:
            parser.error("Source paths require explicit --tokenize")
        cases = synthetic_fixtures()
        result = {"schema_version": 1, "mode": "fixtures_only", "case_count": len(cases),
                  "fixtures_sha256": sha(wire(cases)), "fixtures": cases,
                  "tokenizer_loaded": False, "model_loads": 0, "forward_calls": 0, "games": 0}
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as output:
            output.write(encoded)
        print(json.dumps({"mode": result["mode"], "case_count": result["case_count"], "report_created": True}))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
