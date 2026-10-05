"""Validate offline plans and saved discovery fixtures; never load a model."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "experiments/kev4b_v1"
MAX_ROW_TOKENS = 8192


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_token_count(count):
    """Admission contract only; callers must supply an actual complete-row count."""
    require(type(count) is int and 0 < count <= MAX_ROW_TOKENS,
            "Complete native row must contain 1..8192 tokens without truncation")
    return count


def validate_plans(protocol, storage, artifacts):
    for label, plan in (("protocol", protocol), ("storage", storage), ("artifacts", artifacts)):
        authorization = plan["authorization"]
        require(all(authorization[key] is True for key in
                    ("acquisition", "installation", "model_loading", "inference")),
                label + " must reflect the recorded bounded authorization")
        require(authorization["resource_approval"] == "approved_bounded_preflight" and
                authorization["does_not_authorize_main_campaign"] is True,
                label + " approval scope drift")
        require(authorization["scope"] == {"max_decision_evaluations": 64,
                "max_preflight_wall_seconds": 900, "scored_games": 0,
                "storage_budget_bytes": 56*1024**3,
                "destination": "/Volumes/Models/clef-snake-experiments/kev4b-v1"},
                label + " authorization expanded")
    require(not any(protocol["validation"][key] for key in
                    ("real_tokenizer_checked", "runtime_parity_verified", "quantization_verified")),
            "Offline preparation cannot establish model validation")
    campaign = protocol["main_campaign"]
    require(campaign == {"status": "not_designed_not_frozen", "seeds": None,
            "schedule": None, "games": None, "execution_authorized": False,
            "next_gate": "validate preflight design and results before selecting and freezing a main campaign"},
            "Main campaign must remain undesigned and unfrozen")
    preflight = protocol["preflight"]
    require(preflight["fixed_states"] == 12 and preflight["scored_games"] == 0,
            "Preflight is fixed-state compatibility, not gameplay")
    require(preflight["max_forward_calls"] == 64 and preflight["max_wall_seconds"] == 900,
            "Preflight exceeds proposed bounds")
    require(sum(phase["calls"] for phase in preflight["phases"]) ==
            preflight["planned_forward_calls"] == 60 <= preflight["max_forward_calls"],
            "All warmups, repeats and variants must fit the call budget")
    require(preflight["complete_row_token_limit"] == MAX_ROW_TOKENS and preflight["serial"] is True,
            "Full-row admission or serial inference changed")
    require(protocol["policy"] == {"safety_mask": False, "fallback_action": False,
            "solver": False, "retries": 0, "sampling": False,
            "persist_full_attempt_before_apply": True,
            "errors": "stop technical attempt without moving; never label as a collision or discard"},
            "Native action/failure policy changed")
    environment = protocol["runtime_proposal"]["trusted_launch_environment"]
    require(all(environment[key] == "0" for key in
                ("KEV_PREFIX_CACHE", "KEV_DATE_FACTS", "KEV_TRUNCATE_STATES")),
            "Caching/retry, preprocessing or truncation is enabled")
    require(environment["KEV_BACKEND"] == "mlx" and
            environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1",
            "Backend/offline requirements changed")
    require(protocol["runtime_proposal"]["decision_head_precision"] == "FP32" and
            protocol["runtime_proposal"]["temperature"] == 2.41, "Native head/calibration changed")
    source_bytes = sum(item["bytes"] for source in artifacts["sources"] for item in source["artifacts"])
    require(source_bytes == artifacts["source_snapshot_bytes_exact"] == 9502566604,
            "Pinned source inventory total changed")
    for source in artifacts["sources"]:
        require(sum(item["bytes"] for item in source["artifacts"]) == source["snapshot_bytes_exact"],
                "Per-source byte total differs")
    require(storage["allocations_bytes"]["source_snapshot_exact"] == source_bytes,
            "Storage plan must include all pinned source files")
    initial = sum(storage["allocations_bytes"].values())
    persistent = initial + sum(storage["optional_variant_allocations_bytes"].values())
    require(initial == storage["initial_phase_required_bytes"] <= storage["initial_gate_free_bytes"] == 32 * 1024**3,
            "Initial storage budget drift")
    require(persistent == storage["persisted_phase_required_bytes"] <= storage["persisted_gate_free_bytes"] == 56 * 1024**3,
            "Persisted storage budget drift")
    require(storage["generated_sizes_measured"] is False and storage["reuse_bytes_credited"] == 0,
            "Unverified artifacts cannot receive measured-size/reuse credit")
    require(storage["internal"] is False and storage["required_mount_path"] == "/Volumes/Models" and
            storage["volume_name"] == "Models", "External Models volume is required")
    root = PurePosixPath(storage["planned_root"])
    require(root == PurePosixPath("/Volumes/Models/clef-snake-experiments/kev4b-v1"),
            "Unexpected experiment destination")
    for name, value in storage["environment"].items():
        path = PurePosixPath(value)
        require(path.is_absolute() and ".." not in path.parts and root in path.parents,
                "Cache or temporary path escapes proposed external root: " + name)


def verify_fixtures(root, fixture):
    source = PurePosixPath(fixture["source_path"])
    require(not source.is_absolute() and ".." not in source.parts, "Unsafe fixture source path")
    path = root / source
    require(hashlib.sha256(path.read_bytes()).hexdigest() == fixture["source_sha256"],
            "Published fixture source changed")
    cases = fixture["cases"]
    require(len(cases) == len({case["id"] for case in cases}) == 12, "Expected 12 unique fixture cases")
    wanted = {case["source_line_1_based"]: case for case in cases}
    require(len(wanted) == 12, "Duplicate source request")
    found = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in wanted:
                continue
            row, case = json.loads(line), wanted[line_number]
            wire = json.dumps(case["request"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            source_wire = json.dumps(row["request"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            require(wire == source_wire, "Golden request content/order changed: " + case["id"])
            require(hashlib.sha256(wire.encode()).hexdigest() == case["request_sha256"],
                    "Golden request hash changed")
            require(row["model"] == "bf16" and all(row[key] == case[key] for key in
                    ("seed", "seed_round", "attempt")), "Golden request provenance differs")
            found.add(line_number)
    require(found == set(wanted), "Missing fixture source records")


def verify(root=ROOT):
    root = Path(root)
    folder = root / PREFIX
    read = lambda name: json.loads((folder / name).read_text())
    validate_plans(read("protocol.json"), read("storage-plan.json"), read("artifact-plan.json"))
    verify_fixtures(root, read("preflight-cases.json"))
    require(not any((folder / name).exists() for name in
                    ("freeze.json", "seed-manifest.json", "schedule.json")),
            "No main campaign freeze, seed manifest or schedule is authorized")
    return {"status": "verified_offline_preparation_only", "fixed_discovery_states": 12,
            "model_calls_made_by_verifier": 0, "runtime_parity_verified": False,
            "main_campaign_frozen": False, "bounded_preflight_authorized": True,
            "main_campaign_authorized": False, "execution_performed_by_verifier": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
