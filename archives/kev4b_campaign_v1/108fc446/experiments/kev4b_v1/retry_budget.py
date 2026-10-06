"""One-use budget claim for the explicitly authorized offline-loader retry.

Standard library only. Call before creating the new output directory. A claim
consumes the entire remaining allowance even if launch fails; it is never
removed or resumed automatically. Historical journals are read, never changed.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import time

PRIOR_RUNS = ("2026-10-05-v1", "2026-10-05-v1-setup-continuation")
CLAIM_NAME = "retry-budget-claim.json"
MAX_SECONDS = 900
MAX_RETRY_SECONDS = 719
MAX_DECISIONS = 64
PLANNED_DECISIONS = 60
MAX_JOURNAL_BYTES = 2 * 1024**2
EXPECTED_PRIOR_JOURNAL_SHA256 = {
    (PRIOR_RUNS[0], "supervisor.jsonl"): "e78e5ddcf7115a46781f24c15b244e659934765c2c43d73175ba722f5496c9d0",
    (PRIOR_RUNS[0], "events.jsonl"): "755ef30fcf55015d38272c790be2bb20eb55a84fd76694bd5a8363051657805e",
    (PRIOR_RUNS[1], "supervisor.jsonl"): "79b908d9b3b022a3aa401e4edc08b5a3f8d01ad00fcf5423203d54b0b6ef26fb",
    (PRIOR_RUNS[1], "events.jsonl"): "57b536ed601b491901db931c68dde0b2d0700ee39c0f57d94e2eb49566db05ff",
}


class RetryBudgetError(RuntimeError):
    """Incomplete or incompatible evidence, or an already consumed claim."""


def _require(condition, message):
    if not condition:
        raise RetryBudgetError(message)


def _number(value, label):
    _require(type(value) in (int, Decimal), f"{label}: expected finite number")
    value = Decimal(value)
    _require(value.is_finite(), f"{label}: expected finite number")
    return value


def _integer(value, expected, label):
    _require(type(value) is int and value == expected, f"{label}: expected integer {expected}")


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise RetryBudgetError(f"Non-finite JSON number: {value}")


def _finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, Decimal):
        _require(value.is_finite(), "Non-finite journal value")


def _read_journal(root, run, name):
    path = root / run / name
    _require(path.is_file() and path.resolve() == path, f"Missing or symlinked journal: {run}/{name}")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JOURNAL_BYTES + 1)
    _require(0 < len(raw) <= MAX_JOURNAL_BYTES and raw.endswith(b"\n"),
             f"Empty, incomplete or oversized journal: {run}/{name}")
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == EXPECTED_PRIOR_JOURNAL_SHA256[(run, name)],
             f"Historical journal hash mismatch: {run}/{name}")
    rows = []
    try:
        for sequence, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            row = json.loads(line, parse_float=Decimal, parse_constant=_invalid_constant,
                             object_pairs_hook=_object)
            _require(type(row) is dict, "Journal row must be an object")
            _finite_tree(row)
            _integer(row.get("sequence"), sequence, "Journal sequence")
            _require(type(row.get("event")) is str, "Journal event must be a string")
            for clock in ("monotonic", "unix_time"):
                value = _number(row.get(clock), clock)
                _require(value > 0, f"{clock}: nonpositive timestamp")
                if rows:
                    _require(value >= rows[-1][clock], f"{clock}: journal time reversed")
                row[clock] = value
            rows.append(row)
    except (UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise RetryBudgetError(f"Malformed journal: {run}/{name}: {exc}") from exc
    return rows, {"run": run, "file": name, "sha256": digest,
                  "bytes": len(raw), "rows": len(rows)}


def _validate_run(supervisors, events, *, original):
    supervisor_kinds = [row["event"] for row in supervisors]
    _require(supervisor_kinds[0] == "supervisor_start"
             and supervisor_kinds[-1] == "worker_exit"
             and supervisor_kinds.count("supervisor_start") == 1
             and supervisor_kinds.count("worker_exit") == 1
             and set(supervisor_kinds) <= {"supervisor_start", "resource_sample", "worker_exit"},
             "Prior supervisor lacks a unique terminal worker exit")
    _integer(supervisors[-1].get("returncode"), 1, "Prior worker exit")
    _require(_number(supervisors[-1].get("wall_seconds"), "Reported wall seconds") >= 0,
             "Negative reported worker duration")
    event_kinds = [row["event"] for row in events]
    allowed = {"worker_start", "verified_file", "inputs_verified", "fatal"}
    if not original:
        allowed |= {"run_plan", "runtime", "load_start", "checkpoint_metadata"}
    _require(event_kinds[0] == "worker_start" and event_kinds[-1] == "fatal"
             and event_kinds.count("worker_start") == 1 and event_kinds.count("fatal") == 1,
             "Prior worker lacks a unique terminal fatal event")
    _require(set(event_kinds) <= allowed,
             "Prior run contains decision/backbone activity or an unrecognized event")
    _integer(events[-1].get("decisions_reserved"), 0, "Prior fatal reserved decisions")
    _integer(events[-1].get("games"), 0, "Prior fatal games")
    limits = events[0].get("limits")
    _require(type(limits) is dict, "Missing prior worker limits")
    for field, expected in (("seconds", MAX_SECONDS), ("decisions", MAX_DECISIONS),
                            ("planned_decisions", PLANNED_DECISIONS), ("games", 0)):
        _integer(limits.get(field), expected, f"Prior limit {field}")
    for row in events:
        for field in ("decisions_reserved", "games"):
            if field in row:
                _integer(row[field], 0, f"Prior event {field}")
        if row["event"] == "run_plan":
            _integer(row.get("planned_decisions"), PLANNED_DECISIONS, "Prior planned decisions")
            _integer(row.get("retries"), 0, "Prior planned retries")
    for clock in ("monotonic", "unix_time"):
        _require(supervisors[0][clock] <= events[0][clock]
                 <= events[-1][clock] <= supervisors[-1][clock],
                 f"Prior worker timestamps outside supervisor interval: {clock}")
    return supervisors[0], supervisors[-1]


def _inventory(root, *, claimed=False):
    allowed = set(PRIOR_RUNS) | ({CLAIM_NAME} if claimed else set())
    _require({path.name for path in root.iterdir()} == allowed,
             "Unaccounted preflight attempt or entry; retry forbidden")
    for run in PRIOR_RUNS:
        path = root / run
        _require(path.is_dir() and path.resolve() == path,
                 "Missing or symlinked historical run directory")


def claim_retry_budget(preflight_root, output_path):
    """Atomically reserve the sole retry and return its remaining budget.

    The caller must enforce the returned wall/decision limits and create the
    output only after this returns. This does not launch anything. Validation
    errors leave no claim; errors after the exclusive create consume it.
    """
    root, output = Path(preflight_root), Path(output_path)
    _require(root.is_absolute() and root.is_dir() and root.resolve() == root,
             "Preflight root must be an existing absolute, non-symlink directory")
    _require(output.is_absolute() and output.parent == root
             and output.resolve() == output and not os.path.lexists(output)
             and output.name not in {*PRIOR_RUNS, CLAIM_NAME},
             "Retry output must be an absent direct child of preflight root")
    claim_path = root / CLAIM_NAME
    _require(not os.path.lexists(claim_path), "Retry budget already claimed; no automatic retry")
    _inventory(root)
    journals, evidence = {}, []
    for run in PRIOR_RUNS:
        for name in ("supervisor.jsonl", "events.jsonl"):
            rows, identity = _read_journal(root, run, name)
            journals[(run, name)] = rows
            evidence.append(identity)
    first, first_exit = _validate_run(journals[(PRIOR_RUNS[0], "supervisor.jsonl")],
                                      journals[(PRIOR_RUNS[0], "events.jsonl")], original=True)
    second, last_exit = _validate_run(journals[(PRIOR_RUNS[1], "supervisor.jsonl")],
                                     journals[(PRIOR_RUNS[1], "events.jsonl")], original=False)
    deadline = _number(first.get("deadline"), "Original deadline")
    _require(Decimal(0) <= first["monotonic"] + MAX_SECONDS - deadline <= Decimal("0.01"),
             "Original deadline does not preserve the 900-second budget")
    _require(_number(second.get("deadline"), "Continuation deadline") == deadline,
             "Continuation changed the original deadline")
    prior = second.get("prior_setup")
    _require(type(prior) is dict and prior.get("path") == str(root / PRIOR_RUNS[0]),
             "Continuation does not name the original attempt")
    for field, name in (("supervisor_sha256", "supervisor.jsonl"), ("events_sha256", "events.jsonl")):
        _require(prior.get(field) == EXPECTED_PRIOR_JOURNAL_SHA256[(PRIOR_RUNS[0], name)],
                 f"Continuation prior {field} binding mismatch")
    _require(type(first.get("harness_sha256")) is str and len(first["harness_sha256"]) == 64
             and all(c in "0123456789abcdef" for c in first["harness_sha256"])
             and prior.get("prior_harness_sha256") == first["harness_sha256"],
             "Continuation prior harness binding mismatch")
    _require(_number(prior.get("original_deadline"), "Bound original deadline") == deadline,
             "Continuation prior deadline binding mismatch")
    _integer(prior.get("prior_decisions_reserved"), 0, "Continuation prior decisions")
    for clock in ("monotonic", "unix_time"):
        _require(first_exit[clock] <= second[clock], "Historical attempt intervals overlap")
    elapsed = last_exit["monotonic"] - first["monotonic"]
    _require(0 < elapsed < MAX_SECONDS and last_exit["monotonic"] <= deadline,
             "Original aggregate time budget is exhausted or invalid")
    charged = int(elapsed.to_integral_value(rounding=ROUND_CEILING))
    remaining = min(MAX_RETRY_SECONDS, MAX_SECONDS - charged)
    _require(0 < remaining <= MAX_RETRY_SECONDS, "No approved retry time remains")
    claim = {
        "schema_version": 1, "status": "claimed_once", "automatic_retry": False,
        "output_path": str(output), "claim_path": str(claim_path),
        "claimed_unix_time": time.time(), "claimed_monotonic": time.monotonic(),
        "aggregate_seconds_limit": MAX_SECONDS, "remaining_seconds": remaining,
        "prior_elapsed_seconds": float(elapsed), "prior_charged_seconds": charged,
        "prior_decisions": 0, "remaining_decisions": MAX_DECISIONS,
        "planned_decisions": PLANNED_DECISIONS, "games": 0,
        "evidence": {"journals": evidence,
                     "original_start_monotonic": str(first["monotonic"]),
                     "last_exit_monotonic": str(last_exit["monotonic"]),
                     "original_deadline": str(deadline),
                     "accounting": "ceil(last supervisor exit - original supervisor start); includes inter-attempt gap; reported wall_seconds are not summed",
                     "decision_start_events": 0, "raw_decision_events": 0,
                     "backbone_events": 0},
    }
    encoded = (json.dumps(claim, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(claim_path, flags, 0o600)
    except FileExistsError as exc:
        raise RetryBudgetError("Retry budget already claimed; no automatic retry") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    # A concurrent change after validation consumes the claim and stops launch.
    _inventory(root, claimed=True)
    _require(not os.path.lexists(output), "Output appeared during budget claim")
    for identity in evidence:
        _read_journal(root, identity["run"], identity["file"])
    return claim
