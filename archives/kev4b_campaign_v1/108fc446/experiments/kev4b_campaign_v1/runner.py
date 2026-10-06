"""Serial, approval-gated Kev campaign runner. Import and validation are offline.

Default CLI behavior only validates the frozen preparation. Real execution needs
an external, exact-freeze approval, an unused approval claim, the verified Models
volume and a separate supervised worker. Native model imports occur only inside
that worker after authorization. Tests inject explicitly synthetic backends.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import plistlib
import re
import signal
import stat
import subprocess
import sys
import time

from clef_snake.game import Game
from experiments.kev4b_v1 import adapter

CAMPAIGN_ID = "kev4b_campaign_v1"
CONDITIONS = ("bf16", "mlx-affine8-g64", "mlx-affine4-g64")
ASSET_ROOT = Path("/Volumes/Models/clef-snake-experiments/kev4b-v1")
LIMITS = {
    "campaign_wall_seconds": 43200, "decision_wall_seconds": 120,
    "max_decision_evaluations": 45180, "max_games": 90,
    "successful_move_cap": 500, "max_input_tokens": 8192,
    "max_rss_bytes": 48 * 1024**3, "max_mlx_active_bytes": 48 * 1024**3,
    "max_campaign_evidence_bytes": 16 * 1024**3,
    "max_root_bytes": 56 * 1024**3, "min_free_bytes": 8 * 1024**3,
    "warmups_per_load": 2, "planned_loads": 90,
}
PREFIX = "experiments/kev4b_campaign_v1/"
EXCLUDED_MANIFESTS = (
    "data/2026-10-05/seed-manifest.json",
    "data/2026-10-05-fiveway/seed-manifest.json",
    "experiments/qwen3_snake_v1/seed-manifest.json",
)


class CampaignError(RuntimeError):
    """Technical stop: never an action, retry, collision or game score."""


def require(condition, message):
    if not condition:
        raise CampaignError(message)


def wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, "Duplicate JSON object key")
        out[key] = value
    return out


def read_json(path):
    def reject(value):
        raise CampaignError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=reject)


def canonical_path(path, *, parent=None, exists=True):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, "Canonical absolute paths without symlinks required")
    require(not path.is_symlink(), "Symlink paths forbidden")
    require(parent is None or path.is_relative_to(parent), "Path escapes approved root")
    require(not exists or path.exists(), "Required local path missing: " + str(path))
    return path


@dataclass(frozen=True)
class CampaignPlan:
    protocol: dict
    seeds: tuple
    games: tuple
    warmups: tuple
    freeze_sha256: str
    input_hashes: dict


def validate_bundle(repo):
    """Read and verify the frozen offline preparation, without writing files."""
    repo = Path(repo).resolve()
    folder = repo / PREFIX
    freeze_path = folder / "freeze.json"
    freeze = read_json(freeze_path)
    require(isinstance(freeze.get("files"), dict) and freeze["files"], "Freeze file hashes required")
    critical = {PREFIX + name for name in ("runner.py", "native.py", "audit.py", "verify.py", "context_check.py",
                "analysis-plan.json", "protocol.json", "seed-manifest.json", "schedule.json")}
    critical.update(EXCLUDED_MANIFESTS)
    critical |= {"clef_snake/game.py", "experiments/kev4b_v1/adapter.py",
                 "experiments/kev4b_v1/local_resolver.py", "experiments/kev4b_v1/preflight-cases.json",
                 "experiments/kev4b_v1/preflight.py", "experiments/kev4b_v1/artifact-plan.json",
                 "experiments/kev4b_v1/provenance.json", "experiments/kev4b_v1/storage-plan.json",
                 "clef_snake/__init__.py", "experiments/__init__.py",
                 "experiments/kev4b_v1/__init__.py", PREFIX + "__init__.py"}
    require(critical <= set(freeze["files"]), "Freeze omits execution or input files")
    for relative, expected in freeze["files"].items():
        require(isinstance(relative, str) and not Path(relative).is_absolute(), "Freeze paths must be relative")
        path = canonical_path(repo / relative, parent=repo)
        require(path.is_file() and re.fullmatch(r"[0-9a-f]{64}", expected or ""), "Invalid frozen file entry")
        require(file_digest(path) == expected, "Frozen file changed: " + relative)
    protocol = read_json(folder / "protocol.json")
    require(protocol.get("campaign_id") == CAMPAIGN_ID, "Wrong campaign identity")
    require(protocol.get("inference_authorized_by_preparation") is False, "Preparation must not itself authorize execution")
    require(protocol.get("conditions") == list(CONDITIONS), "Condition family changed")
    require(protocol.get("limits") == LIMITS
            and all(type(v) is int for v in protocol["limits"].values()), "Reviewed limits changed")
    manifest = read_json(folder / "seed-manifest.json")
    seeds = manifest["seeds"]
    require(len(seeds) == 30 and [s["seed_round"] for s in seeds] == list(range(1, 31)), "Thirty ordered paired seeds required")
    require(all(re.fullmatch(r"[0-9a-f]{32}", s["seed"]) for s in seeds)
            and len({s["seed"] for s in seeds}) == 30, "Seeds must be unique saved 128-bit values")
    exclusions = manifest["excluded_manifests"]
    require([entry["path"] for entry in exclusions] == list(EXCLUDED_MANIFESTS), "Seed exclusion sources changed")
    excluded_seeds = set()
    for entry in exclusions:
        path = repo / entry["path"]
        require(file_digest(path) == entry["sha256"] == freeze["files"][entry["path"]],
                "Excluded seed manifest hash mismatch")
        previous = read_json(path)
        values = previous["seeds"] + previous.get("preflight_seeds", [])
        require(len(values) == entry["seed_count"] and all(isinstance(value, str) for value in values),
                "Excluded seed manifest count/schema mismatch")
        excluded_seeds.update(values)
    require(len(excluded_seeds) == manifest["excluded_distinct_seed_count"] == 51,
            "Distinct excluded seed count differs from frozen design")
    require(not ({s["seed"] for s in seeds} & excluded_seeds), "Campaign seeds overlap earlier experiments")
    games = read_json(folder / "schedule.json")["games"]
    require(len(games) == 90 and [g["sequence"] for g in games] == list(range(1, 91)), "Ninety contiguous games required")
    require(len({g["game_id"] for g in games}) == 90, "Game IDs must be unique")
    orders = []
    for index, seed in enumerate(seeds):
        group = games[3 * index:3 * index + 3]
        require(all(g["seed_round"] == seed["seed_round"] and g["seed"] == seed["seed"] for g in group),
                "Games must retain contiguous seed pairing")
        order = tuple(g["variant"] for g in group)
        require(set(order) == set(CONDITIONS) and len(order) == 3, "Each seed requires all three conditions")
        orders.append(order)
    require(Counter(orders) == Counter({p: 5 for p in itertools.permutations(CONDITIONS)}),
            "All six condition orders must occur exactly five times")
    require(orders == list(itertools.permutations(CONDITIONS)) * 5,
            "Prescribed permutation order changed")
    warmup_spec = protocol["warmups"]
    require(warmup_spec["fixture_path"] == "experiments/kev4b_v1/preflight-cases.json"
            and warmup_spec["case_ids"] == ["bf16-round-01-first", "bf16-round-01-middle"],
            "Fixed warmup fixtures changed")
    fixture_cases = read_json(repo / warmup_spec["fixture_path"])["cases"]
    by_id = {case["id"]: case for case in fixture_cases}
    warmups = tuple(by_id[key] for key in warmup_spec["case_ids"])
    for case in warmups:
        require(digest(wire(case["request"])) == case["request_sha256"], "Warmup request hash mismatch")
    return CampaignPlan(protocol, tuple(seeds), tuple(games), warmups,
                        file_digest(freeze_path), dict(freeze["files"]))


def validate_context_report(repo, plan, asset_root):
    """Require the auditor's complete admission check before spending a claim."""
    spec = plan.protocol["context_validation"]
    require(spec.get("required_before_execution") is True
            and spec.get("report_bound_in_external_approval") is True
            and spec.get("report_relative_path") == "evidence/kev-campaign-context-2026-10-06.json"
            and spec.get("utility_path") == PREFIX + "context_check.py"
            and spec.get("mode") == "native_tokenizer_only" and spec.get("synthetic_cases") == 113,
            "Frozen context-validation gate changed")
    report_path = canonical_path(asset_root / spec["report_relative_path"], parent=asset_root)
    # The independent auditor imports only the standard library, and never
    # imports this runner, a tokenizer, the Game implementation, or a model.
    from . import audit
    repo = Path(repo).resolve()
    require(Path(audit.__file__).resolve() == repo / PREFIX / "audit.py",
            "Context auditor import differs from frozen checkout")
    try:
        frozen = audit.read_frozen(repo)
        require(frozen["freeze_sha256"] == plan.freeze_sha256
                and frozen["freeze"]["files"] == plan.input_hashes
                and frozen["protocol"] == plan.protocol,
                "Context auditor and runner freeze identities differ")
        raw = report_path.read_bytes()
        audit.verify_context_report(audit.strict_json(raw), frozen)
    except (audit.EvidenceError, KeyError, TypeError, ValueError, OSError) as exc:
        raise CampaignError("Independent context report validation failed: " + str(exc)) from exc
    return {"sha256": digest(raw), "path": report_path}


def validate_approval(path, plan, asset_root, *, context_report_sha256=None):
    path = canonical_path(path, parent=asset_root)
    approval = read_json(path)
    require(approval.get("schema_version") == 1 and approval.get("campaign_id") == CAMPAIGN_ID,
            "Separate campaign approval schema required")
    require(approval.get("execution_authorized") is True
            and approval.get("freeze_sha256") == plan.freeze_sha256,
            "Approval must explicitly authorize this exact freeze")
    require(approval.get("limits") == plan.protocol["limits"], "Approval budgets differ from freeze")
    require(isinstance(context_report_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", context_report_sha256)
            and approval.get("context_report_sha256") == context_report_sha256,
            "Approval must bind the verified native-tokenizer context report")
    require(isinstance(approval.get("one_use_id"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]{8,100}", approval["one_use_id"]), "Approval needs a safe one-use ID")
    return {"approval_sha256": file_digest(path), "one_use_id": approval["one_use_id"],
            "freeze_sha256": plan.freeze_sha256, "limits": approval["limits"],
            "context_report_sha256": context_report_sha256}


def require_module_origins(repo, *, native_module=None):
    """Real execution must use the same source files whose freeze was checked."""
    repo = Path(repo).resolve()
    expected = {
        "clef_snake": "clef_snake/__init__.py",
        "clef_snake.game": "clef_snake/game.py",
        "experiments": "experiments/__init__.py",
        "experiments.kev4b_v1": "experiments/kev4b_v1/__init__.py",
        "experiments.kev4b_v1.adapter": "experiments/kev4b_v1/adapter.py",
        "experiments.kev4b_campaign_v1": PREFIX + "__init__.py",
    }
    require(Path(__file__).resolve() == repo / PREFIX / "runner.py", "Executing runner differs from frozen checkout")
    for name, relative in expected.items():
        module = sys.modules.get(name)
        require(module is not None and getattr(module, "__file__", None) is not None
                and Path(module.__file__).resolve() == repo / relative,
                "Imported module differs from frozen checkout: " + name)
    if native_module is not None:
        require(Path(native_module.__file__).resolve() == repo / PREFIX / "native.py",
                "Imported native module differs from frozen checkout")
        for name in ("preflight", "local_resolver"):
            module = sys.modules.get("experiments.kev4b_v1." + name)
            require(module is not None and Path(module.__file__).resolve()
                    == repo / "experiments/kev4b_v1" / (name + ".py"),
                    "Imported native helper differs from frozen checkout: " + name)


class Journal:
    """Exclusive JSONL writer: every returned write has been flushed and fsynced."""
    def __init__(self, path, *, synthetic=False, clock=time.monotonic):
        self.file = Path(path).open("x", encoding="utf-8")
        self.sequence = 0
        self.synthetic = synthetic
        self.clock = clock

    def write(self, event, **fields):
        require(not ({"sequence", "event", "synthetic", "monotonic", "unix_time"} & fields.keys()),
                "Journal metadata cannot be overridden")
        self.sequence += 1
        row = {"sequence": self.sequence, "event": event, "synthetic": self.synthetic,
               "monotonic": self.clock(), "unix_time": time.time(), **fields}
        self.file.write(wire(row).decode() + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        return self.sequence

    def close(self):
        self.file.close()


def directory_bytes(path):
    total = 0
    for candidate in Path(path).rglob("*"):
        try:
            info = candidate.stat()
        except FileNotFoundError:
            # Control/ack files are atomically renamed while resource samples
            # run. Count each observed file once; a vanished pending file is
            # replaced by its durable destination, not a storage failure.
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def record_resource_sample(journal, guard, *, sample_phase, worker_state, current_rss_bytes):
    require(sample_phase in {"initial", "periodic", "final"}, "Invalid resource sample phase")
    if sample_phase == "final":
        require(worker_state == "exited" and current_rss_bytes is None, "Final sample requires exited worker")
    else:
        require(worker_state == "running" and type(current_rss_bytes) is int
                and 0 < current_rss_bytes <= guard.limits["max_rss_bytes"], "Live worker needs measured positive bounded RSS")
    storage = guard.check_storage()
    return journal.write("resource_sample", sample_phase=sample_phase, worker_state=worker_state,
                         current_rss_bytes=current_rss_bytes, storage=storage)


def await_initial_resources(guard, claim):
    """No native imports or loads before the supervisor measures live resources."""
    while True:
        guard.check()
        supervisor_pid = int(os.environ["KEV_CAMPAIGN_SUPERVISOR_PID"])
        require(supervisor_pid == os.getppid(), "Supervising parent changed before startup")
        path = guard.output_dir / "supervisor-ready.json"
        if path.exists():
            ready = read_json(path)
            require(ready.get("worker_nonce") == claim["worker_nonce"]
                    and ready.get("deadline") == guard.deadline
                    and type(ready.get("resource_sample_sequence")) is int
                    and ready["resource_sample_sequence"] > 0, "Initial supervisor resource acknowledgment mismatch")
            return
        time.sleep(0.01)


class CampaignGuard:
    def __init__(self, limits, *, asset_root=None, output_dir=None, volume_uuid=None,
                 execution_authorized=False, clock=time.monotonic, deadline=None):
        self.limits = dict(limits)
        self.asset_root, self.output_dir = asset_root, output_dir
        self.volume_uuid, self.execution_authorized = volume_uuid, execution_authorized
        self.clock = clock
        self.deadline = deadline if deadline is not None else clock() + limits["campaign_wall_seconds"]
        self.calls = 0
        self.active = None
        self.last_memory = {}

    def _control(self):
        if not self.execution_authorized:
            return
        path = self.output_dir / "control.json"
        temporary = self.output_dir / "control.pending"
        payload = {"campaign_deadline": self.deadline, "decisions_reserved": self.calls,
                   "active_decision": self.active, "memory": self.last_memory}
        with temporary.open("w") as target:
            target.write(wire(payload).decode())
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)

    def check(self, *, mlx_active_bytes=None, mlx_peak_bytes=None):
        require(self.clock() < self.deadline, "Whole-campaign deadline reached")
        if self.active is not None:
            require(self.clock() < self.active["deadline"], "Per-decision deadline reached")
        if mlx_active_bytes is not None:
            require(type(mlx_active_bytes) is int and 0 <= mlx_active_bytes <= self.limits["max_mlx_active_bytes"],
                    "MLX active memory exceeds campaign limit")
            self.last_memory = {"mlx_active_bytes": mlx_active_bytes, "mlx_peak_bytes": mlx_peak_bytes}
            self._control()

    def check_storage(self):
        if not self.execution_authorized:
            return {"synthetic": True}
        root = canonical_path(self.asset_root)
        require(root == ASSET_ROOT and os.path.ismount("/Volumes/Models"), "Approved Models mount missing")
        info = plistlib.loads(subprocess.check_output(
            ["/usr/sbin/diskutil", "info", "-plist", "/Volumes/Models"], timeout=10))
        require(info.get("MountPoint") == "/Volumes/Models" and info.get("Internal") is False
                and info.get("VolumeName") == "Models"
                and info.get("VolumeUUID", "").lower() == self.volume_uuid.lower(), "Models identity changed")
        stat = os.statvfs(root)
        measured = {"free_bytes": stat.f_bavail * stat.f_frsize,
                    "root_bytes": directory_bytes(root),
                    "evidence_bytes": directory_bytes(self.output_dir) if self.output_dir.exists() else 0}
        require(measured["free_bytes"] >= self.limits["min_free_bytes"], "Free-space margin exhausted")
        require(measured["root_bytes"] <= self.limits["max_root_bytes"], "Root storage budget exceeded")
        require(measured["evidence_bytes"] <= self.limits["max_campaign_evidence_bytes"], "Evidence budget exceeded")
        self.check()
        return measured

    def reserve(self, context):
        self.check()
        require(self.active is None, "A decision is already active")
        require(self.calls < self.limits["max_decision_evaluations"], "Cumulative decision budget exhausted")
        require(context["call_id"] == self.calls + 1, "Call IDs must be contiguous")
        self.calls += 1
        self.active = {"call_id": self.calls,
                       "deadline": min(self.deadline, self.clock() + self.limits["decision_wall_seconds"])}
        self._control()
        if self.execution_authorized:
            # A native call cannot start until the supervisor confirms that its
            # hard timer is armed for this exact reserved call. Merely polling
            # control.json would leave a gap during a slow storage check.
            self._notify_supervisor()
            while True:
                self.check()
                ack_path = self.output_dir / "watchdog-ack.json"
                if ack_path.exists():
                    ack = read_json(ack_path)
                    if ack == self.active:
                        break
                time.sleep(0.01)

    def _notify_supervisor(self):
        supervisor_pid = int(os.environ["KEV_CAMPAIGN_SUPERVISOR_PID"])
        require(supervisor_pid == os.getppid(), "Supervising parent changed")
        os.kill(supervisor_pid, signal.SIGUSR1)

    def finish_decision(self):
        self.check()
        require(self.active is not None, "No reserved decision to finish")
        self.active = None
        self._control()
        if self.execution_authorized:
            self._notify_supervisor()


def warmup_game(case):
    """Reconstruct a frozen factual request; never spawn food or apply a move."""
    game = Game.__new__(Game)
    state = deepcopy(case["request"]["state"])
    game.body, game.direction, game.food = state["body"], state["direction"], state["food"]
    game.alive = True
    expected = deepcopy(case["request"])
    expected["model"] = adapter.MODEL_ALIAS
    require(adapter.make_request(game) == wire(expected), "Warmup factual reconstruction differs from frozen request")
    return game


class CampaignRunner:
    def __init__(self, plan, backend, journal, guard, *, synthetic=False, approval=None):
        require(synthetic is True or guard.execution_authorized is True, "Real execution requires separate approval")
        require(not synthetic or getattr(backend, "synthetic", False) is True,
                "Offline tests require an explicitly synthetic backend")
        require(journal.synthetic is synthetic, "Journal execution label differs from runner")
        if not synthetic:
            require(approval is not None and approval["freeze_sha256"] == plan.freeze_sha256,
                    "Real runner must retain freeze-bound approval evidence")
        self.plan, self.backend, self.journal, self.guard = plan, backend, journal, guard
        self.synthetic, self.approval = synthetic, approval
        self.completed = 0
        self.loaded = False

    def decide(self, game, row, *, phase, game_attempt=None, warmup_index=None, fixture_id=None):
        context = {"call_id": self.guard.calls + 1, "game_id": row["game_id"],
                   "phase": phase, "game_attempt": game_attempt, "warmup_index": warmup_index,
                   "fixture_id": fixture_id}
        called = False
        expected_request = adapter.make_request(game)
        self.guard.reserve(context)
        self.journal.write("call_reserved", **context,
                           request_base64=base64.b64encode(expected_request).decode(),
                           request_sha256=digest(expected_request))

        def transport(raw):
            nonlocal called
            require(not called, "Transport retry forbidden")
            called = True
            require(raw == expected_request, "Request changed after durable reservation")
            return self.backend.evaluate(raw, context)

        result = adapter.attempt(game, transport)
        record = result.to_record()
        record["response_sha256"] = digest(result.response_body) if result.response_body is not None else None
        native_sequence = getattr(self.backend, "last_response_sequence", None)
        self.journal.write("attempt", **context, attempt=record,
                           native_response_sequence=native_sequence if result.error is None else None)
        # This write has been fsynced before any Game.apply, including failures.
        require(called, "No transport attempt occurred")
        require(result.error is None, "Technical decision failure: " + str(result.error))
        self.guard.finish_decision()
        require(getattr(self.backend, "calls", self.guard.calls) == self.guard.calls,
                "Independent native and runner call accounting differs")
        return result.choice, context

    def run(self):
        self.journal.write("run_start", campaign_id=CAMPAIGN_ID,
                           freeze_sha256=self.plan.freeze_sha256, input_hashes=self.plan.input_hashes,
                           limits=self.plan.protocol["limits"], approval=self.approval,
                           planned_games=len(self.plan.games), retries=0, resume=False)
        current = None
        try:
            self.guard.check_storage()
            verified = self.backend.verify_inputs()
            self.journal.write("inputs_verified", metadata=verified)
            for row in self.plan.games:
                current = row["game_id"]
                self.guard.check_storage()
                game = Game(row["seed"])
                self.journal.write("game_start", game_sequence=row["sequence"],
                                   **{key: value for key, value in row.items() if key != "sequence"},
                                   initial_snapshot=game.snapshot(),
                                   food_events=deepcopy(game.food_events))
                self.journal.write("load_start", game_id=current, variant=row["variant"])
                self.loaded = True  # Cleanup also covers a partially failed load.
                metadata = self.backend.load(row["variant"], game_id=current)
                self.journal.write("load_complete", game_id=current, variant=row["variant"], metadata=metadata)
                for index, case in enumerate(self.plan.warmups, 1):
                    self.decide(warmup_game(case), row, phase="warmup", warmup_index=index, fixture_id=case["id"])
                attempts = 0
                cap = self.plan.protocol["limits"]["successful_move_cap"]
                while game.alive and game.steps < cap:
                    self.guard.check()
                    attempts += 1
                    require(attempts <= cap, "Per-game decision-attempt bound exceeded")
                    before = game.snapshot()
                    previous_food_count = len(game.food_events)
                    choice, context = self.decide(game, row, phase="game", game_attempt=attempts)
                    game.apply(choice)
                    self.journal.write("transition", **context, move=choice, before=before,
                                       after=game.snapshot(), new_food_events=deepcopy(game.food_events[previous_food_count:]),
                                       food_events=deepcopy(game.food_events))
                self.journal.write("unload_start", game_id=current, variant=row["variant"])
                metadata = self.backend.unload()
                self.loaded = False
                self.journal.write("unload_complete", game_id=current, variant=row["variant"], metadata=metadata)
                self.guard.check_storage()
                self.completed += 1
                self.journal.write("game_complete", game_id=current, seed_round=row["seed_round"],
                                   seed=row["seed"], variant=row["variant"], score=game.score,
                                   steps=game.steps, attempts=attempts, alive=game.alive,
                                   end_reason="cap" if game.alive else "collision", censored=game.alive,
                                   final_snapshot=game.snapshot(), food_events=deepcopy(game.food_events))
            self.journal.write("run_complete", games_completed=self.completed,
                               decisions_reserved=self.guard.calls, warmups=len(self.plan.games) * len(self.plan.warmups))
            return {"games_completed": self.completed, "decisions_reserved": self.guard.calls}
        except BaseException as exc:
            self.journal.write("run_error", game_id=current, error_type=type(exc).__name__,
                               message=str(exc), games_completed=self.completed,
                               decisions_reserved=self.guard.calls, active_decision=self.guard.active)
            if self.loaded:
                try:
                    metadata = self.backend.unload()
                    self.loaded = False
                    self.journal.write("cleanup_complete", game_id=current, metadata=metadata)
                except BaseException as cleanup:
                    self.journal.write("cleanup_error", game_id=current,
                                       error_type=type(cleanup).__name__, message=str(cleanup))
            raise


def _offline_environment(root):
    env = os.environ.copy()
    paths = {"HF_HOME": root / "hf", "HF_HUB_CACHE": root / "hf/hub",
             "XDG_CACHE_HOME": root / "cache", "TMPDIR": root / "tmp",
             "TORCH_HOME": root / "cache/torch", "UV_CACHE_DIR": root / "cache/uv",
             "PIP_CACHE_DIR": root / "cache/pip"}
    for key, path in paths.items():
        canonical_path(path, parent=root)
        env[key] = str(path)
    for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if key in env:
            canonical_path(env[key], parent=root)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "TOKENIZERS_PARALLELISM": "false", "KEV_PREFIX_CACHE": "0", "KEV_DATE_FACTS": "0",
                "KEV_TRUNCATE_STATES": "0", "KEV_BACKEND": "mlx"})
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    env.pop("PYTHONPATH", None)
    return env


def _no_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"}:
        raise CampaignError("Network forbidden during campaign")


def worker(args, plan):
    root, output = Path(args.root), Path(args.output)
    require_module_origins(args.repo)
    context_report = validate_context_report(args.repo, plan, root)
    approval = validate_approval(args.approval, plan, root, context_report_sha256=context_report["sha256"])
    claim = read_json(args.claim)
    require(claim["approval"] == approval and claim["output"] == str(output), "Approval claim does not match worker")
    require(claim["worker_nonce"] == os.environ.get("KEV_CAMPAIGN_WORKER_NONCE"), "Supervisor worker handoff mismatch")
    guard = CampaignGuard(plan.protocol["limits"], asset_root=root, output_dir=output,
                          volume_uuid=args.volume_uuid, execution_authorized=True, deadline=claim["deadline"])
    guard.check_storage()
    guard._control()
    await_initial_resources(guard, claim)
    sys.dont_write_bytecode = True
    sys.addaudithook(_no_network)
    runner_journal = Journal(output / "runner.jsonl")
    native_journal = Journal(output / "native.jsonl")
    try:
        from . import native  # No model import before execution authorization.
        require_module_origins(args.repo, native_module=native)
        backend = native.NativeRuntime(root, Path(args.repo).resolve(), native_journal, guard)
        CampaignRunner(plan, backend, runner_journal, guard, approval={**approval, "claim_sha256": file_digest(args.claim)}).run()
        return 0
    except BaseException as exc:
        # Core runner records its own errors; this also covers import/constructor failures.
        runner_journal.write("worker_error", error_type=type(exc).__name__, message=str(exc),
                             decisions_reserved=guard.calls)
        return 1
    finally:
        runner_journal.close()
        native_journal.close()


def launch(args, plan):
    """Only real execution entry: separate approval, one-use claim, hard watchdog."""
    root = canonical_path(args.root)
    require_module_origins(args.repo)
    require(root == ASSET_ROOT and Path(sys.prefix).resolve() == root / "venv", "Dedicated external campaign environment required")
    output = canonical_path(args.output, parent=root, exists=False)
    campaign_parent = canonical_path(root / plan.protocol["storage"]["campaign_subdirectory"], parent=root, exists=False)
    require(output.parent == campaign_parent and not output.exists(), "New campaign output required; no resume")
    context_report = validate_context_report(args.repo, plan, root)
    approval = validate_approval(args.approval, plan, root, context_report_sha256=context_report["sha256"])
    guard = CampaignGuard(plan.protocol["limits"], asset_root=root, output_dir=output,
                          volume_uuid=args.volume_uuid, execution_authorized=True)
    guard.check_storage()
    env = _offline_environment(root)
    claim_folder = root / "campaign-claims"
    canonical_path(claim_folder, parent=root, exists=False)
    claim_folder.mkdir(exist_ok=True)
    claim_path = claim_folder / (approval["one_use_id"] + ".json")
    # Consume before spawning, including failures. This code never resumes or retries.
    nonce = os.urandom(32).hex()
    claim = {"schema_version": 1, "approval": approval, "output": str(output),
             "deadline": guard.deadline, "created_unix": time.time(), "worker_nonce": nonce}
    with claim_path.open("x") as target:
        target.write(wire(claim).decode())
        target.flush()
        os.fsync(target.fileno())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for source, name in ((Path(args.approval), "approval.json"), (claim_path, "claim.json"),
                         (context_report["path"], "context-report.json")):
        with (output / name).open("xb") as target:
            target.write(source.read_bytes())
            target.flush()
            os.fsync(target.fileno())
    env["KEV_CAMPAIGN_WORKER_NONCE"] = nonce
    env["KEV_CAMPAIGN_SUPERVISOR_PID"] = str(os.getpid())
    journal = Journal(output / "supervisor.jsonl")
    command = [sys.executable, "-B", "-m", "experiments.kev4b_campaign_v1.runner", "--repo", str(Path(args.repo).resolve()),
               "--root", str(root), "--output", str(output), "--approval", str(Path(args.approval)),
               "--volume-uuid", args.volume_uuid, "--_worker", "--claim", str(claim_path)]
    proc = None
    old_handler = signal.getsignal(signal.SIGALRM)
    old_update_handler = signal.getsignal(signal.SIGUSR1)

    def kill():
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def timeout(signum, frame):
        kill()
        raise CampaignError("Supervisor deadline expired; worker killed")

    def arm_control_inner():
        deadline = guard.deadline
        active = None
        control_path = output / "control.json"
        if control_path.exists():
            state = read_json(control_path)
            require(state["campaign_deadline"] == guard.deadline, "Worker altered campaign deadline")
            require(type(state["decisions_reserved"]) is int
                    and 0 <= state["decisions_reserved"] <= LIMITS["max_decision_evaluations"], "Worker count invalid")
            active = state["active_decision"]
            if active is not None:
                require(active["call_id"] == state["decisions_reserved"], "Active call ID differs from reservation")
                require(type(active["deadline"]) in (int, float) and math.isfinite(active["deadline"])
                        and active["deadline"] <= guard.deadline
                        and active["deadline"] - time.monotonic() <= LIMITS["decision_wall_seconds"],
                        "Invalid per-decision deadline")
                deadline = active["deadline"]
        require(time.monotonic() < deadline, "Decision or campaign deadline expired")
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        if active is not None:
            pending = output / "watchdog-ack.pending"
            with pending.open("w") as target:
                target.write(wire(active).decode())
                target.flush()
                os.fsync(target.fileno())
            os.replace(pending, output / "watchdog-ack.json")
        return deadline

    def arm_control(signum=None, frame=None):
        # Prevent a second update signal from re-entering an atomic ack write;
        # SIGALRM remains unblocked throughout to enforce the hard deadline.
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        try:
            return arm_control_inner()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)

    try:
        journal.write("supervisor_start", freeze_sha256=plan.freeze_sha256, approval=approval,
                      claim_sha256=file_digest(claim_path), deadline=guard.deadline, limits=plan.protocol["limits"])
        with (output / "stdout.txt").open("x") as stdout, (output / "stderr.txt").open("x") as stderr:
            signal.signal(signal.SIGALRM, timeout)
            signal.signal(signal.SIGUSR1, arm_control)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, guard.deadline - time.monotonic()))
            proc = subprocess.Popen(command, cwd=Path(args.repo).resolve(), env=env,
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            last_storage_check = 0.0
            initial_sample = False
            while proc.poll() is None:
                deadline = arm_control()
                rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True,
                                     text=True, timeout=min(2, max(0.001, deadline - time.monotonic())))
                current_rss = None
                if rss.returncode == 0 and rss.stdout.strip().isdigit():
                    current_rss = int(rss.stdout.strip()) * 1024
                    require(current_rss <= LIMITS["max_rss_bytes"], "Worker RSS exceeds 48 GiB")
                else:
                    require(proc.poll() is not None, "Cannot measure RSS for live worker")
                ready_for_initial = (output / "control.json").exists()
                if current_rss is not None and proc.poll() is None and ready_for_initial \
                        and (not initial_sample or time.monotonic() - last_storage_check >= 5):
                    sequence = record_resource_sample(journal, guard,
                        sample_phase="periodic" if initial_sample else "initial",
                        worker_state="running", current_rss_bytes=current_rss)
                    if not initial_sample:
                        pending = output / "supervisor-ready.pending"
                        with pending.open("w") as target:
                            target.write(wire({"worker_nonce": nonce, "deadline": guard.deadline,
                                               "resource_sample_sequence": sequence}).decode())
                            target.flush()
                            os.fsync(target.fileno())
                        os.replace(pending, output / "supervisor-ready.json")
                        initial_sample = True
                    last_storage_check = time.monotonic()
                try:
                    proc.wait(timeout=min(0.25, max(0.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            require(proc.returncode != 0 or initial_sample, "Successful worker lacks initial resource measurement")
            record_resource_sample(journal, guard, sample_phase="final", worker_state="exited", current_rss_bytes=None)
            journal.write("worker_exit", returncode=proc.returncode)
            return proc.returncode
    except BaseException as exc:
        kill()
        if proc is not None:
            proc.wait(timeout=10)
        journal.write("supervisor_error", error_type=type(exc).__name__, message=str(exc))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.signal(signal.SIGUSR1, old_update_handler)
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approval")
    parser.add_argument("--root")
    parser.add_argument("--output")
    parser.add_argument("--volume-uuid")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--claim", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = validate_bundle(args.repo)
    if not args.execute and not args._worker:
        print(json.dumps({"validated": True, "execution": False, "freeze_sha256": plan.freeze_sha256,
                          "games_planned": len(plan.games), "limits": plan.protocol["limits"]}, sort_keys=True))
        return 0
    require(all((args.approval, args.root, args.output, args.volume_uuid)), "Explicit approval and storage arguments required")
    if args._worker:
        require(args.claim is not None, "Worker requires one-use approval claim")
        return worker(args, plan)
    return launch(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
