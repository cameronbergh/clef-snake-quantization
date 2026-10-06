"""Guarded native Kev campaign backend; importing this module does no inference.

Only an independently approved runner may call verify_inputs/load/evaluate.
The old preflight's tensor hashing/inventory helpers are reused unchanged; its
loader, worker, supervisor, 64-call limit, deadline and spent claim are not used.
No HTTP, download, game loop, quantized checkpoint export, or retry lives here.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

from clef_snake.game import QUESTION
from experiments.kev4b_v1 import preflight as audited
from experiments.kev4b_v1.adapter import TransportResponse, parse_response
from experiments.kev4b_v1.local_resolver import local_snapshot_resolution

BASE_REPO, BASE_REVISION = audited.BASE_REPO, audited.BASE_REVISION
KEV_REPO, KEV_REVISION = audited.KEV_REPO, audited.KEV_REVISION
CODE_REVISION = audited.CODE_REVISION
PINS = {BASE_REPO: BASE_REVISION, KEV_REPO: KEV_REVISION}
VARIANTS = ("bf16", "mlx-affine8-g64", "mlx-affine4-g64")
ACTIONS = ("up", "down", "left", "right")
MAX_DECISIONS, MAX_LOADS, MAX_TOKENS = 45180, 90, 8192
MAX_MEMORY = 48 * 1024**3
PREFIX_CHUNK = 1024
TEMPERATURE = 2.406050072164233
HEAD_FILE_SHA256 = "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c"
HEAD_INVENTORY_SHA256 = "362a712478d849967537009392f58dc463f828388f6403e334cf2160a400f9d8"
FIXTURE_SHA256 = "a2af995c0c1aa1d8f73988862127edc7ade2424c71653ec8580da5ed5c2639fb"
VERIFIED_FILES_SHA256 = "0d2ba4a16303637bda6806e0f492e914b242b64bda6da39c1db68068aef7ce91"
EXPECTED_INVENTORIES = {
    "bf16": "8d8c6cfe11874d898639c86fec0b1bbc1413d25a528bbef76f0afeac4edba641",
    "mlx-affine8-g64": "69d4a1dfba80246035dc1dcbd67cdca98a081de7938b87a1c4762f2254d5d472",
    "mlx-affine4-g64": "bb107d751b200ae58bcd831ac57ec62a6fdba9148e5956cde22c4fc3ab3303dc",
}
VERSIONS = {"mlx": "0.32.2", "mlx-metal": "0.32.2", "mlx-lm": "0.31.3",
            "numpy": "2.5.3", "torch": "2.8.0", "transformers": "5.17.0",
            "peft": "0.21.0", "huggingface-hub": "1.32.0"}
wire, sha, file_hash, safe_path = audited.wire, audited.sha, audited.file_hash, audited.safe_path


class NativeError(RuntimeError):
    """A technical stop; never a substitute action or implicit retry."""


def require(condition, message):
    if not condition:
        raise NativeError(message)


def inventory_hash(inventory):
    return sha(json.dumps(inventory, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode())


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate request JSON key")
        result[key] = value
    return result


def _constant(value):
    raise NativeError("Nonfinite request JSON: " + value)


def parse_request(raw):
    """Check the exact canonical native schema without constructing a game."""
    require(type(raw) is bytes, "Request must be bytes")
    try:
        req = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        require(type(req) is dict and wire(req) == raw, "Request must use canonical ordered JSON")
        require(list(req) == ["model", "state", "questions"] and req["model"] == "kev-latest",
                "Unexpected native request envelope")
        require(type(req["state"]) is dict and list(req["questions"]) == ["move"],
                "Exactly one factual-state move question is required")
        require(req["questions"] == QUESTION
                and list(req["questions"]["move"]["criteria"]) == list(ACTIONS),
                "Frozen four-way question or option order changed")
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
        raise NativeError("Malformed native request") from exc
    return req


def validate_encoding(enc, rows):
    """Strict admission through 8192 complete tokens, including long states."""
    ids = enc.get("ids")
    require(type(ids) is list and 0 < len(ids) <= MAX_TOKENS
            and all(type(x) is int and x >= 0 for x in ids), "Invalid/overlong token IDs")
    require(enc.get("state_truncated") is False and enc.get("option_isolation") is False,
            "Truncation or changed native option isolation forbidden")
    require(all(type(enc.get(k)) is list and len(enc[k]) == len(ids)
                for k in ("seg", "pos", "opt")), "Incomplete native token layout")
    require(all(type(x) is int for k in ("seg", "pos", "opt") for x in enc[k]),
            "Native token layout contains non-integer values")
    state_tokens = enc["seg"].count(0)
    require(type(enc.get("state_tokens")) is int and enc["state_tokens"] == state_tokens > 0,
            "State-token accounting mismatch")
    require(enc["seg"] == [0] * state_tokens + [1] * (len(ids) - state_tokens)
            and enc["pos"] == list(range(len(ids))), "Invalid single-question causal layout")
    require(len(rows) == 1 and len(rows[0]["opts"]) == 4,
            "Exactly one native four-option row required")
    require(enc.get("decide_idx") == [len(ids) - 1]
            and len(enc.get("opt_idx", [])) == 1
            and len(set(enc["opt_idx"][0])) == 4
            and all(type(x) is int and state_tokens <= x < len(ids) - 1
                    for x in enc["opt_idx"][0]), "Invalid option/decision positions")
    require(state_tokens + len(rows[0]["ids"]) == len(ids), "Incomplete native question row")
    passes = math.ceil(state_tokens / PREFIX_CHUNK) + 1
    require(passes <= 9, "Unexpected native pass bound")
    return state_tokens, passes


def validate_quantization(before, after, variant):
    bits = {"mlx-affine8-g64": 8, "mlx-affine4-g64": 4}[variant]
    expected = set(before["linears"])
    require(len(expected) == 248 and set(after["quantized"]) == expected and not after["linears"],
            "Quantized linear scope mismatch")
    require(all(q == {"bits": bits, "group_size": 64, "mode": "affine", "class": "QuantizedLinear"}
                for q in after["quantized"].values()), "Quantization recipe mismatch")
    fixed = {name: spec for name, spec in before["tensors"].items()
             if name not in {n + ".weight" for n in expected}}
    require(len(fixed) == 178 and all(after["tensors"].get(n) == v for n, v in fixed.items()),
            "Retained backbone tensors changed")
    require(after["head"] == before["head"], "Native FP32 head changed")
    added = {name + suffix for name in expected for suffix in (".scales", ".biases")}
    require(set(after["tensors"]) == set(before["tensors"]) | added,
            "Quantized tensor names mismatch")
    require(inventory_hash(after) == EXPECTED_INVENTORIES[variant],
            "Quantized tensor values differ from frozen preflight inventory")


class NativeRuntime:
    """Serial runtime using only audited inventory helpers from the preflight.

    Runner guard: execution_authorized, volume_uuid, check(), check_storage().
    Runner journal: write(event, **fields), sequence, durable flush on each write.
    Caller reserves its independent global attempt before evaluate. Native
    reservation consumes a call even if encoding or evaluation subsequently fails.
    """
    # Explicitly reuse only the source-verified accounting/hash helpers. None
    # of the preflight's execution, loading or retry methods are inherited.
    memory = audited.Runtime.memory
    tensor_digest = audited.Runtime.tensor_digest
    head_inventory = audited.Runtime.head_inventory
    inventory = audited.Runtime.inventory

    def __init__(self, root, repo, journal, guard):
        # Deliberately do not invoke the preflight constructor/deadline logic.
        self.root, self.repo = Path(root).absolute(), Path(repo).resolve()
        self.journal, self.guard = journal, guard
        self.calls = 0
        self.last_response_sequence = None
        self.mx = self.nn = self.np = self.torch = None
        self.model = self.tokenizer = self.api = None
        self.baseline = self.loaded_inventory = None
        self.variant = self.game_id = None
        self.seen_games = set()
        self.failed = False
        self.booted = False
        self.warmups = self.game_attempts = 0
        self.verification_count = 0
        self.warmup_requests = ()

    def check(self):
        require(getattr(self.guard, "execution_authorized", False) is True,
                "Separate campaign execution approval is required")
        kwargs = {} if self.mx is None else {
            "mlx_active_bytes": self.mx.get_active_memory(), "mlx_peak_bytes": self.mx.get_peak_memory()}
        self.guard.check(**kwargs)
        require(self.memory()["rss_peak_bytes"] <= MAX_MEMORY, "RSS peak exceeds 48 GiB")

    def verify_inputs(self):
        """Rehash all pinned assets before each load, without old approval coupling."""
        verification_started = time.monotonic()
        self.check()
        self.verification_count += 1
        self.guard.check_storage()
        require(platform.system() == "Darwin" and platform.machine() == "arm64",
                "Reviewed Apple Silicon runtime required")
        require(platform.python_version() == "3.12.13", "Pinned Python version changed")
        require(Path(sys.prefix).resolve() == self.root / "venv", "Dedicated verified venv required")
        prep = self.repo / "experiments/kev4b_v1"
        artifact = json.loads((prep / "artifact-plan.json").read_text())
        provenance = json.loads((prep / "provenance.json").read_text())
        storage = json.loads((prep / "storage-plan.json").read_text())
        require({x["repo_id"]: x["revision"] for x in artifact["sources"]} == PINS,
                "Pinned artifact plan changed")
        for key, value in audited.offline_environment(self.root, storage).items():
            os.environ[key] = value
        os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        os.environ.pop("PYTHONPATH", None)
        require(os.environ["HF_HOME"] == str(self.root / "hf")
                and os.environ["HF_HUB_CACHE"] == str(self.root / "hf/hub"), "HF cache layout changed")
        acquisition_path = safe_path(self.root / "acquisition.json", self.root)
        acquired = json.loads(acquisition_path.read_text())
        require(acquired["status"] == "verified" and acquired["code_revision"] == CODE_REVISION,
                "Verified acquisition required")
        require({x["repo_id"]: x["revision"] for x in acquired["sources"]} == PINS,
                "Acquisition pin mismatch")
        planned = {x["repo_id"]: x for x in artifact["sources"]}
        snapshots, identities = {}, []
        payload_started = time.monotonic()
        for source in acquired["sources"]:
            ident = source["repo_id"]
            snapshot = safe_path(source["snapshot_path"], self.root)
            require(snapshot == self.root / "hf/hub" / ("models--" + ident.replace("/", "--"))
                    / "snapshots" / PINS[ident], "Snapshot location mismatch")
            expected = {x["path"]: x for x in planned[ident]["artifacts"]}
            entries = {x["path"]: x for x in source["files"]}
            require(set(entries) == set(expected)
                    == {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()},
                    "Snapshot inventory mismatch")
            for relative, entry in entries.items():
                self.check()
                path = safe_path(snapshot / relative, self.root)
                spec = expected[relative]
                value = file_hash(path)
                require(path.stat().st_size == entry["bytes"] == spec["bytes"]
                        and value == entry["sha256"], "Acquisition payload identity mismatch")
                require(value == spec["sha256"] if "sha256" in spec
                        else file_hash(path, git_blob=True) == spec["git_blob_sha1"], "Pinned payload changed")
                identities.append({"repo_id": ident, "path": relative, "bytes": spec["bytes"], "sha256": value})
            snapshots[ident] = snapshot
        payload_seconds = time.monotonic() - payload_started
        verified_files_sha256 = inventory_hash(sorted(identities, key=lambda r: (r["repo_id"], r["path"])))
        require(len(identities) == 29 and verified_files_sha256 == VERIFIED_FILES_SHA256,
                "Complete verified file inventory differs from accepted preflight")
        source_started = time.monotonic()
        source = safe_path(self.root / "code/kev", self.root)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                           text=True, timeout=10).strip()
        status = subprocess.check_output(["git", "--no-optional-locks", "-C", str(source),
                                          "status", "--porcelain", "--untracked-files=all"],
                                         text=True, timeout=10).strip()
        require(revision == CODE_REVISION and not status
                and provenance["upstream_code"]["revision"] == CODE_REVISION, "Pinned source must be clean")
        hashes = {}
        for entry in provenance["upstream_code"]["files"]:
            path = safe_path(source / entry["path"], self.root)
            require(path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"],
                    "Pinned upstream source file changed")
            hashes[entry["path"]] = entry["sha256"]
        source_seconds = time.monotonic() - source_started
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        require(versions == VERSIONS, "Pinned runtime versions changed")
        fixture_path = prep / "preflight-cases.json"
        require(file_hash(fixture_path) == FIXTURE_SHA256, "Saved warmup fixture file changed")
        cases = json.loads(fixture_path.read_text())["cases"][:2]
        require([c["id"] for c in cases] == ["bf16-round-01-first", "bf16-round-01-middle"],
                "Saved warmup order changed")
        warmups = []
        for case in cases:
            request = deepcopy(case["request"])
            require(sha(wire(request)) == case["request_sha256"], "Warmup fixture identity mismatch")
            request["model"] = "kev-latest"
            warmups.append(wire(request))
        self.source, self.verified_snapshots = source, snapshots
        self.warmup_requests = tuple(warmups)
        evidence = {"pins": PINS, "code_revision": CODE_REVISION, "whole_source_git_clean": True,
                    "source_hashes": hashes, "verified_files": identities, "versions": versions,
                    "fixture_sha256": FIXTURE_SHA256, "acquisition_sha256": file_hash(acquisition_path),
                    "verified_files_sha256": verified_files_sha256,
                    "verification_invocation": self.verification_count,
                    "payload_bytes": sum(x["bytes"] for x in identities),
                    "payload_verification_seconds": payload_seconds,
                    "source_verification_seconds": source_seconds,
                    "verification_wall_seconds": time.monotonic() - verification_started}
        self.journal.write("inputs_verified", **evidence)
        return evidence

    def _boot(self):
        if self.booted:
            return
        self.check()
        sys.dont_write_bytecode = True
        sys.addaudithook(audited.no_network)
        sys.path.insert(0, str(self.source))
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import torch
        import kev.checkpoint as checkpoint
        from kev.api import SystemOneRequest, to_record, to_answers, output_tokens
        from kev.model import rows_of
        from kev.mlx_model import PREFILL_CHUNK
        require(Path(checkpoint.__file__).resolve() == self.source / "kev/checkpoint.py",
                "Imported Kev source differs from verified checkout")
        require(PREFILL_CHUNK == PREFIX_CHUNK and mx.metal.is_available(),
                "Native prefix chunk or Metal backend unavailable")
        self.mx, self.nn, self.np, self.torch = mx, nn, np, torch
        self.api = SimpleNamespace(checkpoint=checkpoint, SystemOneRequest=SystemOneRequest,
                                   to_record=to_record, to_answers=to_answers,
                                   output_tokens=output_tokens, rows_of=rows_of)
        mx.set_default_device(mx.gpu)
        self.booted = True
        self.journal.write("runtime", versions=VERSIONS, python=sys.version,
                           platform=platform.platform(), memory=self.memory(), network_allowed=False)

    def load(self, variant, *, game_id):
        require(not self.failed and self.model is None, "Runtime failed or previous model still loaded")
        require(variant in VARIANTS and type(game_id) is str and game_id
                and game_id not in self.seen_games and len(self.seen_games) < MAX_LOADS,
                "Invalid, repeated or excessive game load")
        self.variant, self.game_id = variant, game_id
        self.seen_games.add(game_id)
        try:
            self.verify_inputs()
            self._boot()
            self.journal.write("load_start", game_id=game_id, variant=variant, memory=self.memory())
            snapshot = self.verified_snapshots[KEV_REPO]
            require(file_hash(snapshot / "head.pt") == HEAD_FILE_SHA256, "Native head payload changed")
            ckmod = self.api.checkpoint
            with local_snapshot_resolution(ckmod, self.verified_snapshots,
                                           cache_root=self.root / "hf/hub") as resolver:
                ck = ckmod.Checkpoint(str(snapshot))
                require(ck.meta.base == BASE_REPO and ck.meta.base_revision == BASE_REVISION
                        and not ck.full and not ck.meta.option_isolation and not ck.meta.special_embeddings,
                        "Checkpoint architecture/pin mismatch")
                require(type(ck.meta.temperature) in (int, float) and ck.meta.temperature == TEMPERATURE,
                        "Native temperature changed")
                self.journal.write("checkpoint_metadata", game_id=game_id, variant=variant,
                                   exact_native_temperature=ck.meta.temperature, head_file_sha256=HEAD_FILE_SHA256)
                self.journal.write("local_snapshot_resolution", game_id=game_id, variant=variant,
                                   resolver=resolver.to_record())
                load_started = time.monotonic()
                self.tokenizer, self.model = ck.load("mps", ckmod.LoadOptions(
                    backend="mlx", merge=True, lora_scale=1.0, temperature=None))
            self.check()
            require(self.model.backend == self.model.device == "mlx" and self.model.dtype == "bfloat16",
                    "Native backend/dtype mismatch")
            self.mx.eval(self.model.text.parameters())
            load_seconds = time.monotonic() - load_started
            inventory_started = time.monotonic()
            before = self.inventory(self.model)
            inventory_seconds = time.monotonic() - inventory_started
            require(inventory_hash(before) == EXPECTED_INVENTORIES["bf16"],
                    "Fresh BF16 inventory differs from frozen preflight")
            require(len(before["linears"]) == 248 and not before["quantized"]
                    and sum(math.prod(v["shape"]) for v in before["linears"].values()) == 3569090560,
                    "Eligible native linear inventory mismatch")
            require(before["head"]["sha256"] == HEAD_INVENTORY_SHA256
                    and before["head"]["temperature"] == TEMPERATURE, "Native head identity mismatch")
            if self.baseline is None:
                self.baseline = deepcopy(before)
            require(before == self.baseline, "Fresh BF16 load changed across games")
            self.journal.write("native_inventory", game_id=game_id, variant=variant,
                               inventory=before, inventory_sha256=inventory_hash(before), memory=self.memory(),
                               native_load_and_merge_seconds=load_seconds, inventory_seconds=inventory_seconds)
            after = before
            if variant != "bf16":
                bits = {"mlx-affine8-g64": 8, "mlx-affine4-g64": 4}[variant]
                self.check()
                self.journal.write("quantize_start", game_id=game_id, variant=variant, bits=bits,
                                   group_size=64, mode="affine", quantize_input=False)
                quantize_started = time.monotonic()
                self.nn.quantize(self.model.text, group_size=64, bits=bits, mode="affine",
                                 quantize_input=False, class_predicate=lambda _, m: isinstance(m, self.nn.Linear))
                self.mx.eval(self.model.text.parameters())
                self.check()
                quantize_seconds = time.monotonic() - quantize_started
                inventory_started = time.monotonic()
                after = self.inventory(self.model)
                inventory_seconds = time.monotonic() - inventory_started
                validate_quantization(before, after, variant)
                self.journal.write("quantized_inventory", game_id=game_id, variant=variant,
                                   inventory=after, inventory_sha256=inventory_hash(after), memory=self.memory(),
                                   derivation="fresh_native_merge_then_in_memory_quantization",
                                   quantize_seconds=quantize_seconds, inventory_seconds=inventory_seconds,
                                   persisted_variant=False, export_reload_validated=False)
            self.loaded_inventory = deepcopy(after)
            self.warmups = self.game_attempts = 0
            record = {"game_id": game_id, "variant": variant, "inventory_sha256": inventory_hash(after),
                      "head": after["head"], "retained_backbone_tensors": 178,
                      "export_reload_validated": False, "memory": self.memory()}
            self.journal.write("load_complete", **record)
            return record
        except Exception as exc:
            self.failed = True
            self.journal.write("load_error", game_id=game_id, variant=variant,
                               error_type=type(exc).__name__, message=str(exc))
            raise

    def _reserve(self, raw, context):
        require(not self.failed and self.model is not None, "No healthy loaded native model")
        require(type(raw) is bytes and type(context) is dict, "Native call needs bytes and context")
        require(set(context) == {"call_id", "game_id", "phase", "game_attempt", "warmup_index", "fixture_id"},
                "Native call context fields differ")
        number = context["call_id"]
        require(type(number) is int and number == self.calls + 1 <= MAX_DECISIONS,
                "Native call ID is noncontiguous or exceeds ceiling")
        require(context["game_id"] == self.game_id, "Native call belongs to another loaded game")
        if context["phase"] == "warmup":
            require(context["game_attempt"] is None and type(context["warmup_index"]) is int
                    and context["warmup_index"] == self.warmups + 1 <= 2 and self.game_attempts == 0,
                    "Warmup count/order changed")
            require(raw == self.warmup_requests[self.warmups], "Warmup differs from frozen fixture")
            require(context["fixture_id"] == ("bf16-round-01-first", "bf16-round-01-middle")[self.warmups],
                    "Warmup fixture identifier changed")
            self.warmups += 1
        else:
            require(context["phase"] == "game" and context["warmup_index"] is None
                    and context["fixture_id"] is None and self.warmups == 2
                    and type(context["game_attempt"]) is int
                    and context["game_attempt"] == self.game_attempts + 1 <= 500,
                    "Gameplay call order/count changed")
            self.game_attempts += 1
        self.calls = number
        self.last_response_sequence = None
        fields = {**context, "decision_id": number, "variant": self.variant}
        self.journal.write("decision_start", **fields, request_base64=base64.b64encode(raw).decode(),
                           request_sha256=sha(raw))
        return fields

    def evaluate(self, request_bytes, context):
        fields = self._reserve(request_bytes, context)
        response_bytes = None
        try:
            self.check()
            request = parse_request(request_bytes)
            rec, meta = self.api.to_record(self.api.SystemOneRequest(**request))
            enc = self.model.encode(self.tokenizer, rec, max_state=MAX_TOKENS,
                                    max_branch=MAX_TOKENS, strict=True)
            state, _, rows = self.api.rows_of(enc)
            state_tokens, expected_passes = validate_encoding(enc, rows)
            require(len(state) == state_tokens and len(meta) == 1 and meta[0]["keys"] == list(ACTIONS),
                    "Native metadata or prefix mismatch")
            self.journal.write("encoded", **fields, native_record=rec, native_meta=meta,
                               encoding=enc, complete_row_tokens=len(enc["ids"]))
            passes = []
            text_type, text_instance = type(self.model.text), self.model.text
            original = text_type.__call__
            prefix_count = expected_passes - 1

            def counted(instance, *args, **kwargs):
                if instance is not text_instance:
                    return original(instance, *args, **kwargs)
                self.check()
                index = len(passes)
                length = min(PREFIX_CHUNK, state_tokens - index * PREFIX_CHUNK) if index < prefix_count else len(rows[0]["ids"])
                shape = list(args[0].shape) if args else None
                require(index < expected_passes and shape == [1, length] and kwargs.get("cache") is not None,
                        "Unexpected native internal pass shape/cache")
                item = {"pass_number": index + 1, "input_shape": shape, "cache_supplied": True,
                        "native_stage": "state_prefix" if index < prefix_count else "question_branch"}
                passes.append(item)
                self.journal.write("backbone_pass_start", **fields, **item)
                try:
                    result = original(instance, *args, **kwargs)
                except Exception as exc:
                    self.journal.write("backbone_pass_error", **fields, **item,
                                       error_type=type(exc).__name__, message=str(exc))
                    raise
                # MLX may return a lazy array; model.forward's own evaluations
                # provide completion. This event records return, not GPU timing.
                self.journal.write("backbone_pass_result", **fields, **item, outcome="returned")
                return result

            self.check()
            self.journal.write("native_forward_start", **fields, expected_backbone_passes=expected_passes)
            started = time.monotonic()
            text_type.__call__ = counted
            try:
                logits = self.model.forward(enc)
                require(len(logits) == 1 and tuple(logits[0].shape) == (4,), "Native logits shape changed")
                z = logits[0].detach().cpu().tolist()
                p = self.torch.softmax(logits[0], -1).detach().cpu().tolist()
            finally:
                text_type.__call__ = original
            finite = lambda values: [v if type(v) in (int, float) and math.isfinite(v) else repr(v) for v in values]
            valid = len(z) == len(p) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in z + p)
            selected = max(range(4), key=p.__getitem__) if valid else None
            self.journal.write("raw_decision", **fields, logits=[finite(z)], probabilities=[finite(p)],
                               selected_index=selected, choice=ACTIONS[selected] if selected is not None else None,
                               internal_backbone_passes=passes, memory=self.memory(),
                               forward_wall_seconds_including_instrumentation=time.monotonic() - started)
            require(valid and all(0 <= v <= 1 for v in p) and abs(math.fsum(p) - 1) <= 1e-6,
                    "Native logits/probabilities are invalid")
            require(len(passes) == expected_passes, "Native backbone pass count changed")
            answers = self.api.to_answers([p], meta)
            response_bytes = wire({"model": "kev-latest", "answers": answers, "truncated": False,
                                   "usage": {"input_tokens": len(enc["ids"]),
                                             "output_tokens": self.api.output_tokens(self.tokenizer, answers),
                                             "state_tokens": enc["state_tokens"], "state_tokens_used": len(state)}})
            choice = parse_response(response_bytes)
            require(choice == ACTIONS[selected], "Native response changed unrounded argmax")
            self.check()
            self.journal.write("native_response", **fields, status=200, choice=choice,
                               response_base64=base64.b64encode(response_bytes).decode(),
                               response_sha256=sha(response_bytes))
            self.last_response_sequence = self.journal.sequence
            return TransportResponse(200, response_bytes)
        except Exception as exc:
            self.failed = True
            self.journal.write("decision_error", **fields, error_type=type(exc).__name__, message=str(exc),
                               response_base64=(base64.b64encode(response_bytes).decode()
                                                if response_bytes is not None else None),
                               response_sha256=sha(response_bytes) if response_bytes is not None else None)
            raise

    def unload(self):
        require(self.model is not None, "No native model to unload")
        try:
            self.check()
            inventory_started = time.monotonic()
            after = self.inventory(self.model)
            require(after == self.loaded_inventory, "Loaded weights/head changed during the game")
            record = {"game_id": self.game_id, "variant": self.variant, "inventory": after,
                      "inventory_sha256": inventory_hash(after), "head_after": after["head"], "unchanged": True,
                      "inventory_seconds": time.monotonic() - inventory_started}
            self.model = self.tokenizer = self.loaded_inventory = None
            gc.collect()
            self.mx.clear_cache()
            self.check()
            record["memory"] = self.memory()
            self.journal.write("unloaded", **record)
            return record
        except Exception as exc:
            self.failed = True
            self.journal.write("unload_error", game_id=self.game_id, variant=self.variant,
                               error_type=type(exc).__name__, message=str(exc))
            raise
