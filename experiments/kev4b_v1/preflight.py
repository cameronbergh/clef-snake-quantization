#!/usr/bin/env python3
"""Reviewable, opt-in Kev compatibility preflight; NEVER a gameplay runner.

Run only from the approved dedicated environment, after acquisition verification:
  ROOT/venv/bin/python -B preflight.py --root ROOT --repo PREP_CHECKOUT \
      --output ROOT/preflight/NEW_RUN --volume-uuid VERIFIED_MODELS_UUID

The supervisor permits at most 900 seconds from worker launch (stricter than
900 seconds from first model load), less every accounted prior attempt. It enforces
48 GiB worker RSS and one worker process.
The worker permits 64 decision evaluations, plans exactly 60, and stops at
48 GiB MLX active memory. Warmups count. An explicitly authorized retry must claim the aggregate remaining budget. Two
warmups each: native BF16, wrapped BF16, Q8, Q4. Then 12 native BF16, 12 wrapped
BF16, four wrapped BF16 repeats, 12 Q8, 12 Q4. Native and wrapped observations
are independently evaluated, not two labels for one model call.

Requires acquisition.json produced by the separate approved acquisition step.
Every source snapshot file is rehashed before imports/loading; no downloader is
provided. Quantized variants are derived in memory from separate fresh native
loads. This does NOT validate an export/reload format or freeze a campaign.
All model imports, model loading, and inference live behind the worker entry.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import plistlib
import resource
import signal
import subprocess
import sys
import time
import traceback

CODE_REVISION = "5e42a7a03f28134853dd3ff77461457e921e5ec1"
BASE_REPO = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_REPO = "jaredpalmer/kev-4b"
KEV_REVISION = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
ROOT = Path("/Volumes/Models/clef-snake-experiments/kev4b-v1")
MAX_SECONDS = 900
MAX_DECISIONS = 64
PLANNED_DECISIONS = 60
MAX_MEMORY = 48 * 1024**3
MAX_STORAGE = 56 * 1024**3
MAX_EVIDENCE = 1024**3
MIN_FREE = 8 * 1024**3
MAX_TOKENS = 8192
EXPECTED_LINEARS = 248
EXPECTED_LINEAR_WEIGHTS = 3569090560
PARITY_TOLERANCE = 1e-6
ACTIONS = ["up", "down", "left", "right"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(path, root, *, exists=True):
    path = Path(path).absolute()
    require(path.is_relative_to(root), f"Path outside approved root: {path}")
    require(path.resolve() == path, f"Symlink or unresolved ancestor: {path}")
    require(not exists or path.exists(), f"Required local path missing: {path}")
    return path


def mount_guard(root, expected_uuid):
    require(root == ROOT and root.resolve() == ROOT, "Unexpected Models root")
    require(os.path.ismount("/Volumes/Models"), "Models mount point is not mounted")
    info = plistlib.loads(subprocess.check_output(
        ["/usr/sbin/diskutil", "info", "-plist", "/Volumes/Models"], timeout=10))
    require(info.get("MountPoint") == "/Volumes/Models", "Models is not mounted")
    require(info.get("VolumeName") == "Models" and info.get("Internal") is False,
            "Models identity is not the approved external volume")
    require(info.get("VolumeUUID", "").lower() == expected_uuid.lower(),
            "Models volume UUID mismatch")
    stat = os.statvfs(root)
    require(stat.f_bavail * stat.f_frsize >= MIN_FREE, "Models free margin below 8 GiB")
    occupied = directory_bytes(root)
    require(occupied <= MAX_STORAGE, "Approved root exceeds the 56 GiB storage ceiling")
    return {"volume_uuid": info["VolumeUUID"],
            "free_bytes": stat.f_bavail * stat.f_frsize, "root_bytes": occupied}


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class Journal:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")
        self.sequence = 0

    def write(self, event, **fields):
        self.sequence += 1
        row = {"sequence": self.sequence, "event": event,
               "unix_time": time.time(), "monotonic": time.monotonic(), **fields}
        encoded = wire(row).decode() + "\n"
        # Two journals plus stdout/stderr share a 1 GiB supervisor-monitored
        # evidence ceiling. Stop a single journal earlier, reserving space for
        # a final supervisor/error record.
        require(self.file.tell() + len(encoded.encode()) < MAX_EVIDENCE // 2,
                "Journal reached its 512 MiB evidence sublimit")
        self.file.write(encoded)
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


def offline_environment(root, storage):
    env = os.environ.copy()
    for name, value in storage["environment"].items():
        safe_path(value, root)
        env[name] = value
    # Reject alternate cache overrides rather than allowing an internal fallback.
    for name in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                 "PYTORCH_PRETRAINED_BERT_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if name in env:
            safe_path(env[name], root)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false",
                "KEV_BACKEND": "mlx", "KEV_PREFIX_CACHE": "0",
                "KEV_DATE_FACTS": "0", "KEV_TRUNCATE_STATES": "0",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0"})
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    env.pop("PYTHONPATH", None)
    return env


def verify_inputs(args, journal):
    root, repo = Path(args.root).absolute(), Path(args.repo).resolve()
    mount = mount_guard(root, args.volume_uuid)
    require(platform.system() == "Darwin" and platform.machine() == "arm64",
            "Only the reviewed Apple Silicon runtime is permitted")
    require(Path(sys.prefix).resolve() == root / "venv", "Dedicated venv required")
    prep = repo / "experiments/kev4b_v1"
    plans = {name: json.loads((prep / f"{name}.json").read_text())
             for name in ("artifact-plan", "provenance", "storage-plan", "protocol")}
    auth = plans["artifact-plan"]["authorization"]
    require(auth["resource_approval"] == "approved_bounded_preflight"
            and auth["inference"] is True
            and auth["scope"]["scored_games"] == 0, "Bounded approval missing")
    require(auth["scope"]["max_decision_evaluations"] == MAX_DECISIONS
            and auth["scope"]["max_preflight_wall_seconds"] == MAX_SECONDS,
            "Reviewed budget changed")
    for name, value in offline_environment(root, plans["storage-plan"]).items():
        os.environ[name] = value
    require(os.environ["HF_HOME"] == str(root / "hf")
            and os.environ["HF_HUB_CACHE"] == str(root / "hf/hub"),
            "Hugging Face cache paths differ from reviewed layout")
    acquisition_path = safe_path(root / "acquisition.json", root)
    acquired = json.loads(acquisition_path.read_text())
    require(acquired.get("status") == "verified"
            and acquired.get("code_revision") == CODE_REVISION,
            "Source acquisition is not verified at the pinned revision")
    expected_pins = {BASE_REPO: BASE_REVISION, KEV_REPO: KEV_REVISION}
    require({s["repo_id"]: s["revision"] for s in acquired["sources"]} == expected_pins,
            "Acquired source pin mismatch")
    snapshots = {}
    planned = {s["repo_id"]: s for s in plans["artifact-plan"]["sources"]}
    for source in acquired["sources"]:
        ident, revision = source["repo_id"], source["revision"]
        snapshot = safe_path(source["snapshot_path"], root)
        expected = root / "hf/hub" / ("models--" + ident.replace("/", "--")) / "snapshots" / revision
        require(snapshot == expected, "Snapshot location differs from pinned cache")
        metadata = {f["path"]: f for f in planned[ident]["artifacts"]}
        entries = {f["path"]: f for f in source["files"]}
        require(set(entries) == set(metadata), "Acquisition file inventory mismatch")
        actual_paths = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()}
        require(actual_paths == set(metadata), "Unexpected or missing snapshot files")
        for relative, entry in entries.items():
            path = safe_path(snapshot / relative, root)
            spec = metadata[relative]
            require(path.is_file() and path.stat().st_size == entry["bytes"] == spec["bytes"],
                    f"Artifact size mismatch: {ident}/{relative}")
            digest = file_hash(path)
            require(digest == entry["sha256"], f"Acquisition hash mismatch: {relative}")
            if "sha256" in spec:
                require(digest == spec["sha256"], f"Pinned LFS hash mismatch: {relative}")
            else:
                require(file_hash(path, git_blob=True) == spec["git_blob_sha1"],
                        f"Pinned Git blob mismatch: {relative}")
            journal.write("verified_file", repo_id=ident, path=relative,
                          bytes=entry["bytes"], sha256=digest)
        snapshots[ident] = snapshot
    source = safe_path(root / "code/kev", root)
    git_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    git_status = subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True, timeout=10).strip()
    require(git_revision == CODE_REVISION and not git_status,
            "Entire Kev source checkout must be clean at the pinned commit")
    require(plans["provenance"]["upstream_code"]["revision"] == CODE_REVISION,
            "Source plan changed")
    source_hashes = {}
    for entry in plans["provenance"]["upstream_code"]["files"]:
        path = safe_path(source / entry["path"], root)
        require(path.stat().st_size == entry["bytes"] and file_hash(path) == entry["sha256"],
                f"Pinned source changed: {entry['path']}")
        source_hashes[entry["path"]] = entry["sha256"]
    fixtures_path = prep / "preflight-cases.json"
    fixtures = json.loads(fixtures_path.read_text())["cases"]
    require(len(fixtures) == 12 and len({c["id"] for c in fixtures}) == 12,
            "Exactly 12 unique saved fixtures required")
    for case in fixtures:
        require(sha(wire(case["request"])) == case["request_sha256"],
                "Saved fixture request hash mismatch")
        require(list(case["request"]["questions"]) == ["move"]
                and list(case["request"]["questions"]["move"]["criteria"]) == ACTIONS,
                "Fixture question/options changed")
    journal.write("inputs_verified", mount=mount, source_hashes=source_hashes,
                  acquisition_sha256=file_hash(acquisition_path),
                  fixture_sha256=file_hash(fixtures_path),
                  adapter_sha256=file_hash(prep / "adapter.py"),
                  resolver_sha256=file_hash(prep / "local_resolver.py"),
                  budget_sha256=file_hash(prep / "retry_budget.py"),
                  game_sha256=file_hash(repo / "clef_snake/game.py"),
                  plan_hashes={name: file_hash(prep / f"{name}.json") for name in plans},
                  pins=expected_pins, code_revision=CODE_REVISION,
                  whole_source_git_clean=True)
    return root, repo, source, snapshots, fixtures


def no_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                 "socket.gethostbyname", "socket.sendto"}:
        raise RuntimeError("Network access forbidden during offline preflight")


class Runtime:
    def __init__(self, args, journal):
        self.args, self.journal = args, journal
        self.calls = 0
        self.deadline = float(os.environ["KEV_PREFLIGHT_DEADLINE"])
        self.first_load = None
        self.mx = self.np = self.torch = None

    def memory(self):
        # Darwin ru_maxrss is bytes; this is peak RSS, not current RSS.
        result = {"rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if self.mx is not None:
            result.update({"mlx_active_bytes": self.mx.get_active_memory(),
                           "mlx_peak_bytes": self.mx.get_peak_memory(),
                           "mlx_cache_bytes": self.mx.get_cache_memory()})
        return result

    def check(self):
        require(time.monotonic() < self.deadline, "Preflight deadline reached")
        mem = self.memory()
        require(mem["rss_peak_bytes"] <= MAX_MEMORY, "RSS peak exceeded 48 GiB")
        require(mem.get("mlx_active_bytes", 0) <= MAX_MEMORY, "MLX active memory exceeded 48 GiB")

    def tensor_digest(self, array):
        """Canonical value hash: source dtype + shape + lossless FP32 for BF16.

        Floating values here are BF16/FP16/FP32, so FP32 conversion is exact.
        Integer packed values retain their native integer representation.
        This is a value identity hash, explicitly not a safetensors file hash.
        Bounded chunks avoid materializing a full embedding in host memory.
        """
        dtype, shape = str(array.dtype), list(array.shape)
        h = hashlib.sha256(wire({"dtype": dtype, "shape": shape}))
        size = math.prod(shape)
        flat = array.reshape(-1)
        for start in range(0, size, 1024**2):
            self.check()
            chunk = flat[start:start + 1024**2]
            if "float" in dtype:
                chunk = chunk.astype(self.mx.float32)
            self.mx.eval(chunk)
            h.update(self.np.asarray(chunk).tobytes(order="C"))
        return {"dtype": dtype, "shape": shape, "elements": size,
                "canonical_value_sha256": h.hexdigest()}

    def head_inventory(self, model):
        entries = {}
        for name, tensor in model.head.state_dict().items():
            require(tensor.dtype == self.torch.float32 and tensor.device.type == "cpu",
                    "Native pointer head must remain CPU FP32")
            require(bool(self.torch.isfinite(tensor).all()), "Nonfinite native head")
            value = tensor.detach().contiguous().numpy()
            entries[name] = {"shape": list(value.shape), "dtype": str(tensor.dtype),
                             "sha256": sha(value.tobytes())}
        return {"tensors": entries, "temperature": float(model.head.temperature),
                "sha256": sha(wire(entries))}

    def inventory(self, model):
        from mlx.utils import tree_flatten
        nn = self.nn
        modules = dict(model.text.named_modules())
        linear = {name: mod for name, mod in modules.items() if isinstance(mod, nn.Linear)}
        quantized = {name: mod for name, mod in modules.items()
                     if isinstance(mod, nn.QuantizedLinear)}
        tensors = {name: self.tensor_digest(value)
                   for name, value in tree_flatten(model.text.parameters())}
        return {"tensors": tensors,
                "linears": {name: {"shape": list(mod.weight.shape),
                                    "dtype": str(mod.weight.dtype)}
                            for name, mod in linear.items()},
                "quantized": {name: {"bits": mod.bits, "group_size": mod.group_size,
                                      "mode": mod.mode, "class": type(mod).__name__}
                              for name, mod in quantized.items()},
                "head": self.head_inventory(model)}

    def load(self, snapshot, variant, baseline=None):
        from kev.checkpoint import Checkpoint, LoadOptions
        self.check()
        mount_guard(Path(self.args.root), self.args.volume_uuid)
        if self.first_load is None:
            self.first_load = time.monotonic()
        self.journal.write("load_start", variant=variant, first_load=self.first_load,
                           memory=self.memory())
        started = time.monotonic()
        ck = Checkpoint(str(snapshot))
        require(ck.meta.base == BASE_REPO and ck.meta.base_revision == BASE_REVISION,
                "Checkpoint base pin mismatch")
        require(not ck.full and not ck.meta.option_isolation and not ck.meta.special_embeddings,
                "Unexpected native checkpoint layout")
        require(type(ck.meta.temperature) in (int, float)
                and math.isfinite(ck.meta.temperature) and ck.meta.temperature > 0,
                "Pinned native head temperature must be a finite positive scalar")
        # Exact candidate scalar in the pinned runs/release/kev-4b-r10.json.
        # The model-card 2.41 is rounded; result.json's 2.297... refit was not adopted.
        require(math.isclose(ck.meta.temperature, 2.406050072164233,
                             rel_tol=0, abs_tol=1e-12),
                "Pinned head temperature differs from exact release metadata")
        self.journal.write("checkpoint_metadata", variant=variant,
                           exact_native_temperature=ck.meta.temperature,
                           head_file_sha256=file_hash(snapshot / "head.pt"),
                           temperature_policy="preserve pinned head scalar; no display-value substitution")
        import kev.checkpoint as checkpoint_module
        from experiments.kev4b_v1.local_resolver import local_snapshot_resolution
        with local_snapshot_resolution(checkpoint_module, self.verified_snapshots,
                                       cache_root=Path(self.args.root) / "hf/hub") as resolver:
            self.journal.write("local_snapshot_resolution", variant=variant,
                               resolver=resolver.to_record())
            tok, model = ck.load("mps", LoadOptions(backend="mlx", merge=True,
                                                  lora_scale=1.0, temperature=None))
        self.check()
        require(model.backend == "mlx" and model.device == "mlx" and model.dtype == "bfloat16",
                "Unexpected native backend or stored dtype")
        self.mx.eval(model.text.parameters())
        before = self.inventory(model)
        require(len(before["linears"]) == EXPECTED_LINEARS and not before["quantized"],
                "Fresh native linear module inventory mismatch")
        require(sum(math.prod(v["shape"]) for v in before["linears"].values())
                == EXPECTED_LINEAR_WEIGHTS, "Eligible linear weight count mismatch")
        require(all(v["dtype"] == "mlx.core.bfloat16" and v["shape"][-1] % 64 == 0
                    for v in before["linears"].values()),
                "Native linear dtype or quantization group eligibility mismatch")
        require(before["head"]["temperature"] == ck.meta.temperature, "Temperature changed during load")
        if baseline is not None:
            require(before == baseline, "Fresh native model differs from BF16 baseline before quantization")
        self.journal.write("native_inventory", variant=variant, inventory=before,
                           load_and_hash_seconds=time.monotonic() - started,
                           memory=self.memory())
        if variant != "bf16":
            bits = {"mlx-affine8-g64": 8, "mlx-affine4-g64": 4}[variant]
            mount_guard(Path(self.args.root), self.args.volume_uuid)
            self.journal.write("quantize_start", variant=variant, bits=bits,
                               group_size=64, mode="affine", quantize_input=False,
                               scope="text nn.Linear only", memory=self.memory())
            started = time.monotonic()
            self.nn.quantize(model.text, group_size=64, bits=bits, mode="affine",
                             quantize_input=False,
                             class_predicate=lambda _, module: isinstance(module, self.nn.Linear))
            self.mx.eval(model.text.parameters())
            self.check()
            after = self.inventory(model)
            expected = set(before["linears"])
            require(set(after["quantized"]) == expected and not after["linears"],
                    "Quantized module scope mismatch")
            require(all(q == {"bits": bits, "group_size": 64, "mode": "affine",
                              "class": "QuantizedLinear"} for q in after["quantized"].values()),
                    "Quantization parameters differ from reviewed recipe")
            changed_weights = {name + ".weight" for name in expected}
            fixed = {name: value for name, value in before["tensors"].items()
                     if name not in changed_weights}
            require(all(after["tensors"].get(name) == value for name, value in fixed.items()),
                    "A retained embedding, norm, convolution, recurrent or bias tensor changed")
            require(after["head"] == before["head"], "Native FP32 head changed")
            allowed_added = {name + suffix for name in expected for suffix in (".scales", ".biases")}
            require(set(after["tensors"]) - set(before["tensors"]) <= allowed_added
                    and set(before["tensors"]) <= set(after["tensors"]),
                    "Unexpected quantization tensor names")
            self.journal.write("quantized_inventory", variant=variant, inventory=after,
                               seconds=time.monotonic() - started, memory=self.memory(),
                               derivation="fresh_native_merge_then_in_memory_quantization",
                               persisted_variant=False, export_reload_validated=False)
        return tok, model, before

    def evaluate(self, tok, model, request, *, case, phase, variant):
        from kev.api import SystemOneRequest, to_record, to_answers, output_tokens
        from kev.model import rows_of
        from kev.mlx_model import PREFILL_CHUNK
        self.check()
        request_bytes = wire(request)
        self.journal.write("request", case=case["id"], phase=phase, variant=variant,
                           request_base64=base64.b64encode(request_bytes).decode(),
                           request_sha256=sha(request_bytes))
        rec, meta = to_record(SystemOneRequest(**request))
        enc = model.encode(tok, rec, max_state=MAX_TOKENS, max_branch=MAX_TOKENS, strict=True)
        state, _, rows = rows_of(enc)
        require(len(rows) == 1 and len(meta) == 1 and meta[0]["keys"] == ACTIONS,
                "Exactly one native four-option question required")
        require(not enc["state_truncated"] and len(enc["ids"]) <= MAX_TOKENS
                and len(state) + len(rows[0]["ids"]) <= MAX_TOKENS,
                "Complete native row exceeds strict token budget")
        require(len(rows[0]["opts"]) == 4, "Native option readout count changed")
        self.journal.write("encoded", case=case["id"], phase=phase, variant=variant,
                           native_record=rec, native_meta=meta, encoding=enc,
                           complete_row_tokens=len(enc["ids"]))
        require(self.calls < MAX_DECISIONS, "Cumulative 64-decision ceiling reached")
        self.calls += 1  # Reserve BEFORE native evaluation, including failed calls.
        number = self.calls
        self.journal.write("decision_start", number=number, case=case["id"], phase=phase,
                           variant=variant, memory=self.memory())
        # Count actual internal backbone passes without changing computation. The
        # class wrapper only observes this exact text instance and is restored.
        text_type, text_instance = type(model.text), model.text
        original = text_type.__call__
        passes = []
        expected_prefix_passes = math.ceil(len(state) / PREFILL_CHUNK)

        def counted(instance, *args, **kwargs):
            if instance is text_instance:
                self.check()
                shape = list(args[0].shape) if args else None
                passes.append({"input_shape": shape, "cache_supplied": kwargs.get("cache") is not None,
                               "native_stage": ("state_prefix" if len(passes) < expected_prefix_passes
                                                else "question_branch")})
                self.journal.write("backbone_pass_start", number=number,
                                   pass_number=len(passes), **passes[-1])
            return original(instance, *args, **kwargs)

        started = time.monotonic()
        text_type.__call__ = counted
        try:
            logits = model.forward(enc)  # EXACTLY ONE decision evaluation.
            require(len(logits) == 1 and tuple(logits[0].shape) == (4,),
                    "Unexpected native logits shape")
            raw_logits = [z.detach().cpu().tolist() for z in logits]
            probabilities = [self.torch.softmax(z, -1).detach().cpu().tolist() for z in logits]
        finally:
            text_type.__call__ = original
        seconds = time.monotonic() - started
        # Store raw values before contract/parity failures. JSON nonfinite values
        # are represented explicitly as strings rather than invalid JSON.
        def finite_json(values):
            return [[v if math.isfinite(v) else repr(v) for v in row] for row in values]
        self.journal.write("raw_decision", number=number, case=case["id"], phase=phase,
                           variant=variant, logits=finite_json(raw_logits),
                           probabilities=finite_json(probabilities),
                           forward_wall_seconds_including_instrumentation=seconds,
                           internal_backbone_passes=passes, memory=self.memory())
        require(all(math.isfinite(v) for row in raw_logits + probabilities for v in row),
                "Native logits/probabilities contain nonfinite values")
        require(len(passes) == expected_prefix_passes + 1,
                "Unexpected internal prefix/branch evaluation count")
        require(abs(math.fsum(probabilities[0]) - 1) <= 1e-6, "Native softmax sum invalid")
        answers = to_answers(probabilities, meta)
        response = {"model": "kev-latest", "answers": answers, "truncated": False,
                    "usage": {"input_tokens": len(enc["ids"]),
                              "output_tokens": output_tokens(tok, answers),
                              "state_tokens": enc["state_tokens"],
                              "state_tokens_used": len(state)}}
        response_bytes = wire(response)
        self.journal.write("native_response", number=number,
                           response_base64=base64.b64encode(response_bytes).decode(),
                           response_sha256=sha(response_bytes))
        selected = self.adapter.parse_response(response_bytes)
        require(selected == ACTIONS[max(range(4), key=lambda i: probabilities[0][i])],
                "Native returned action differs from unrounded argmax")
        self.check()
        return {"number": number, "encoding": enc, "probabilities": probabilities[0],
                "logits": raw_logits[0], "choice": selected,
                "request_sha256": sha(request_bytes), "response_bytes": response_bytes}

    def native(self, tok, model, case, phase, variant):
        request = deepcopy(case["request"])
        request["model"] = "kev-latest"
        return self.evaluate(tok, model, request, case=case, phase=phase, variant=variant)

    def wrapped(self, tok, model, case, phase, variant):
        from clef_snake.game import Game, QUESTION
        # Hydrate a frozen request object without Game.__init__, spawn or apply.
        # Game.request recomputes every factual feature independently. No state
        # transition, game loop, food generation, or played game occurs.
        game = Game.__new__(Game)
        state = deepcopy(case["request"]["state"])
        game.body, game.direction, game.food = state["body"], state["direction"], state["food"]
        game.alive = True
        original_question = deepcopy(QUESTION)
        expected = deepcopy(case["request"])
        expected["model"] = "kev-latest"
        require(self.adapter.make_request(game) == wire(expected),
                "Reconstructed adapter request differs from immutable fixture")
        observations = []

        def transport(raw):
            require(not observations, "Unexpected adapter retry")
            outcome = self.evaluate(tok, model, json.loads(raw), case=case,
                                    phase=phase, variant=variant)
            observations.append(outcome)
            return self.adapter.TransportResponse(200, outcome["response_bytes"])

        attempt = self.adapter.attempt(game, transport)
        self.journal.write("adapter_attempt", case=case["id"], phase=phase,
                           variant=variant, attempt=attempt.to_record())
        require(attempt.error is None, f"Adapter technical stop: {attempt.error}")
        require(len(observations) == 1, "Adapter did not make exactly one native call")
        require(QUESTION == original_question and self.adapter.make_request(game) == wire(expected),
                "Request construction changed the frozen state or QUESTION")
        require(attempt.choice == observations[0]["choice"], "Adapter changed native choice")
        return observations[0]

    def compare(self, left, right, *, case, kind, required):
        same_encoding = left["encoding"] == right["encoding"]
        difference = max(abs(a - b) for a, b in zip(left["probabilities"], right["probabilities"]))
        same_choice = left["choice"] == right["choice"]
        self.journal.write("comparison", case=case["id"], kind=kind,
                           left_decision=left["number"], right_decision=right["number"],
                           exact_encoding=same_encoding,
                           exact_request=left["request_sha256"] == right["request_sha256"],
                           same_choice=same_choice, maximum_probability_difference=difference,
                           acceptance_required=required, tolerance=PARITY_TOLERANCE if required else None)
        require(same_encoding and left["request_sha256"] == right["request_sha256"],
                "Input parity failed")
        if required:
            require(same_choice and difference <= PARITY_TOLERANCE,
                    "BF16 numerical/choice parity failed; stopping without retries")


def worker(args):
    journal = Journal(Path(args.output) / "events.jsonl")
    runtime = Runtime(args, journal)
    try:
        sys.dont_write_bytecode = True
        sys.addaudithook(no_network)
        journal.write("worker_start", pid=os.getpid(), limits={"seconds": MAX_SECONDS,
                      "decisions": MAX_DECISIONS, "planned_decisions": PLANNED_DECISIONS,
                      "rss_bytes": MAX_MEMORY, "mlx_active_bytes": MAX_MEMORY, "games": 0},
                      cache_policy="fresh native intrarequest cache; no cross-request reuse",
                      cpu_policy="upstream CPU LoRA merge and FP32 pointer head only")
        root, repo, source, snapshots, fixtures = verify_inputs(args, journal)
        runtime.check()
        runtime.verified_snapshots = snapshots
        ids = [case["id"] for case in fixtures]
        journal.write("run_plan", planned_decisions=PLANNED_DECISIONS,
                      phases=[
                          {"variant": "bf16", "phase": "warmup_native", "cases": ids[:2]},
                          {"variant": "bf16", "phase": "warmup_wrapped", "cases": ids[:2]},
                          {"variant": "bf16", "phase": "native", "cases": ids},
                          {"variant": "bf16", "phase": "wrapped", "cases": ids},
                          {"variant": "bf16", "phase": "repeatability", "cases": ids[:4]},
                          {"variant": "mlx-affine8-g64", "phase": "warmup_quantized", "cases": ids[:2]},
                          {"variant": "mlx-affine8-g64", "phase": "quantized", "cases": ids},
                          {"variant": "mlx-affine4-g64", "phase": "warmup_quantized", "cases": ids[:2]},
                          {"variant": "mlx-affine4-g64", "phase": "quantized", "cases": ids}],
                      games=0, retries=0, cross_request_cache=False,
                      serialized_quantized_artifacts=False)
        sys.path[:0] = [str(source), str(repo)]
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        import torch
        from experiments.kev4b_v1 import adapter
        import kev.checkpoint as checkpoint_module
        require(Path(checkpoint_module.__file__).resolve() == source / "kev/checkpoint.py",
                "Imported Kev source differs from verified source")
        require(Path(adapter.__file__).resolve() == repo / "experiments/kev4b_v1/adapter.py",
                "Imported adapter differs from reviewed checkout")
        runtime.mx, runtime.nn, runtime.np, runtime.torch, runtime.adapter = mx, nn, np, torch, adapter
        versions = {name: importlib.metadata.version(name)
                    for name in ("mlx", "mlx-metal", "mlx-lm", "numpy", "torch", "transformers", "peft")}
        provenance = json.loads((repo / "experiments/kev4b_v1/provenance.json").read_text())
        locked_versions = provenance["runtime_plan"]["author_lock_versions"]
        require(versions == {name: locked_versions[name] for name in versions},
                "All seven reviewed author-lock runtime versions are required")
        require(mx.metal.is_available(), "MLX Metal GPU unavailable; CPU fallback forbidden")
        mx.set_default_device(mx.gpu)
        journal.write("runtime", versions=versions, python=sys.version, platform=platform.platform(),
                      imported_source=str(checkpoint_module.__file__), memory=runtime.memory())
        tok, model, baseline_inventory = runtime.load(snapshots[KEV_REPO], "bf16")
        for case in fixtures[:2]:
            runtime.native(tok, model, case, "warmup_native", "bf16")
        for case in fixtures[:2]:
            runtime.wrapped(tok, model, case, "warmup_wrapped", "bf16")
        baseline = [runtime.native(tok, model, case, "native", "bf16") for case in fixtures]
        wrapped = []
        for case, reference in zip(fixtures, baseline):
            result = runtime.wrapped(tok, model, case, "wrapped", "bf16")
            runtime.compare(reference, result, case=case, kind="native_vs_wrapper", required=True)
            wrapped.append(result)
        for case, reference in zip(fixtures[:4], wrapped[:4]):
            result = runtime.wrapped(tok, model, case, "repeatability", "bf16")
            runtime.compare(reference, result, case=case, kind="bf16_repeat", required=True)
        require(runtime.head_inventory(model) == baseline_inventory["head"], "BF16 head changed")
        del model, tok
        gc.collect()
        mx.clear_cache()
        journal.write("unloaded", variant="bf16", memory=runtime.memory())
        for variant in ("mlx-affine8-g64", "mlx-affine4-g64"):
            tok, model, _ = runtime.load(snapshots[KEV_REPO], variant, baseline_inventory)
            for case in fixtures[:2]:
                runtime.wrapped(tok, model, case, "warmup_quantized", variant)
            for case, reference in zip(fixtures, baseline):
                result = runtime.wrapped(tok, model, case, "quantized", variant)
                runtime.compare(reference, result, case=case, kind=variant + "_vs_bf16", required=False)
            require(runtime.head_inventory(model) == baseline_inventory["head"], "Quantized head changed")
            del model, tok
            gc.collect()
            mx.clear_cache()
            journal.write("unloaded", variant=variant, memory=runtime.memory())
            runtime.check()
        require(runtime.calls == PLANNED_DECISIONS, "Planned decision accounting mismatch")
        journal.write("completed", decisions=runtime.calls, games=0,
                      seconds_since_first_load=time.monotonic() - runtime.first_load,
                      export_reload_validated=False, campaign_authorized=False,
                      memory=runtime.memory())
        return 0
    except BaseException as exc:
        journal.write("fatal", error_type=type(exc).__name__, message=str(exc),
                      traceback=traceback.format_exc(), decisions_reserved=runtime.calls,
                      games=0, memory=runtime.memory())
        return 1
    finally:
        journal.close()


def supervise(args):
    # The authorized launcher creates this exclusive receipt before exec. It is
    # never removed, including admission failures before the budget claim.
    receipt_path = ROOT / "evidence/retry-launch-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    started = receipt["accounting_started_monotonic"]
    require(type(started) in (int, float) and math.isfinite(started)
            and 0 <= time.monotonic() - started < 719,
            "Retry launch receipt is missing a live remaining-budget start")
    require(receipt["output"] == str(Path(args.output).absolute())
            and receipt["harness_sha256"] == file_hash(Path(__file__)),
            "Retry launch receipt does not bind this output and harness")
    journal = None
    proc = None
    last_resource_log = 0.0
    prior_handler = signal.getsignal(signal.SIGALRM)

    def kill_worker():
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def hard_deadline(signum, frame):
        # The timer interrupts a blocked ps/journal operation; it cannot permit
        # those operations to extend the worker's inference deadline.
        if proc is not None and proc.poll() is None:
            kill_worker()
        raise TimeoutError("Aggregate remaining preflight deadline reached; worker killed")

    try:
        signal.signal(signal.SIGALRM, hard_deadline)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, started + 719 - time.monotonic()))
        root = Path(args.root).absolute()
        mount = mount_guard(root, args.volume_uuid)
        require(Path(sys.prefix).resolve() == root / "venv", "Dedicated venv required")
        out = safe_path(args.output, root, exists=False)
        require(out.is_relative_to(root / "preflight"), "Output must be under ROOT/preflight")
        require(not out.exists(), "Output exists; retries/resume are forbidden")
        storage = json.loads((Path(args.repo) / "experiments/kev4b_v1/storage-plan.json").read_text())
        env = offline_environment(root, storage)
        require(args.authorized_retry, "This harness requires the explicitly authorized remaining-budget retry")
        # Import only the stdlib ledger before any model/runtime imports. It verifies
        # all previous attempts and atomically prevents the allowance being reused.
        sys.path.insert(0, str(Path(args.repo).resolve()))
        from experiments.kev4b_v1.retry_budget import claim_retry_budget
        prior_evidence = claim_retry_budget(root / "preflight", out)
        require(prior_evidence["prior_decisions"] == 0
                and 0 < prior_evidence["remaining_seconds"] <= 719,
                "Authorized remaining allowance invalid")
        deadline = started + prior_evidence["remaining_seconds"]
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        out.parent.mkdir(parents=True, exist_ok=True)
        safe_path(out.parent, root)
        out.mkdir()
        journal = Journal(out / "supervisor.jsonl")
        env["KEV_PREFLIGHT_DEADLINE"] = str(deadline)
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--root", str(root), "--repo", str(Path(args.repo).resolve()),
                   "--output", str(out), "--volume-uuid", args.volume_uuid, "--_worker"]
        journal.write("supervisor_start", pid=os.getpid(), mount=mount, deadline=deadline,
                      deadline_scope="aggregate prior time debited; retry includes verification/import/load/hash/quantize/unload",
                      retry_budget=prior_evidence, accounting_started_monotonic=started,
                      launch_receipt_sha256=file_hash(receipt_path),
                      harness_sha256=file_hash(Path(__file__)), command=command,
                      maximum_rss_bytes=MAX_MEMORY)
        with (out / "worker.stdout.txt").open("x") as stdout, (out / "worker.stderr.txt").open("x") as stderr:
            proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True, cwd=root)
            signal.signal(signal.SIGALRM, hard_deadline)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
            while proc.poll() is None:
                reason = None
                if time.monotonic() >= deadline:
                    reason = "aggregate_remaining_wall_limit"
                current_rss = None
                if reason is None:
                    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(proc.pid)],
                                         capture_output=True, text=True,
                                         timeout=min(2, max(0.001, deadline - time.monotonic())))
                    if rss.returncode != 0:
                        require(proc.poll() is not None,
                                "Cannot measure RSS for live worker: ps returned nonzero")
                    elif rss.stdout.strip().isdigit():
                        current_rss = int(rss.stdout.strip()) * 1024
                    else:
                        require(proc.poll() is not None,
                                "Cannot measure RSS for live worker: ps returned no numeric value")
                    if current_rss is not None and current_rss > MAX_MEMORY:
                        reason = "48_GiB_current_RSS_limit"
                    if directory_bytes(out) > MAX_EVIDENCE:
                        reason = "1_GiB_evidence_limit"
                    if time.monotonic() >= deadline:
                        reason = "aggregate_remaining_wall_limit"
                if reason:
                    # No grace period can extend GPU work beyond the hard bound.
                    kill_worker()
                    journal.write("worker_killed", reason=reason, current_rss_bytes=current_rss)
                    proc.wait(timeout=10)
                    return 1
                if time.monotonic() - last_resource_log >= 5:
                    journal.write("resource_sample", worker_pid=proc.pid,
                                  current_rss_bytes=current_rss, evidence_bytes=directory_bytes(out))
                    last_resource_log = time.monotonic()
                try:
                    proc.wait(timeout=min(0.5, max(0.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            journal.write("worker_exit", returncode=proc.returncode,
                          wall_seconds=time.monotonic() - started)
            return proc.returncode
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            kill_worker()
            proc.wait(timeout=10)
        if journal is not None:
            journal.write("supervisor_error", error_type=type(exc).__name__, message=str(exc),
                          wall_seconds=time.monotonic() - started)
        else:
            # An existing one-use claim remains consumed on setup failure.
            # Fail closed before any model work; stderr remains launch evidence.
            print(wire({"event": "supervisor_setup_error", "error_type": type(exc).__name__,
                        "message": str(exc), "wall_seconds": time.monotonic() - started}).decode(),
                  file=sys.stderr, flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        if journal is not None:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--volume-uuid", required=True)
    parser.add_argument("--authorized-retry", action="store_true",
                        help="Use the one authorized remaining-budget retry; never reset the 900s total")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        require("KEV_PREFLIGHT_DEADLINE" in os.environ, "Worker requires supervisor deadline")
        require(0 < float(os.environ["KEV_PREFLIGHT_DEADLINE"]) - time.monotonic() <= MAX_SECONDS,
                "Worker deadline is absent, expired or exceeds maximum")
        return worker(args)
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
