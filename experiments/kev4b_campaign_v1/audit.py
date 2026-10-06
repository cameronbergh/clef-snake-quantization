"""Independent, stdlib-only audit of the frozen Kev campaign evidence.

No runner, adapter, game, tokenizer or model implementation is imported. Audit is
read-only. A truncated journal or technical interruption is never scored as a
loss, and synthetic fixtures cannot receive a real-campaign pass.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

VARIANTS = ("bf16", "mlx-affine8-g64", "mlx-affine4-g64")
ACTIONS = ("up", "down", "left", "right")
VECTORS = ((0, -1), (0, 1), (-1, 0), (1, 0))
INSTRUCTIONS = ("Select the next move for the snake head. Avoid collisions and try to "
                "reach the food while preserving open space. No program selects or "
                "overrides your answer.")
CRITERIA = dict(zip(ACTIONS, ("Move north (y - 1)", "Move south (y + 1)",
                            "Move west (x - 1)", "Move east (x + 1)")))
MAX_GAME_CALLS, MAX_WARMUPS, MAX_CALLS = 45000, 180, 45180
HEAD_TEMPERATURE = 2.406050072164233


class EvidenceError(ValueError):
    """Evidence contradicts the frozen protocol or cannot be parsed safely."""


class IncompleteEvidence(EvidenceError):
    """Evidence ended before a complete result was durably recorded."""


def need(condition, message):
    if not condition:
        raise EvidenceError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def number(text):
        value = float(text)
        need(math.isfinite(value), "Nonfinite JSON number")
        return value

    def constant(_):
        raise EvidenceError("Nonfinite JSON constant")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                          parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("Malformed JSON or UTF-8 evidence") from exc


def exact(a, b, message):
    # JSON equality alone conflates booleans and integers and ignores key order.
    need(wire(a) == wire(b), message)


def body(record, prefix, *, optional=False):
    value = record.get(prefix + "_base64")
    if value is None and optional:
        need(record.get(prefix + "_sha256") is None, "Hash without raw bytes")
        return None
    need(isinstance(value, str), "Missing lossless " + prefix + " bytes")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise EvidenceError("Invalid base64 " + prefix) from exc
    need(base64.b64encode(raw).decode("ascii") == value, "Noncanonical base64")
    need(digest(raw) == record.get(prefix + "_sha256"), prefix + " hash mismatch")
    return raw


class ReferenceBoard:
    """Independent reconstruction of the documented 12x12 food/game rules."""

    def __init__(self, seed):
        need(isinstance(seed, str) and seed.isascii(), "Invalid seed")
        self.seed = seed
        self.body = [(5, 6), (4, 6), (3, 6), (2, 6)]
        self.direction, self.score, self.steps, self.alive = "right", 0, 0, True
        self.food_events = []
        self._food()

    def _food(self):
        # Permute row-major cell numbers, independently of movement/body state.
        cells = list(range(144))
        for j in range(143, 0, -1):
            hashed = hashlib.sha256(("clef-snake-food-v1:%s:%d:%d" %
                                     (self.seed, self.score, j)).encode("ascii"))
            k = int(hashed.hexdigest(), 16) % (j + 1)
            cells[j], cells[k] = cells[k], cells[j]
        occupied = sorted(y * 12 + x for x, y in self.body)
        available = [(rank, cell) for rank, cell in enumerate(cells)
                     if cell not in occupied]
        rank, value = available[0] if available else (-1, 0)
        self.food = (value % 12, value // 12)
        self.food_events.append({"event": self.score, "rank": rank,
                                 "selected": list(self.food), "occupied": occupied})

    @staticmethod
    def _inside(point):
        return 0 <= point[0] < 12 and 0 <= point[1] < 12

    @staticmethod
    def open_space(start, blocked):
        if not ReferenceBoard._inside(start) or start in blocked:
            return 0
        visited, pending = {start}, deque([start])
        while pending:
            x, y = pending.popleft()
            for dx, dy in VECTORS:
                point = x + dx, y + dy
                if ReferenceBoard._inside(point) and point not in blocked and point not in visited:
                    visited.add(point)
                    pending.append(point)
        return len(visited)

    def request(self):
        x, y = self.body[0]
        fx, fy = self.food
        dx, dy = fx - x, fy - y
        blocked = set(self.body[:-1])
        safe, analyses = [], {}
        for name, (vx, vy) in zip(ACTIONS, VECTORS):
            target = x + vx, y + vy
            if len(self.body) > 1 and target == self.body[1]:
                reason = "neck"
            elif not self._inside(target):
                reason = "wall"
            elif target in blocked:
                reason = "body"
            else:
                reason = "safe"
                safe.append(name)
            distance = abs(fx - target[0]) + abs(fy - target[1])
            analyses[name] = {"safe": reason == "safe", "reason": reason,
                              "manhattan_to_food": distance,
                              "open_space": self.open_space(target, blocked) if reason == "safe" else 0,
                              "reduces_distance": distance < abs(dx) + abs(dy)}
        state = {"grid": {"width": 12, "height": 12,
                          "coordinates": "x increases right, y increases down"},
                 "body": [list(p) for p in self.body], "direction": self.direction,
                 "head": [x, y], "food": [fx, fy],
                 "food_delta": {"dx": dx, "dy": dy,
                                "preferred_horizontal": "right" if dx > 0 else "left" if dx < 0 else "aligned",
                                "preferred_vertical": "down" if dy > 0 else "up" if dy < 0 else "aligned"},
                 "safe_moves": safe, "move_analysis": analyses}
        return {"model": "kev-latest", "state": state,
                "questions": {"move": {"type": "choice", "instructions": INSTRUCTIONS,
                                        "criteria": dict(CRITERIA)}}}

    def apply(self, action):
        need(self.alive and self.steps < 500 and action in ACTIONS,
             "Attempt after terminal game/cap or unknown action")
        dx, dy = VECTORS[ACTIONS.index(action)]
        target = self.body[0][0] + dx, self.body[0][1] + dy
        if not self._inside(target) or target in self.body[:-1]:
            self.alive = False
            return
        self.body.insert(0, target)
        self.direction = action
        if target == self.food:
            self.score += 1
            self._food()
        else:
            self.body.pop()
        self.steps += 1

    def snapshot(self):
        return {"snake": [list(p) for p in self.body], "food": list(self.food),
                "score": self.score, "steps": self.steps, "alive": self.alive}


def native_record(request):
    """Independently implement the pinned native API's textual record shape."""
    def render(value, depth=0):
        pad = "  " * depth
        if isinstance(value, dict):
            lines = []
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    lines.append(pad + key + ":\n" + render(child, depth + 1))
                else:
                    lines.append(pad + key + ": " + render(child))
            return "\n".join(lines)
        if isinstance(value, list):
            return "\n".join(pad + "- " + render(child, depth + 1).lstrip()
                             for child in value)
        return "" if value is None else str(value)
    q = request["questions"]["move"]
    return {"state": render(request["state"]),
            "questions": [{"instr": q["instructions"],
                           "options": [name + ": " + q["criteria"][name] for name in ACTIONS],
                           "label": 0}]}


def verify_encoding(event, request):
    exact(event["native_record"], native_record(request), "Native textual record drift")
    exact(event["native_meta"], [{"id": "move", "type": "choice", "keys": list(ACTIONS)}],
          "Native question/options metadata drift")
    enc = event["encoding"]
    ids = enc["ids"]
    need(isinstance(ids, list) and 0 < len(ids) <= 8192 and all(integer(x) for x in ids),
         "Invalid or oversized full token row")
    n, s = len(ids), enc["state_tokens"]
    need(integer(s, 1) and s < n and enc["state_truncated"] is False
         and enc["option_isolation"] is False, "Truncation/isolation/state length mismatch")
    exact(enc["seg"], [0] * s + [1] * (n - s), "Token segmentation mismatch")
    exact(enc["pos"], list(range(n)), "Token position mismatch")
    exact(enc["decide_idx"], [n - 1], "Decision token position mismatch")
    exact(enc["labels"], [0], "Native label mismatch")
    opts = enc["opt_idx"]
    need(isinstance(opts, list) and len(opts) == 1 and len(opts[0]) == 4,
         "Exactly four native option readouts required")
    ends = opts[0]
    need(all(integer(x) and s < x < n - 1 for x in ends)
         and all(a < b for a, b in zip(ends, ends[1:])) and ends[-1] == n - 2,
         "Option readout position/order mismatch")
    assignment = enc["opt"]
    need(len(assignment) == n and assignment[-1] == -2
         and all(type(x) is int for x in assignment), "Option token map mismatch")
    first = next((i for i, value in enumerate(assignment) if value == 0), None)
    need(first is not None and first > s and all(x == -1 for x in assignment[:first]),
         "Option prefix map mismatch")
    start = first
    for index, end in enumerate(ends):
        need(end >= start + 1 and assignment[start:end + 1] == [index] * (end + 1 - start),
             "Option span map mismatch")
        start = end + 1
    need(event["complete_row_tokens"] == n, "Full native token count mismatch")
    return s, n


def verify_raw(event, response, encoding):
    logits, probs = event["logits"], event["probabilities"]
    need(isinstance(logits, list) and len(logits) == 1 and len(logits[0]) == 4
         and isinstance(probs, list) and len(probs) == 1 and len(probs[0]) == 4,
         "Native raw decision must contain one four-option row")
    z, p = logits[0], probs[0]
    need(all(finite(v) for v in z + p) and all(0 <= v <= 1 for v in p),
         "Nonfinite or invalid native decision")
    normalizer = math.fsum(math.exp(v - max(z)) for v in z)
    softmax = [math.exp(v - max(z)) / normalizer for v in z]
    need(abs(math.fsum(p) - 1) <= 1e-6 and
         all(abs(a - b) <= 2e-6 for a, b in zip(p, softmax)), "Raw logits/softmax disagree")
    index = max(range(4), key=p.__getitem__)
    need(event.get("selected_index") == index and event.get("choice") == ACTIONS[index],
         "Native unrounded argmax changed (including first-option tie rule)")
    need(response.get("model") == "kev-latest" and response.get("truncated") is False
         and response.get("error") is None, "Native API response identity/error/truncation")
    need(list(response["answers"]) == ["move"], "Unexpected response question")
    answer = response["answers"]["move"]
    need(answer["type"] == "choice" and answer["choice"] == ACTIONS[index],
         "Response choice differs from unrounded native argmax")
    exact(answer["probabilities"], {name: round(float(value), 4) for name, value in zip(ACTIONS, p)},
          "Serialized native probability rounding/order mismatch")
    confidence = round((max(p) / sum(p) - .25) / .75, 4)
    need(answer.get("confidence") == confidence, "Native confidence mismatch")
    usage = response["usage"]
    need(usage["input_tokens"] == len(encoding["ids"])
         and usage["state_tokens"] == encoding["state_tokens"]
         and usage["state_tokens_used"] == encoding["state_tokens"]
         and integer(usage["output_tokens"]), "Native usage/truncation mismatch")
    return ACTIONS[index]


def journal(path, source):
    """Stream strict newline-delimited evidence; reject missing/torn/empty rows."""
    previous = -math.inf
    try:
        stream = path.open("rb")
    except FileNotFoundError as exc:
        raise IncompleteEvidence("Missing " + path.name) from exc
    with stream:
        for sequence, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n"):
                raise IncompleteEvidence("Torn final line in " + path.name)
            row = strict_json(raw)
            need(isinstance(row, dict) and row.get("sequence") == sequence
                 and type(row.get("sequence")) is int, "Noncontiguous " + source + " journal")
            now = row.get("monotonic")
            need(finite(now) and now >= previous, "Nonmonotonic " + source + " journal")
            need(isinstance(row.get("event"), str), "Journal event missing")
            previous = now
            yield now, source, row


def merged_journals(root):
    # Common process monotonic clock orders durable reservations, native work,
    # adapter attempts and transitions across the two independently saved logs.
    return heapq.merge(journal(root / "runner.jsonl", "runner"),
                      journal(root / "native.jsonl", "native"), key=lambda x: x[0])

NATIVE_INVENTORY_SHA256 = "8d8c6cfe11874d898639c86fec0b1bbc1413d25a528bbef76f0afeac4edba641"
HEAD_FILE_SHA256 = "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c"
HEAD_INVENTORY_SHA256 = "362a712478d849967537009392f58dc463f828388f6403e334cf2160a400f9d8"


def canonical_sha(value):
    return digest(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def hexhash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_inventory(before, after, variant, *, synthetic=False):
    """Check full native identity, exact eligible scope, retained and packed tensors."""
    need(isinstance(before, dict) and isinstance(after, dict), "Missing tensor inventory")
    if not synthetic:
        need(canonical_sha(before) == NATIVE_INVENTORY_SHA256, "Native baseline differs from accepted preflight")
    head = before["head"]
    need(head["temperature"] == HEAD_TEMPERATURE and head["sha256"] == HEAD_INVENTORY_SHA256,
         "Pinned native head identity or calibration changed")
    need(digest(wire(head["tensors"])) == head["sha256"], "Head inventory content/hash mismatch")
    for tensor in head["tensors"].values():
        need(tensor["dtype"] == "torch.float32" and hexhash(tensor["sha256"]),
             "Head must remain FP32 with complete hashes")
    need(before["quantized"] == {} and len(before["linears"]) == 248,
         "Fresh native load must have exactly 248 unquantized linear modules")
    linears, tensors = before["linears"], before["tensors"]
    need(sum(math.prod(v["shape"]) for v in linears.values()) == 3569090560,
         "Eligible projection weight count changed")
    weights = {name + ".weight" for name in linears}
    need(len(set(tensors) - weights) == 178 and weights <= set(tensors),
         "Retained tensor count/scope changed")
    for name, module in linears.items():
        need(module["dtype"] == "mlx.core.bfloat16" and len(module["shape"]) == 2
             and module["shape"][1] % 64 == 0, "Native linear dtype/shape changed")
        need(tensors[name + ".weight"]["shape"] == module["shape"]
             and tensors[name + ".weight"]["dtype"] == module["dtype"],
             "Native module/tensor inventory disagrees")
    for inventory in (before, after):
        for value in inventory["tensors"].values():
            need(isinstance(value["shape"], list) and all(integer(n, 1) for n in value["shape"])
                 and value["elements"] == math.prod(value["shape"])
                 and hexhash(value["canonical_value_sha256"]), "Incomplete tensor evidence")
    if variant == "bf16":
        need(before == after, "BF16 inventory changed")
        return
    need(variant in VARIANTS, "Unknown precision condition")
    bits = 8 if variant == VARIANTS[1] else 4
    need(after["head"] == before["head"] and after["linears"] == {}
         and set(after["quantized"]) == set(linears), "Head/quantization module scope changed")
    expected = set(tensors) | {name + suffix for name in linears for suffix in (".scales", ".biases")}
    need(set(after["tensors"]) == expected, "Missing/extra packed or retained tensors")
    for name in set(tensors) - weights:
        need(after["tensors"][name] == tensors[name], "Retained tensor changed")
    for name, source in linears.items():
        need(after["quantized"][name] == {"bits": bits, "group_size": 64,
                                         "mode": "affine", "class": "QuantizedLinear"},
             "Quantization recipe changed")
        out, width = source["shape"]
        for suffix, shape, dtype in ((".weight", [out, width * bits // 32], "mlx.core.uint32"),
                                     (".scales", [out, width // 64], "mlx.core.bfloat16"),
                                     (".biases", [out, width // 64], "mlx.core.bfloat16")):
            actual = after["tensors"][name + suffix]
            need(actual["shape"] == shape and actual["dtype"] == dtype,
                 "Packed weight/scale/bias representation changed")


def read_frozen(repo):
    """Validate the frozen source and trial design without production helpers."""
    prep = repo / "experiments/kev4b_campaign_v1"
    freeze_path = prep / "freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    freeze = strict_json(freeze_bytes)
    files = freeze["files"]
    need(isinstance(files, dict) and files, "Empty freeze file inventory")
    for name, expected in files.items():
        relative = Path(name)
        need(not relative.is_absolute() and ".." not in relative.parts and hexhash(expected),
             "Invalid frozen file path/hash")
        path = repo / relative
        need(path.resolve().is_relative_to(repo.resolve()) and not path.is_symlink(),
             "Frozen path escapes repository")
        h = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
        need(h.hexdigest() == expected, "Frozen file changed: " + name)
    required = ["protocol.json", "seed-manifest.json", "schedule.json", "analysis-plan.json",
                "audit.py", "runner.py", "native.py", "context_check.py"]
    for name in ("clef_snake/game.py", "experiments/kev4b_v1/adapter.py",
                 "experiments/kev4b_v1/local_resolver.py", "experiments/kev4b_v1/preflight.py",
                 "experiments/kev4b_v1/provenance.json", "experiments/kev4b_v1/artifact-plan.json",
                 "experiments/kev4b_v1/preflight-results-2026-10-06.json", "data/2026-10-05/seed-manifest.json"):
        need(name in files, "Required source identity absent from freeze: " + name)
    for name in required:
        need("experiments/kev4b_campaign_v1/" + name in files,
             "Required campaign identity absent from freeze: " + name)
    protocol = strict_json((prep / "protocol.json").read_bytes())
    manifest = strict_json((prep / "seed-manifest.json").read_bytes())
    schedule = strict_json((prep / "schedule.json").read_bytes())["games"]
    need(protocol["campaign_id"] == "kev4b_campaign_v1" and protocol["conditions"] == list(VARIANTS)
         and protocol["inference_authorized_by_preparation"] is False, "Wrong campaign identity or condition family")
    seed_rows = manifest["seeds"]
    need(len(seed_rows) == 30 and [r["seed_round"] for r in seed_rows] == list(range(1, 31))
         and len({r["seed"] for r in seed_rows}) == 30, "Expected 30 distinct ordered seed units")
    need(all(isinstance(r["seed"], str) and len(r["seed"]) == 32
             and all(c in "0123456789abcdef" for c in r["seed"]) for r in seed_rows),
         "Invalid seed strings")
    exclusions = manifest["excluded_manifests"]
    need([r["path"] for r in exclusions] == ["data/2026-10-05/seed-manifest.json",
         "data/2026-10-05-fiveway/seed-manifest.json", "experiments/qwen3_snake_v1/seed-manifest.json"],
         "Seed exclusion sources changed")
    excluded = set()
    for entry in exclusions:
        need(entry["path"] in files and files[entry["path"]] == entry["sha256"],
             "Excluded seed manifest missing from freeze or hash changed")
        prior = strict_json((repo / entry["path"]).read_bytes())
        values = prior["seeds"] + prior.get("preflight_seeds", [])
        need(len(values) == entry["seed_count"] and all(isinstance(v, str) for v in values),
             "Excluded seed manifest count/schema changed")
        excluded.update(values)
    need(len(excluded) == manifest["excluded_distinct_seed_count"] == 51,
         "Distinct excluded seed count changed")
    need(not excluded & {r["seed"] for r in seed_rows}, "Campaign reused earlier experiment seeds")
    need(len(schedule) == 90 and [r["sequence"] for r in schedule] == list(range(1, 91)),
         "Expected 90 sequential games")
    need(len({r["game_id"] for r in schedule}) == 90, "Duplicate game identities")
    permutations, orders = Counter(), []
    for index, seed in enumerate(seed_rows):
        group = schedule[index * 3:index * 3 + 3]
        need(all(r["seed"] == seed["seed"] and r["seed_round"] == seed["seed_round"] for r in group),
             "Seed/model pairing drift")
        order = tuple(r["variant"] for r in group)
        need(set(order) == set(VARIANTS), "Every seed must have all three conditions")
        permutations[order] += 1
        orders.append(order)
    need(permutations == Counter({p: 5 for p in itertools.permutations(VARIANTS)}),
         "All six model orders must occur exactly five times")
    need(orders == list(itertools.permutations(VARIANTS)) * 5, "Prescribed condition permutation order changed")
    verify_context_spec(protocol, files)
    limits = protocol["limits"]
    fixed = {"campaign_wall_seconds": 43200, "decision_wall_seconds": 120,
             "max_decision_evaluations": MAX_CALLS, "max_games": 90,
             "successful_move_cap": 500, "max_input_tokens": 8192,
             "max_rss_bytes": 51539607552, "max_mlx_active_bytes": 51539607552,
             "max_campaign_evidence_bytes": 17179869184, "max_root_bytes": 60129542144,
             "min_free_bytes": 8589934592, "warmups_per_load": 2, "planned_loads": 90}
    need(limits == fixed and all(type(v) is int for v in limits.values()), "Frozen safety/experiment limits changed")
    warmup = protocol["warmups"]
    fixture_path = warmup["fixture_path"]
    need(fixture_path in files, "Warmup fixture not frozen")
    cases = strict_json((repo / fixture_path).read_bytes())["cases"]
    need(warmup["case_ids"] == [c["id"] for c in cases[:2]], "Warmup case selection changed")
    warmups = {}
    for case in cases[:2]:
        need(digest(wire(case["request"])) == case["request_sha256"], "Frozen warmup fixture hash mismatch")
        request = dict(case["request"])
        request["model"] = "kev-latest"
        warmups[case["id"]] = wire(request)
    return {"freeze": freeze, "freeze_sha256": digest(freeze_bytes), "protocol": protocol,
            "manifest": manifest, "schedule": schedule, "warmups": warmups,
            "source_hashes": {r["path"]: r["sha256"] for r in strict_json((repo / "experiments/kev4b_v1/provenance.json").read_bytes())["upstream_code"]["files"]},
            "acquisition_sha256": strict_json((repo / "experiments/kev4b_v1/preflight-results-2026-10-06.json").read_bytes())["source_provenance"]["acquisition_sha256"]}

EXPECTED_FILES_SHA256 = "0d2ba4a16303637bda6806e0f492e914b242b64bda6da39c1db68068aef7ce91"


def verify_inputs_event(row, frozen, *, synthetic=False):
    protocol = frozen["protocol"]
    pins = {k: v for k, v in protocol["source_pins"].items() if k != "upstream_code"}
    need(row["pins"] == pins and row["code_revision"] == protocol["source_pins"]["upstream_code"]
         and row["whole_source_git_clean"] is True, "Pinned native source/checkpoint identity changed")
    need(row["versions"] == protocol["runtime_identity"]["packages"], "Native dependency versions changed")
    expected_sources = frozen["source_hashes"]
    need(row["source_hashes"] == expected_sources, "Executed upstream source hashes changed")
    need(row["fixture_sha256"] == frozen["freeze"]["files"][protocol["warmups"]["fixture_path"]]
         and row["acquisition_sha256"] == frozen["acquisition_sha256"], "Acquisition/warmup identities changed")
    files = row["verified_files"]
    need(len(files) == 29 and all(set(f) == {"repo_id", "path", "bytes", "sha256"} for f in files),
         "Exact complete 29-file checkpoint inventory required")
    need(synthetic or canonical_sha(sorted(files, key=lambda r: (r["repo_id"], r["path"]))) == EXPECTED_FILES_SHA256,
         "Checkpoint/tokenizer/head file inventory differs from accepted preflight")


def verify_resolver(resolver, frozen):
    pins = {k: v for k, v in frozen["protocol"]["source_pins"].items() if k != "upstream_code"}
    need(resolver["kind"] == "pinned_local_snapshot_resolution_v1"
         and resolver["network_calls"] == 0
         and resolver["unknown_input_policy"] == "refuse_without_fallback",
         "Offline resolver network/fallback policy changed")
    need(resolver["pinned_base_identifier"] == "Qwen/Qwen3.5-4B-Base@" + pins["Qwen/Qwen3.5-4B-Base"],
         "Unpinned base identifier")
    cache = frozen["protocol"]["storage"]["asset_root"] + "/hf/hub"
    need(resolver["cache_root"] == cache and set(resolver["snapshots"]) == set(pins),
         "Resolver escaped verified local snapshot allowlist")
    for ident, revision in pins.items():
        need(resolver["snapshots"][ident] == {"revision": revision,
             "path": cache + "/models--" + ident.replace("/", "--") + "/snapshots/" + revision},
             "Resolved native checkpoint path/pin changed")


CONTEXT_TOKENIZER_HASHES = {
    "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}


def verify_context_spec(protocol, files):
    spec = protocol["context_validation"]
    need(spec == {"required_before_execution": True,
         "report_relative_path": "evidence/kev-campaign-context-2026-10-06.json",
         "utility_path": "experiments/kev4b_campaign_v1/context_check.py",
         "mode": "native_tokenizer_only", "synthetic_cases": 113,
         "status_at_freeze": "not_executed", "report_bound_in_external_approval": True,
         "no_model_loads_or_forwards": True}, "Frozen context-validation gate changed")
    need(spec["utility_path"] in files, "Context validation utility must be frozen")
    return spec


def context_fixtures():
    """Independently reconstruct static stress requests; never play a game.

    These are the declared geometric fixture construction, not imported helper
    output. No initializer, food draw, transition, tokenizer or model is called.
    """
    cycle = [(x, 0) for x in range(12)]
    cycle += [(x, y) for y in range(1, 12)
              for x in (range(11, 0, -1) if y % 2 else range(1, 12))]
    cycle += [(0, y) for y in range(11, 0, -1)]
    cases = []

    def add(case_id, cells, food, legacy=False):
        board = ReferenceBoard.__new__(ReferenceBoard)
        board.body, board.food = cells, food
        delta = tuple(cells[0][axis] - cells[1][axis] for axis in range(2))
        board.direction = ACTIONS[VECTORS.index(delta)]
        cases.append({"case_id": case_id, "body_length": len(cells),
                      "legacy_tail_food_duplicate": legacy,
                      "origin": "synthetic_context_stress_not_a_played_game",
                      "request": board.request()})

    for length, head, orientation in itertools.product((4, 16, 32, 64, 96, 128, 143, 144),
                                                      (0, 35, 72, 107), (1, -1)):
        cells = [cycle[(head + orientation * i) % 144] for i in range(length)]
        free = [cell for cell in cycle if cell not in set(cells)]
        foods = list(dict.fromkeys((free[0], free[-1]))) if free else [(0, 0)]
        for food_index, food in enumerate(foods):
            add(f"length-{length:03d}-head-{head:03d}-order-{orientation:+d}-food-{food_index}", cells, food)
    add("legacy-length-145-tail-food", [(0, 0)] + cycle[1:] + cycle[:1], (0, 0), True)
    return cases


def verify_context_report(report, frozen):
    """Check saved tokenizer-only evidence without importing its producer."""
    protocol = frozen["protocol"]
    spec = verify_context_spec(protocol, frozen["freeze"]["files"])
    need(report.get("schema_version") == 1 and report.get("mode") == spec["mode"]
         and report.get("status") == "passed" and report.get("case_count") == 113
         and report.get("synthetic", False) is False,
         "Completed native tokenizer-only context report required")
    need(report.get("utility_sha256") == frozen["freeze"]["files"][spec["utility_path"]],
         "Context report utility hash differs from freeze")
    cases = context_fixtures()
    need(report.get("fixtures_sha256") == canonical_sha(cases), "Context fixture hash mismatch")
    expected_provenance = {
        "base_repository": "Qwen/Qwen3.5-4B-Base",
        "base_revision": protocol["source_pins"]["Qwen/Qwen3.5-4B-Base"],
        "tokenizer_files_sha256": CONTEXT_TOKENIZER_HASHES,
        "kev_code_revision": protocol["source_pins"]["upstream_code"],
        "kev_files_sha256": {"kev/__init__.py": digest(b""),
                             **{name: frozen["source_hashes"][name] for name in ("kev/api.py", "kev/model.py")}},
        "game_source_sha256": protocol["game"]["source_sha256"],
    }
    need(report.get("provenance") == expected_provenance, "Context report pinned provenance changed")
    versions, packages = report.get("versions"), protocol["runtime_identity"]["packages"]
    need(isinstance(versions, dict) and set(versions) == {"transformers", "tokenizers", "torch", "pydantic"}
         and all(isinstance(value, str) and value for value in versions.values())
         and all(versions[name] == packages[name] for name in versions if name in packages),
         "Context report package versions missing or changed")
    for key in ("truncated_requests", "model_loads", "forward_calls", "games"):
        need(type(report.get(key)) is int and report[key] == 0, "Context report records truncation or forbidden activity")
    counters = report.get("guard_counters")
    need(isinstance(counters, dict) and set(counters) == {"network_attempts", "weight_file_open_attempts",
         "model_load_attempts", "forward_call_attempts"}
         and all(type(v) is int and v == 0 for v in counters.values()), "Context guard counters missing or nonzero")
    need(report.get("limits") == {"complete_row_tokens": 8192}, "Context report token limit changed")
    rows = report.get("cases")
    need(isinstance(rows, list) and len(rows) == len(cases) == 113, "Incomplete context case coverage")
    for case, row in zip(cases, rows):
        need(row.get("case_id") == case["case_id"] and row.get("body_length") == case["body_length"]
             and row.get("legacy_tail_food_duplicate") is case["legacy_tail_food_duplicate"]
             and row.get("request_sha256") == canonical_sha(case["request"]), "Context request coverage changed")
        need(all(integer(row.get(key), 1) and row[key] <= 8192
                 for key in ("state_tokens", "branch_tokens", "complete_row_tokens")), "Invalid context token counts")
        state, complete = row["state_tokens"], row["complete_row_tokens"]
        need(complete == state + row["branch_tokens"] and row.get("state_truncated") is False
             and type(row.get("decision_position")) is int and row["decision_position"] == complete - 1,
             "Context row accounting/truncation changed")
        positions = row.get("option_positions")
        need(isinstance(positions, list) and len(positions) == 4
             and all(integer(v) and state < v < complete - 1 for v in positions)
             and positions == sorted(set(positions)) and positions[-1] == complete - 2,
             "Context option positions changed")
        need(type(row.get("would_require_internal_passes")) is int
             and row["would_require_internal_passes"] == (state + 1023) // 1024 + 1,
             "Context projected pass count changed")
        record = native_record(case["request"])
        need(hexhash(row.get("native_encoding_sha256"))
             and row.get("native_record_sha256") == canonical_sha(record)
             and row.get("rendered_state_utf8_bytes") == len(record["state"].encode("utf-8")),
             "Context native record/hash/size mismatch")
    for maximum, field in (("maximum_complete_row_tokens", "complete_row_tokens"),
                           ("maximum_state_tokens", "state_tokens"),
                           ("maximum_projected_internal_passes", "would_require_internal_passes")):
        need(type(report.get(maximum)) is int and report[maximum] == max(row[field] for row in rows),
             "Context maximum disagrees with recorded cases")


def verify_supervisor(root, engine):
    """Real completion also requires the independent supervisor/one-use receipt."""
    approval_raw, claim_raw = (root / "approval.json").read_bytes(), (root / "claim.json").read_bytes()
    approval, claim = strict_json(approval_raw), strict_json(claim_raw)
    need(approval.get("schema_version") == 1 and approval.get("campaign_id") == "kev4b_campaign_v1"
         and approval.get("execution_authorized") is True
         and approval.get("freeze_sha256") == engine.frozen["freeze_sha256"]
         and approval.get("limits") == engine.frozen["protocol"]["limits"],
         "Missing exact-freeze external approval")
    need(isinstance(approval.get("one_use_id"), str) and 8 <= len(approval["one_use_id"]) <= 100,
         "Invalid one-use approval identity")
    context_raw = (root / "context-report.json").read_bytes()
    need(approval.get("context_report_sha256") == digest(context_raw),
         "Approval context report hash mismatch")
    verify_context_report(strict_json(context_raw), engine.frozen)
    evidence = {"approval_sha256": digest(approval_raw), "one_use_id": approval["one_use_id"],
                "freeze_sha256": engine.frozen["freeze_sha256"], "limits": approval["limits"],
                "context_report_sha256": digest(context_raw)}
    need(claim.get("schema_version") == 1 and claim.get("approval") == evidence,
         "Claim approval hash/identity mismatch")
    need(engine.approval == {**evidence, "claim_sha256": digest(claim_raw)},
         "Runner approval/claim evidence mismatch")
    started = exited = False
    initial_sample = final_sample = False
    deadline = claim["deadline"]
    for _, _, row in journal(root / "supervisor.jsonl", "supervisor"):
        need(row.get("synthetic") is False and not exited, "Invalid supervisor label/lifecycle")
        event = row["event"]
        if event == "supervisor_start":
            need(not started and row["approval"] == evidence and row["claim_sha256"] == digest(claim_raw)
                 and row["freeze_sha256"] == engine.frozen["freeze_sha256"]
                 and row["limits"] == approval["limits"] and row["deadline"] == deadline,
                 "Supervisor claim/freeze/limits mismatch")
            need(finite(deadline) and 0 < deadline - row["monotonic"] <= 43200
                 and row["monotonic"] <= engine.start_time, "Supervisor deadline/start ordering mismatch")
            started = True
        elif event == "resource_sample":
            need(started and not final_sample and row["monotonic"] <= deadline,
                 "Resource sample outside active supervisor lifecycle/deadline")
            phase, worker_state = row.get("sample_phase"), row.get("worker_state")
            rss, storage = row.get("current_rss_bytes"), row.get("storage")
            need(isinstance(storage, dict)
                 and integer(storage.get("free_bytes")) and storage["free_bytes"] >= 8589934592
                 and integer(storage.get("root_bytes")) and storage["root_bytes"] <= 60129542144
                 and integer(storage.get("evidence_bytes")) and storage["evidence_bytes"] <= 17179869184,
                 "Missing/invalid supervisor storage measurements or storage budget breach")
            if phase in ("initial", "periodic"):
                need(worker_state == "running" and integer(rss, 1) and rss <= 51539607552,
                     "Live supervisor sample requires positive measured RSS within its limit")
                if phase == "initial":
                    need(not initial_sample and engine.first_native_load_time is not None
                         and row["monotonic"] <= engine.first_native_load_time,
                         "Missing, repeated or late initial resource sample")
                    initial_sample = True
                else:
                    need(initial_sample, "Periodic resource sample precedes initial measurements")
            elif phase == "final":
                need(initial_sample and worker_state == "exited" and rss is None
                     and engine.end_time <= row["monotonic"],
                     "Final storage sample must follow successful runner completion and process exit")
                final_sample = True
            else:
                raise EvidenceError("Missing/unknown supervisor resource sample phase")
        elif event == "worker_exit":
            need(started and initial_sample and final_sample and row["returncode"] == 0 and engine.finished
                 and engine.end_time <= row["monotonic"] <= deadline,
                 "Worker exit missing mandatory initial/final resource evidence or successful completion")
            exited = True
        else:
            raise EvidenceError("Supervisor technical failure or unexpected event")
    if not started or not exited:
        raise IncompleteEvidence("Missing successful supervisor lifecycle")


CONTEXT = ("call_id", "game_id", "phase", "game_attempt", "warmup_index", "fixture_id")


class CampaignAudit:
    """Streaming cross-journal state machine; keeps at most one live decision."""

    def __init__(self, frozen, allow_synthetic=False):
        self.frozen, self.allow_synthetic = frozen, allow_synthetic
        self.synthetic = None
        self.started = self.finished = self.technical = False
        self.game = self.board = self.call = None
        self.completed, self.calls, self.warmups, self.game_calls, self.forwards = [], 0, 0, 0, 0
        self.native_calls = self.internal_passes = self.failed_calls = 0
        self.load_started = self.loaded = self.unloaded = False
        self.runner_loaded = self.runner_unloaded = False
        self.before_inventory = self.loaded_inventory = None
        self.variant_inventories = {}
        self.runtime_seen = self.identities_seen = False
        self.identity_checks = 0
        self.last_identity = None
        self.native_load_seen = self.checkpoint_seen = self.resolver_seen = self.quantize_seen = False
        self.native_loaded_record = self.native_unloaded_record = None
        self.start_time = self.end_time = None
        self.first_native_load_time = None
        self.approval = None
        self.last_source = self.last_sequence = None

    def same_game(self, row):
        need(self.game is not None and row.get("game_id") == self.game["game_id"],
             "Native/runner game identity mismatch")
        if "variant" in row:
            need(row["variant"] == self.game["variant"], "Variant identity mismatch")

    def same_call(self, row):
        need(self.call is not None, "Native/runner record without reserved call")
        for key in CONTEXT:
            need(key in row and key in self.call["reservation"] and row[key] == self.call["reservation"][key], "Decision context mismatch: " + key)
        if "decision_id" in row:
            need(row["decision_id"] == self.calls, "Native decision counter mismatch")
        self.same_game(row)

    def memory(self, row, source):
        required_events = {"runtime", "runtime_identity", "load_start", "native_inventory",
                           "quantized_inventory", "load_complete", "raw_decision", "unloaded"}
        mandatory = self.synthetic is False and source == "native" and row["event"] in required_events
        memory = row.get("memory", {})
        need(isinstance(memory, dict), "Memory evidence must be an object")
        required_fields = ("rss_peak_bytes", "mlx_active_bytes", "mlx_peak_bytes", "mlx_cache_bytes")
        if mandatory:
            need(all(key in memory for key in required_fields),
                 "Real native event is missing mandatory memory measurements")
            need(integer(memory["rss_peak_bytes"], 1), "Native RSS peak must be a positive integer")
        for key in (*required_fields, "rss_bytes", "current_rss_bytes"):
            if key in memory:
                need(integer(memory[key]), "Memory measurements must be nonnegative integers")
        for key in ("rss_peak_bytes", "rss_bytes", "current_rss_bytes", "mlx_active_bytes"):
            if key in memory:
                need(memory[key] <= 51539607552, "Recorded memory exceeded frozen limit")
        if "mlx_active_bytes" in memory and "mlx_peak_bytes" in memory:
            need(memory["mlx_peak_bytes"] >= memory["mlx_active_bytes"],
                 "MLX peak cannot be less than current active memory")

    def feed(self, source, row):
        self.last_source, self.last_sequence = source, row["sequence"]
        need(not self.finished, "Evidence after terminal run_complete")
        if self.synthetic is not None:
            need(row.get("synthetic") is self.synthetic, "Every journal row must carry consistent synthetic label")
        self.memory(row, source)
        if self.start_time is not None:
            need(0 <= row["monotonic"] - self.start_time <= 43200,
                 "Campaign wall duration exceeded frozen ceiling")
        if source == "runner":
            self.runner(row)
        else:
            self.native(row)

    def runner(self, row):
        event = row["event"]
        if event == "run_start":
            need(not self.started and self.calls == 0, "Repeated campaign start")
            need(type(row.get("synthetic")) is bool, "Explicit real/synthetic label required")
            self.synthetic = row["synthetic"]
            need(not self.synthetic or self.allow_synthetic,
                 "Synthetic evidence requires explicit unit-test-only allow_synthetic")
            need(row.get("freeze_sha256") == self.frozen["freeze_sha256"], "Run bound to another freeze")
            need(row.get("input_hashes") == self.frozen["freeze"]["files"], "Executed source/config inventory differs from freeze")
            need(row.get("limits") == self.frozen["protocol"]["limits"], "Runtime limits differ from frozen protocol")
            need(row["campaign_id"] == "kev4b_campaign_v1" and row["planned_games"] == 90
                 and row["retries"] == 0 and row["resume"] is False, "Campaign identity/schedule/retry policy mismatch")
            self.approval = row["approval"]
            self.started, self.start_time = True, row["monotonic"]
            return
        need(self.started, "Runner event outside active campaign")
        if self.technical:
            need(event in ("run_error", "cleanup_complete", "cleanup_error", "worker_error"),
                 "Runner continued execution after technical failure")
        if event == "inputs_verified":
            need(self.identities_seen and row["metadata"] == self.last_identity,
                 "Runner/native input identity evidence mismatch")
            return
        if event in ("cleanup_complete", "cleanup_error", "worker_error"):
            self.technical = True
            return
        if event == "game_start":
            need(self.game is None and self.call is None and len(self.completed) < 90,
                 "Overlapping or excess game")
            scheduled = self.frozen["schedule"][len(self.completed)]
            need(all(row.get(key) == value for key, value in scheduled.items() if key != "sequence"),
                 "Game differs from frozen schedule")
            need(row["game_sequence"] == scheduled["sequence"], "Scheduled game order differs")
            self.game = dict(scheduled)
            self.board = ReferenceBoard(scheduled["seed"])
            self.game.update(attempts=0, warmups=0)
            exact(row["initial_snapshot"], self.board.snapshot(), "Initial board drift")
            exact(row["food_events"], self.board.food_events, "Initial food permutation/event drift")
            self.load_started = self.loaded = self.unloaded = False
            self.runner_loaded = self.runner_unloaded = False
            self.before_inventory = self.loaded_inventory = None
            self.native_load_seen = self.checkpoint_seen = self.resolver_seen = self.quantize_seen = False
            self.native_loaded_record = self.native_unloaded_record = None
        elif event == "load_start":
            self.same_game(row)
            need(not self.load_started and not self.loaded, "Overlapping/repeated native load")
            self.load_started = True
        elif event == "load_complete":
            self.same_game(row)
            need(self.loaded and not self.runner_loaded, "Runner load without verified native load")
            need(row["metadata"] == self.native_loaded_record, "Runner/native loaded metadata differs")
            self.runner_loaded = True
        elif event == "call_reserved":
            self.reserve(row)
        elif event == "attempt":
            self.attempt(row)
        elif event == "transition":
            self.transition(row)
        elif event == "unload_start":
            self.same_game(row)
            need(self.call is None and self.runner_loaded and not self.runner_unloaded,
                 "Unload overlaps a decision or precedes loading")
        elif event == "unload_complete":
            self.same_game(row)
            need(self.unloaded and not self.runner_unloaded, "Runner unload lacks native identity check")
            need(row["metadata"] == self.native_unloaded_record, "Runner/native unload metadata differs")
            self.runner_unloaded = True
        elif event == "game_complete":
            self.same_game(row)
            need(self.call is None and self.runner_unloaded and self.game["warmups"] == 2,
                 "Game completed without calls closed, two warmups and verified unload")
            need(not self.board.alive or self.board.steps == 500, "Unfinished game reported complete")
            expected = {"score": self.board.score, "steps": self.board.steps,
                        "attempts": self.game["attempts"], "alive": self.board.alive,
                        "end_reason": "cap" if self.board.alive else "collision",
                        "censored": self.board.alive}
            for key, value in expected.items():
                exact(row.get(key), value, "Game summary differs from reconstructed " + key)
            need(self.game["attempts"] == self.board.steps + (not self.board.alive),
                 "Collision attempt or successful move accounting drift")
            exact(row["food_events"], self.board.food_events, "Final food events drift")
            exact(row["final_snapshot"], self.board.snapshot(), "Final snapshot drift")
            self.completed.append({key: self.game[key] for key in ("game_id", "seed_round", "seed", "variant")} | expected)
            self.game = self.board = None
        elif event == "run_complete":
            need(self.game is None and self.call is None and len(self.completed) == 90
                 and self.warmups == 180 and self.calls == self.native_calls == self.forwards
                 and not self.failed_calls, "Complete status without all 90 verified games/calls")
            need(self.runtime_seen and self.identity_checks == 91, "Missing runtime or per-load input provenance")
            need(row["games_completed"] == 90 and row["decisions_reserved"] == self.calls
                 and row["warmups"] == 180, "Run completion counts differ")
            self.finished, self.end_time = True, row["monotonic"]
        elif event in ("run_error", "run_incomplete"):
            # Retain all accounting; do not synthesize a collision/score or retry.
            self.technical, self.end_time = True, row["monotonic"]
        else:
            raise EvidenceError("Unknown runner event: " + event)

    def reserve(self, row):
        self.same_game(row)
        need(self.runner_loaded and self.call is None and not self.unloaded,
             "Unloaded/overlapping/retried decision")
        need(row.get("call_id") == self.calls + 1 and self.calls < MAX_CALLS,
             "Global decision reservation sequence or bound violated")
        self.calls += 1
        request = body(row, "request")
        phase = row.get("phase")
        if phase == "warmup":
            index = self.game["warmups"] + 1
            need(index <= 2 and self.game["attempts"] == 0 and row.get("warmup_index") == index
                 and row.get("game_attempt") is None, "Warmup order/count/context drift")
            case = self.frozen["protocol"]["warmups"]["case_ids"][index - 1]
            need(row.get("fixture_id") == case, "Wrong saved warmup case")
            expected = self.frozen["warmups"][case]
            self.warmups += 1
        else:
            need(phase == "game" and self.game["warmups"] == 2 and self.board.alive
                 and self.board.steps < 500 and row.get("warmup_index") is None
                 and row.get("game_attempt") == self.game["attempts"] + 1,
                 "Gameplay attempt order/context or move cap violated")
            expected = wire(self.board.request())
            self.game_calls += 1
            self.game["attempts"] += 1
        need(request == expected, "Full request bytes, field order, factual features or policy drift")
        need(row.get("fixture_id") is None or phase == "warmup", "Gameplay carried warmup identity")
        need(self.game_calls <= MAX_GAME_CALLS and self.warmups <= MAX_WARMUPS,
             "Gameplay/warmup reservation budget exceeded")
        self.call = {"reservation": row, "request": request, "native_started": False,
                     "forward_started": False, "encoded": None, "raw": None,
                     "terminal": None, "attempt": None, "passes": [], "open_pass": None}

    def attempt(self, row):
        self.same_call(row)
        need(self.call["attempt"] is None, "Duplicate adapter attempt or retry")
        record = row["attempt"]
        need(record.get("endpoint_path") == "/v1/systemone", "Adapter endpoint drift")
        need(body(record, "request") == self.call["request"], "Adapter/native request bytes differ")
        response = body(record, "response", optional=True)
        terminal = self.call["terminal"]
        if record.get("error") is not None:
            need(record.get("choice") is None, "Technical failure produced an action")
            self.failed_calls += 1
            if terminal is not None and terminal["event"] == "native_response":
                need(response == body(terminal, "response"), "Failed adapter response differs from native bytes")
            elif terminal is not None:
                need(terminal["event"] == "decision_error", "Unknown native failed terminal")
            self.call["attempt"] = row
            return
        need(terminal is not None and terminal["event"] == "native_response"
             and row.get("native_response_sequence") == terminal["sequence"],
             "Adapter success lacks the matching durable native response")
        need(record.get("response_status") == terminal["status"] == 200
             and response == body(terminal, "response"), "Native/adapter raw response mismatch")
        choice = verify_raw(self.call["raw"], strict_json(response), self.call["encoded"]["encoding"])
        need(record.get("choice") == choice == terminal.get("choice"), "Adapter substituted the native action")
        self.call["choice"], self.call["attempt"] = choice, row
        if row["phase"] == "warmup":
            self.game["warmups"] += 1
            self.call = None

    def transition(self, row):
        self.same_call(row)
        need(row.get("phase") == "game" and self.call["attempt"] is not None
             and self.call["attempt"]["attempt"].get("error") is None,
             "Transition precedes durable successful attempt or follows technical error")
        exact(row["before"], self.board.snapshot(), "Pre-transition board mismatch")
        need(row.get("move") == self.call["choice"], "Applied move differs from raw native argmax")
        previous_events = len(self.board.food_events)
        self.board.apply(row["move"])
        exact(row["after"], self.board.snapshot(), "Board/tail/collision/score transition drift")
        exact(row["new_food_events"], self.board.food_events[previous_events:], "New food event drift")
        exact(row["food_events"], self.board.food_events, "Cumulative food event drift")
        self.call = None

    def native(self, row):
        event = row["event"]
        need(self.started, "Native work predates runner reservation/start")
        if event in ("runtime", "runtime_identity"):
            need(not self.runtime_seen and self.calls == 0, "Repeated/late native runtime identity")
            expected = self.frozen["protocol"]["runtime_identity"]
            need(row["versions"] == expected["packages"] and row["python"].split()[0] == expected["python"]
                 and row["network_allowed"] is False and "arm64" in row["platform"],
                 "Runtime versions/Python/platform/network identity changed")
            self.runtime_seen = True
            return
        if event in ("inputs_verified", "source_verification"):
            need(self.call is None and (self.game is None and not self.completed or self.load_started and not self.loaded),
                 "Input verification during native evaluation or outside load")
            verify_inputs_event(row, self.frozen, synthetic=self.synthetic)
            self.identity_checks += 1
            need(self.identity_checks <= 91, "Unexpected repeated native input verification")
            self.last_identity = {k: v for k, v in row.items() if k not in ("event", "sequence", "monotonic", "unix_time", "synthetic")}
            self.identities_seen = True
            return
        if event in ("verified_file", "native_start"):
            need(self.calls == 0, "Late native source verification")
            return
        if event == "load_start":
            self.same_game(row)
            need(self.load_started and not self.loaded and self.before_inventory is None and not self.native_load_seen
                 and self.identity_checks == len(self.completed) + 2,
                 "Native load outside runner scope")
            self.native_load_seen = True
            if self.first_native_load_time is None:
                self.first_native_load_time = row["monotonic"]
        elif event == "checkpoint_metadata":
            self.same_game(row)
            need(self.native_load_seen and not self.checkpoint_seen and row["exact_native_temperature"] == HEAD_TEMPERATURE
                 and row["head_file_sha256"] == HEAD_FILE_SHA256, "Head file/calibration differs from preflight")
            self.checkpoint_seen = True
        elif event == "local_snapshot_resolution":
            self.same_game(row)
            need(self.checkpoint_seen and not self.resolver_seen, "Missing/repeated local resolver")
            verify_resolver(row["resolver"], self.frozen)
            self.resolver_seen = True
        elif event == "native_inventory":
            self.same_game(row)
            need(self.before_inventory is None and not self.loaded and self.resolver_seen, "Repeated fresh-native inventory or missing local resolver")
            before = row["inventory"]
            verify_inventory(before, before, "bf16", synthetic=self.synthetic)
            need(row["inventory_sha256"] == canonical_sha(before), "Native baseline inventory hash mismatch")
            self.before_inventory = before
        elif event == "quantize_start":
            self.same_game(row)
            bits = {VARIANTS[1]: 8, VARIANTS[2]: 4}.get(self.game["variant"])
            need(self.before_inventory is not None and bits is not None and not self.quantize_seen
                 and row["bits"] == bits and row["group_size"] == 64
                 and row["mode"] == "affine" and row["quantize_input"] is False,
                 "Quantization scope/recipe or fresh load mismatch")
            self.quantize_seen = True
        elif event == "quantized_inventory":
            self.same_game(row)
            need(self.before_inventory is not None and not self.loaded and self.quantize_seen and self.loaded_inventory is None,
                 "Quantization lacks native baseline or repeats")
            verify_inventory(self.before_inventory, row["inventory"], self.game["variant"], synthetic=self.synthetic)
            need(row["derivation"] == "fresh_native_merge_then_in_memory_quantization"
                 and row["persisted_variant"] is False and row["export_reload_validated"] is False,
                 "Unapproved quantization derivation/reload")
            need(row["inventory_sha256"] == canonical_sha(row["inventory"]), "Quantized inventory hash mismatch")
            self.loaded_inventory = row["inventory"]
        elif event == "load_complete":
            self.same_game(row)
            need(not self.loaded and self.before_inventory is not None, "Missing/repeated native baseline")
            if self.game["variant"] == "bf16":
                self.loaded_inventory = self.before_inventory
            need(self.loaded_inventory is not None and row["inventory_sha256"] == canonical_sha(self.loaded_inventory)
                 and row["head"] == self.loaded_inventory["head"] and row["retained_backbone_tensors"] == 178,
                 "Native load completion inventory mismatch")
            previous = self.variant_inventories.setdefault(self.game["variant"], row["inventory_sha256"])
            need(previous == row["inventory_sha256"], "Condition inventory differs between fresh loads")
            if not self.synthetic:
                need(row["inventory_sha256"] == self.frozen["protocol"]["runtime"]["expected_condition_inventory_sha256"][self.game["variant"]],
                     "Condition inventory differs from frozen preflight")
            need(row["export_reload_validated"] is False, "Unapproved export/reload claim")
            self.native_loaded_record = {k: v for k, v in row.items() if k not in ("event", "sequence", "monotonic", "unix_time", "synthetic")}
            self.loaded = True
        elif event == "unloaded":
            self.same_game(row)
            need(self.loaded and not self.unloaded and (self.call is None or self.technical),
                 "Native unload overlaps pending decision")
            need(row["inventory"] == self.loaded_inventory
                 and row["inventory_sha256"] == canonical_sha(row["inventory"])
                 and row["head_after"] == self.loaded_inventory["head"] and row["unchanged"] is True,
                 "Tensor/head mutation during game or missing final rehash")
            self.native_unloaded_record = {k: v for k, v in row.items() if k not in ("event", "sequence", "monotonic", "unix_time", "synthetic")}
            self.unloaded = True
        elif event in ("load_error", "unload_error", "native_error", "fatal"):
            self.technical = True
        else:
            self.native_decision(row)

    def native_decision(self, row):
        event = row["event"]
        self.same_call(row)
        need(self.loaded and not self.unloaded, "Native evaluation outside loaded lifecycle")
        if event == "decision_start":
            need(not self.call["native_started"] and self.call["terminal"] is None
                 and row["call_id"] == self.native_calls + 1,
                 "Native forward unreserved, duplicated or reordered")
            need(body(row, "request") == self.call["request"], "Native request differs from reserved bytes")
            self.native_calls += 1
            self.call["native_started"] = True
        elif event == "encoded":
            need(self.call["native_started"] and self.call["encoded"] is None
                 and not self.call["forward_started"], "Repeated/late encoding or missing reservation")
            verify_encoding(row, strict_json(self.call["request"]))
            self.call["encoded"] = row
        elif event == "native_forward_start":
            need(self.call["encoded"] is not None and not self.call["forward_started"],
                 "Forward without complete encoding or hidden retry")
            s = self.call["encoded"]["encoding"]["state_tokens"]
            need(row["expected_backbone_passes"] == (s + 1023) // 1024 + 1, "Reserved native internal pass count changed")
            self.forwards += 1
            self.call["forward_started"] = True
        elif event == "backbone_pass_start":
            need(self.call["forward_started"] and self.call["raw"] is None
                 and self.call["open_pass"] is None and row["pass_number"] == len(self.call["passes"]) + 1,
                 "Internal forward missing reservation, overlapping or repeated")
            enc = self.call["encoded"]["encoding"]
            s, n, j = enc["state_tokens"], len(enc["ids"]), row["pass_number"]
            prefixes = (s + 1023) // 1024
            expected = min(1024, s - (j - 1) * 1024) if j <= prefixes else n - s
            need(j <= prefixes + 1 and row["input_shape"] == [1, expected]
                 and row["cache_supplied"] is True
                 and row["native_stage"] == ("state_prefix" if j <= prefixes else "question_branch"),
                 "Native prefix/branch count, shape or order mismatch")
            self.call["open_pass"] = row
            self.internal_passes += 1
        elif event in ("backbone_pass_result", "backbone_pass_error"):
            pending = self.call["open_pass"]
            need(pending is not None and row["pass_number"] == pending["pass_number"],
                 "Unreserved or repeated internal pass completion")
            need(all(row.get(k) == pending.get(k) for k in ("input_shape", "native_stage", "cache_supplied")),
                 "Native pass result does not match reserved pass")
            if event == "backbone_pass_result":
                need(row.get("outcome") == "returned", "Native pass missing result")
            self.call["passes"].append(pending)
            self.call["open_pass"] = None
            if event.endswith("error"):
                self.call["pass_error"] = True
        elif event == "raw_decision":
            need(self.call["forward_started"] and self.call["raw"] is None
                 and self.call["open_pass"] is None and not self.call.get("pass_error"),
                 "Raw output lacks completed forward or follows failure")
            s = self.call["encoded"]["encoding"]["state_tokens"]
            need(len(self.call["passes"]) == (s + 1023) // 1024 + 1,
                 "Incomplete/extra native internal forwards")
            need(len(row["internal_backbone_passes"]) == len(self.call["passes"]),
                 "Raw internal pass count mismatch")
            for reported, actual in zip(row["internal_backbone_passes"], self.call["passes"]):
                for key in ("input_shape", "native_stage", "cache_supplied"):
                    need(reported.get(key) == actual.get(key), "Raw/native pass evidence mismatch")
            self.call["raw"] = row
        elif event in ("native_response", "decision_error"):
            need(self.call["native_started"] and self.call["terminal"] is None
                 and self.call["open_pass"] is None, "Unreserved/repeated terminal or unclosed internal forward")
            need(row["monotonic"] - self.call["reservation"]["monotonic"] <= 120,
                 "Decision exceeded frozen wall ceiling")
            if event == "native_response":
                need(self.call["raw"] is not None and row["status"] == 200,
                     "Native response missing raw forward result")
                verify_raw(self.call["raw"], strict_json(body(row, "response")),
                           self.call["encoded"]["encoding"])
            else:
                body(row, "response", optional=True)
            self.call["terminal"] = row
        else:
            raise EvidenceError("Unknown native event: " + event)

    def result(self):
        complete = self.finished and not self.technical
        status = "synthetic_complete" if complete and self.synthetic else "complete" if complete else "technical_failure" if self.technical else "incomplete"
        result = {"schema_version": 1, "campaign_id": "kev4b_campaign_v1",
                  "passed": bool(complete and self.synthetic is False), "evidence_valid": True,
                  "complete": complete, "real": self.synthetic is False,
                  "status": status, "freeze_sha256": self.frozen["freeze_sha256"],
                  "counts": {"completed_games": len(self.completed), "reserved_decisions": self.calls,
                             "native_reserved_decisions": self.native_calls, "native_forwards": self.forwards,
                             "internal_backbone_passes": self.internal_passes, "warmup_reservations": self.warmups,
                             "gameplay_reservations": self.game_calls, "failed_attempts": self.failed_calls},
                  "issues": [], "limitations": [
                      "Evidence integrity and reconstruction do not independently prove that hardware produced logged values.",
                      "Token IDs are structurally checked and retained; this offline audit does not load a tokenizer or recompute tokenization.",
                      "An alive cap observes food through 500 moves; eventual uncapped score is unknown."]}
        if complete:
            result["games"] = self.completed
        else:
            result["issues"].append("Campaign incomplete or technically interrupted; no score dataset or inferential result emitted.")
        return result


def audit(root, *, repo=None, allow_synthetic=False):
    """Return a report without changing evidence. CLI never enables synthetic mode."""
    root = Path(root)
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    engine = None
    try:
        engine = CampaignAudit(read_frozen(repo), allow_synthetic=allow_synthetic)
        for _, source, row in merged_journals(root):
            engine.feed(source, row)
        if engine.finished and engine.synthetic is False:
            verify_supervisor(root, engine)
        return engine.result()
    except (EvidenceError, OSError, KeyError, TypeError, IndexError, OverflowError) as exc:
        result = engine.result() if engine is not None else {
            "schema_version": 1, "campaign_id": "kev4b_campaign_v1", "counts": {}, "real": False}
        result.update(passed=False, complete=False, evidence_valid=False,
                      status="incomplete" if isinstance(exc, IncompleteEvidence) else "technical_failure")
        result.pop("games", None)
        location = f" at {engine.last_source} sequence {engine.last_sequence}" if engine is not None and engine.last_source else ""
        result["issues"] = [(str(exc) if isinstance(exc, EvidenceError) else type(exc).__name__) + location]
        return result


audit_campaign = audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    result = audit(args.evidence, repo=args.repo)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
