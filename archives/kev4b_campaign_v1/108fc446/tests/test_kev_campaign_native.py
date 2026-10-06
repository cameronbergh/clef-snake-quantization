"""Offline native-boundary tests using Python stubs, never model libraries."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.kev4b_campaign_v1 import native as n


class Journal:
    def __init__(self):
        self.sequence, self.rows = 0, []

    def write(self, event, **fields):
        self.sequence += 1
        self.rows.append({"sequence": self.sequence, "event": event, **fields})


class Guard:
    execution_authorized = True

    def check(self, **kwargs):
        pass


class Tensor:
    def __init__(self, values):
        self.values, self.shape = values, (len(values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class Torch:
    @staticmethod
    def softmax(tensor, axis):
        values = [math.exp(v) for v in tensor.values]
        return Tensor([v / sum(values) for v in values])


class Text:
    def __init__(self):
        self.calls = []

    def __call__(self, value, cache=None):
        self.calls.append((list(value.shape), cache))
        return None


def encoding(state_tokens=303, branch_tokens=80):
    size = state_tokens + branch_tokens
    return {"ids": list(range(size)), "seg": [0] * state_tokens + [1] * branch_tokens,
            "pos": list(range(size)), "opt": [-1] * state_tokens + [0] * branch_tokens,
            "option_isolation": False, "state_tokens": state_tokens, "state_truncated": False,
            "decide_idx": [size - 1], "opt_idx": [[state_tokens + i for i in (2, 4, 6, 8)]],
            "labels": [0]}


def rows_of(enc):
    count = enc["seg"].count(0)
    return enc["ids"][:count], list(range(count)), [{
        "ids": enc["ids"][count:], "opts": [i - count for i in enc["opt_idx"][0]],
        "decide": enc["decide_idx"][0] - count}]


class Model:
    def __init__(self, enc=None):
        self.enc = encoding() if enc is None else enc
        self.text, self.forward_calls = Text(), 0
        self.logits = [0.0, 0.0, 0.0, 0.0]
        self.error = None

    def encode(self, tok, rec, **kw):
        self.encode_kwargs = kw
        return deepcopy(self.enc)

    def forward(self, enc):
        self.forward_calls += 1
        cache = object()  # Keep alive in Text.calls to compare cache identities.
        state = enc["state_tokens"]
        for start in range(0, state, n.PREFIX_CHUNK):
            self.text(SimpleNamespace(shape=(1, min(n.PREFIX_CHUNK, state - start))), cache=cache)
        self.text(SimpleNamespace(shape=(1, len(enc["ids"]) - state)), cache=cache)
        if self.error:
            raise self.error
        return [Tensor(self.logits)]


def to_answers(probabilities, meta):
    p = probabilities[0]
    return {"move": {"type": "choice", "choice": n.ACTIONS[max(range(4), key=p.__getitem__)],
                     "confidence": 0, "probabilities": {k: round(v, 4) for k, v in zip(n.ACTIONS, p)}}}


class StubRuntime(n.NativeRuntime):
    def memory(self):
        return {"rss_peak_bytes": 0}


class NativeCampaignTests(unittest.TestCase):
    def setUp(self):
        self.raw = n.wire({"model": "kev-latest", "state": {"test": "fixed factual stub"},
                           "questions": deepcopy(n.QUESTION)})
        self.journal = Journal()
        self.runtime = StubRuntime(Path.cwd(), Path.cwd(), self.journal, Guard())
        self.runtime.game_id, self.runtime.variant = "game-001", "bf16"
        self.runtime.model, self.runtime.tokenizer = Model(), object()
        self.runtime.warmup_requests = (self.raw, self.raw)
        self.runtime.torch = Torch()
        self.runtime.api = SimpleNamespace(
            SystemOneRequest=lambda **kw: kw,
            to_record=lambda req: ({"state": "unchanged", "questions": []},
                                   [{"id": "move", "type": "choice", "keys": list(n.ACTIONS)}]),
            to_answers=to_answers, output_tokens=lambda tok, answers: 32, rows_of=rows_of)

    def context(self, number=1, *, phase="warmup", index=1, attempt=None):
        return {"call_id": number, "game_id": "game-001", "phase": phase,
                "warmup_index": index, "game_attempt": attempt,
                "fixture_id": ("bf16-round-01-first" if index == 1 else "bf16-round-01-middle")
                if phase == "warmup" else None}

    def test_import_has_no_model_imports(self):
        code = "import sys; from experiments.kev4b_campaign_v1 import native; assert not any(k.split('.')[0] in {'torch','mlx','mlx_lm','transformers','kev','numpy'} for k in sys.modules)"
        subprocess.run([sys.executable, "-B", "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])

    def test_missing_separate_approval_refuses_before_asset_reads(self):
        self.runtime.guard.execution_authorized = False
        with patch.object(n, "file_hash", side_effect=AssertionError("should not read")):
            with self.assertRaisesRegex(n.NativeError, "approval"):
                self.runtime.verify_inputs()

    def test_exactly_one_forward_reservation_and_independent_evidence(self):
        original = Text.__call__
        response = self.runtime.evaluate(self.raw, self.context())
        self.assertEqual(response.status, 200)
        self.assertEqual(n.parse_response(response.body), "up")
        self.assertEqual(self.runtime.calls, 1)
        self.assertEqual(self.runtime.model.forward_calls, 1)
        self.assertIs(Text.__call__, original)
        self.assertEqual(self.journal.rows[0]["event"], "decision_start")
        events = [row["event"] for row in self.journal.rows]
        self.assertEqual(events.count("native_forward_start"), 1)
        self.assertEqual(events.count("backbone_pass_start"), 2)
        self.assertEqual(events.count("backbone_pass_result"), 2)
        self.assertEqual(events[-1], "native_response")
        self.assertEqual(self.runtime.last_response_sequence, self.journal.sequence)
        self.assertEqual(self.runtime.model.encode_kwargs,
                         {"max_state": 8192, "max_branch": 8192, "strict": True})
        row = next(x for x in self.journal.rows if x["event"] == "raw_decision")
        self.assertEqual(row["selected_index"], 0)
        self.assertEqual(row["probabilities"], [[.25, .25, .25, .25]])

    def test_raw_argmax_preserved_across_display_rounding_tie(self):
        self.runtime.model.logits = [math.log(v) for v in (.249999, .250001, .25, .25)]
        response = self.runtime.evaluate(self.raw, self.context())
        self.assertEqual(n.parse_response(response.body), "down")

    def test_warmups_then_game_and_fresh_cache(self):
        self.runtime.evaluate(self.raw, self.context())
        self.runtime.evaluate(self.raw, self.context(2, index=2))
        self.runtime.evaluate(self.raw, self.context(3, phase="game", index=None, attempt=1))
        caches = [cache for shape, cache in self.runtime.model.text.calls]
        self.assertIs(caches[0], caches[1])
        self.assertIsNot(caches[0], caches[2])
        self.assertIsNot(caches[2], caches[4])
        self.assertEqual(self.runtime.game_attempts, 1)

    def test_long_state_has_separate_prefix_passes_and_full_ids(self):
        self.runtime.model = Model(encoding(8112, 80))
        self.runtime.evaluate(self.raw, self.context())
        events = self.journal.rows
        enc = next(x["encoding"] for x in events if x["event"] == "encoded")
        self.assertEqual(enc["ids"], list(range(8192)))
        passes = [x for x in events if x["event"] == "backbone_pass_start"]
        self.assertEqual(len(passes), 9)
        self.assertEqual([x["input_shape"][1] for x in passes], [1024] * 7 + [944, 80])

    def test_overlong_or_truncated_encoding_stops_before_forward_but_counts(self):
        for enc in (encoding(8113, 80), {**encoding(), "state_truncated": True},
                    {**encoding(), "state_tokens": 304}):
            with self.subTest(length=len(enc["ids"])):
                self.setUp()
                self.runtime.model = Model(enc)
                with self.assertRaises(n.NativeError):
                    self.runtime.evaluate(self.raw, self.context())
                self.assertEqual(self.runtime.calls, 1)
                self.assertEqual(self.runtime.model.forward_calls, 0)
                self.assertEqual(self.journal.rows[-1]["event"], "decision_error")

    def test_error_restores_instrumentation_and_forbids_retry(self):
        original = Text.__call__
        self.runtime.model.error = RuntimeError("stub forward failed")
        with self.assertRaisesRegex(RuntimeError, "stub forward failed"):
            self.runtime.evaluate(self.raw, self.context())
        self.assertIs(Text.__call__, original)
        self.assertTrue(self.runtime.failed)
        self.assertEqual(self.journal.rows[-1]["event"], "decision_error")
        with self.assertRaises(n.NativeError):
            self.runtime.evaluate(self.raw, self.context(2, index=2))
        self.assertEqual(self.runtime.model.forward_calls, 1)

    def test_nonfinite_output_recorded_before_refusal(self):
        self.runtime.model.logits[0] = float("nan")
        with self.assertRaises(n.NativeError):
            self.runtime.evaluate(self.raw, self.context())
        raw = next(x for x in self.journal.rows if x["event"] == "raw_decision")
        self.assertEqual(raw["logits"][0][0], "nan")
        self.assertIsNone(raw["selected_index"])
        self.assertEqual(self.journal.rows[-1]["event"], "decision_error")

    def test_context_and_budget_no_extra_call(self):
        for context in (self.context(2), self.context(True), self.context(1, index=2),
                        self.context(1, phase="game", index=None, attempt=1)):
            with self.subTest(context=context), self.assertRaises(n.NativeError):
                self.runtime.evaluate(self.raw, context)
        self.assertEqual(self.runtime.model.forward_calls, 0)
        self.runtime.calls = n.MAX_DECISIONS
        with self.assertRaises(n.NativeError):
            self.runtime.evaluate(self.raw, self.context(n.MAX_DECISIONS + 1))

    def test_request_question_order_duplicate_keys_and_nonfinite_refused(self):
        import json
        req = json.loads(self.raw)
        bad = deepcopy(req)
        bad["questions"]["move"]["criteria"] = dict(reversed(list(bad["questions"]["move"]["criteria"].items())))
        for value in (n.wire(bad), self.raw.replace(b'"model":', b'"model":"duplicate","model":', 1),
                      self.raw + b" ", self.raw.replace(b'"fixed factual stub"', b"NaN")):
            with self.subTest(value=value[:20]), self.assertRaises(n.NativeError):
                n.parse_request(value)

    def test_inventory_hash_canonical_key_order(self):
        self.assertEqual(n.inventory_hash({"a": 1, "b": 2}), n.inventory_hash({"b": 2, "a": 1}))

    def test_unload_rechecks_inventory_and_clears_loaded_model(self):
        value = {"head": {"temperature": n.TEMPERATURE}, "tensors": {"retained": "unchanged"}}
        self.runtime.loaded_inventory = deepcopy(value)
        self.runtime.inventory = lambda model: deepcopy(value)
        cleared = []
        self.runtime.mx = SimpleNamespace(get_active_memory=lambda: 0, get_peak_memory=lambda: 0,
                                          clear_cache=lambda: cleared.append(True))
        self.runtime.unload()
        self.assertIsNone(self.runtime.model)
        self.assertIsNone(self.runtime.tokenizer)
        self.assertEqual(cleared, [True])
        self.assertEqual(self.journal.rows[-1]["event"], "unloaded")
        self.assertTrue(self.journal.rows[-1]["unchanged"])

    def test_unload_mutation_is_terminal_failure(self):
        self.runtime.loaded_inventory = {"head": {"temperature": n.TEMPERATURE}}
        self.runtime.inventory = lambda model: {"head": {"temperature": 2.41}}
        with self.assertRaisesRegex(n.NativeError, "changed"):
            self.runtime.unload()
        self.assertTrue(self.runtime.failed)
        self.assertEqual(self.journal.rows[-1]["event"], "unload_error")

    def test_quantization_preserves_all_retained_tensors_and_head(self):
        names = [f"layers.{i}.projection" for i in range(248)]
        before = {"linears": {name: {} for name in names}, "quantized": {},
                  "tensors": {**{name + ".weight": {"value": "bf16"} for name in names},
                              **{f"retained.{i}": {"value": i} for i in range(178)}},
                  "head": {"temperature": n.TEMPERATURE, "sha256": n.HEAD_INVENTORY_SHA256}}
        after = deepcopy(before)
        after["linears"] = {}
        after["quantized"] = {name: {"bits": 8, "group_size": 64, "mode": "affine", "class": "QuantizedLinear"} for name in names}
        for name in names:
            for suffix in (".scales", ".biases"):
                after["tensors"][name + suffix] = {"value": "packed"}
        with patch.dict(n.EXPECTED_INVENTORIES, {"mlx-affine8-g64": n.inventory_hash(after)}):
            n.validate_quantization(before, after, "mlx-affine8-g64")
            broken = deepcopy(after);broken["tensors"]["retained.0"]["value"] = 999
            with self.assertRaisesRegex(n.NativeError, "Retained"):
                n.validate_quantization(before, broken, "mlx-affine8-g64")
            broken = deepcopy(after);broken["head"]["temperature"] = 2.41
            with self.assertRaisesRegex(n.NativeError, "head"):
                n.validate_quantization(before, broken, "mlx-affine8-g64")
            broken = deepcopy(after);broken["quantized"][names[0]]["bits"] = 4
            with self.assertRaisesRegex(n.NativeError, "recipe"):
                n.validate_quantization(before, broken, "mlx-affine8-g64")
        with self.assertRaisesRegex(n.NativeError, "frozen"):
            n.validate_quantization(before, after, "mlx-affine8-g64")


if __name__ == "__main__":
    unittest.main()
