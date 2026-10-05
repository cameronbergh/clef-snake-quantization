"""Offline Kev API contract checks. All transport responses are synthetic."""
import base64
from copy import deepcopy
import hashlib
import json
import unittest

from clef_snake.game import Game, QUESTION
from experiments.kev4b_v1 import adapter


def response(choice="right", probabilities=None):
    return {"model": "kev-latest", "answers": {"move": {
        "type": "choice", "choice": choice,
        "probabilities": probabilities or {
            key: (0.7 if key == choice else 0.1) for key in adapter.ACTIONS},
        "confidence": 0.6,
    }}, "usage": {"input_tokens": 731, "output_tokens": 83}, "latency_ms": 12.3}


def wire(value):
    return json.dumps(value).encode("utf-8")


class KevAdapterTests(unittest.TestCase):
    def setUp(self):
        self.game = Game("b145b41017ce4bce48bf75df6719f853")

    def assertRejected(self, value):
        with self.assertRaises(adapter.ResponseError):
            adapter.parse_response(value if isinstance(value, bytes) else wire(value))

    def test_request_preserves_all_fields_and_order_except_model_alias(self):
        original = self.game.request()
        expected = deepcopy(original)
        expected["model"] = "kev-latest"
        actual = adapter.make_request(self.game)
        self.assertEqual(actual, json.dumps(expected, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8"))
        request = json.loads(actual)
        self.assertEqual(list(request), ["model", "state", "questions"])
        self.assertEqual(list(request["state"]), ["grid", "body", "direction",
                         "head", "food", "food_delta", "safe_moves", "move_analysis"])
        self.assertEqual(list(request["questions"]["move"]["criteria"]),
                         ["up", "down", "left", "right"])
        self.assertEqual(request["state"]["food"], [0, 3])
        self.assertEqual(request["state"]["body"], [[5, 6], [4, 6], [3, 6], [2, 6]])
        self.assertEqual(request["state"]["move_analysis"]["left"]["reason"], "neck")
        self.assertEqual(original["model"], "clef-flash")
        self.assertEqual(adapter.ENDPOINT_PATH, "/v1/systemone")

    def test_request_is_deep_copy_of_game_and_shared_question(self):
        original_question = deepcopy(QUESTION)
        before = self.game.snapshot()
        request = adapter.build_request(self.game)
        request["questions"]["move"]["criteria"].clear()
        request["state"]["body"][0][0] = 99
        request["state"]["move_analysis"]["left"]["safe"] = True
        self.assertEqual(QUESTION, original_question)
        self.assertEqual(self.game.snapshot(), before)
        self.assertEqual(len(adapter.build_request(self.game)["questions"]["move"]["criteria"]), 4)

    def test_unsafe_native_choice_is_retained_without_applying_move(self):
        before = self.game.snapshot()
        calls = []
        raw = wire(response("left"))
        def transport(request):
            calls.append(request)
            return adapter.TransportResponse(200, raw)
        result = adapter.attempt(self.game, transport)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(result.error)
        self.assertEqual(result.choice, "left")
        self.assertEqual(self.game.snapshot(), before)
        record = result.to_record()  # Caller persists this before applying.
        self.assertEqual(base64.b64decode(record["request_base64"]), calls[0])
        self.assertEqual(record["request_sha256"], hashlib.sha256(calls[0]).hexdigest())
        self.assertEqual(base64.b64decode(record["response_base64"]), raw)
        self.game.apply(result.choice)
        self.assertFalse(self.game.alive)
        self.assertEqual(self.game.steps, 0)

    def test_all_directions_and_nonzero_native_output_tokens_are_valid(self):
        for action in adapter.ACTIONS:
            with self.subTest(action=action):
                self.assertEqual(adapter.parse_response(wire(response(action))), action)

    def test_rounding_tie_preserves_server_winner_instead_of_first_maximum(self):
        value = response("right", {"up": 0.4, "down": 0.1, "left": 0.1, "right": 0.4})
        self.assertEqual(adapter.parse_response(wire(value)), "right")
        value["answers"]["move"]["choice"] = "left"
        self.assertRejected(value)

    def test_duplicate_keys_rejected_at_every_json_level(self):
        raw = wire(response())
        replacements = [
            (b'"model": "kev-latest"', b'"model":"kev-latest","model":"kev-latest"'),
            (b'"type": "choice"', b'"type":"choice","type":"choice"'),
            (b'"up": 0.1', b'"up":0.1,"up":0.1'),
            (b'"input_tokens": 731', b'"input_tokens":731,"input_tokens":731'),
        ]
        for old, new in replacements:
            with self.subTest(old=old):
                self.assertIn(old, raw)
                self.assertRejected(raw.replace(old, new))

    def test_probability_shape_membership_finiteness_and_sum(self):
        for invalid in (None, "0.1", True, -0.1, 1.1, float("nan"), float("inf")):
            value = response()
            value["answers"]["move"]["probabilities"]["up"] = invalid
            with self.subTest(invalid=invalid):
                self.assertRejected(value)
        for mutation in ("missing", "extra", "bad_sum", "array"):
            value = response()
            p = value["answers"]["move"]["probabilities"]
            if mutation == "missing": del p["left"]
            elif mutation == "extra": p["stay"] = 0.0
            elif mutation == "bad_sum": p["right"] = 0.6
            else: value["answers"]["move"]["probabilities"] = list(p.values())
            with self.subTest(mutation=mutation):
                self.assertRejected(value)
        self.assertEqual(adapter.parse_response(wire(response("right", {
            "up": 0.2, "down": 0.2, "left": 0.2, "right": 0.4002}))), "right")
        self.assertRejected(response("right", {
            "up": 0.2, "down": 0.2, "left": 0.2, "right": 0.4003}))

    def test_wrong_schema_alias_answer_type_choice_usage_and_truncation(self):
        for action in (None, "Right", "stay", ["up"], {"action": "up"}):
            value = response()
            value["answers"]["move"]["choice"] = action
            self.assertRejected(value)
        for path, invalid in (("model", "clef-flash"), ("answers", {}),
                              ("usage", None), ("truncated", True),
                              ("truncated", "false"), ("error", "failed")):
            value = response()
            value[path] = invalid
            self.assertRejected(value)
        value = response()
        value["answers"]["extra"] = deepcopy(value["answers"]["move"])
        self.assertRejected(value)
        value = response()
        value["answers"]["move"]["type"] = "noul"
        self.assertRejected(value)
        for field in ("input_tokens", "output_tokens"):
            for invalid in (None, -1, True, 1.5):
                value = response()
                value["usage"][field] = invalid
                self.assertRejected(value)

    def test_malformed_json_invalid_utf8_and_numeric_overflow(self):
        for raw in (b"\xff", b"{broken", b"[]", b"null", wire(response()) + b" trailing",
                    wire(response()).replace(b'"latency_ms": 12.3', b'"latency_ms":1e999')):
            with self.subTest(raw=raw[:40]):
                self.assertRejected(raw)

    def test_complete_input_budget_and_truncation_accounting(self):
        for count in (0, 8193):
            value = response()
            value["usage"]["input_tokens"] = count
            self.assertRejected(value)
        value = response()
        value["usage"].update(input_tokens=8192, state_tokens=8000, state_tokens_used=8000)
        self.assertEqual(adapter.parse_response(wire(value)), "right")
        value["usage"]["state_tokens_used"] = 7999
        self.assertRejected(value)
        del value["usage"]["state_tokens_used"]
        self.assertRejected(value)

    def test_http_errors_and_invalid_response_keep_raw_bytes_without_retry(self):
        before = self.game.snapshot()
        for status, body, stage in ((302, b"redirect", "http"), (422, b"too long", "http"),
                                    (503, b"busy", "http"), (200, b"\xffpartial", "response"),
                                    (204, b"", "response")):
            calls = []
            def transport(request):
                calls.append(request)
                return adapter.TransportResponse(status, body)
            result = adapter.attempt(self.game, transport)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(result.choice)
            self.assertEqual(result.error.stage, stage)
            self.assertEqual(result.response_status, status)
            record = json.loads(json.dumps(result.to_record()))
            self.assertEqual(base64.b64decode(record["response_base64"]), body)
            self.assertEqual(self.game.snapshot(), before)

    def test_transport_exception_retains_partial_response_when_supplied(self):
        calls = []
        partial = adapter.TransportResponse(200, b'{"answers":\xff')
        def transport(request):
            calls.append(request)
            raise adapter.TransportFailure("read interrupted", response=partial)
        result = adapter.attempt(self.game, transport)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.response_body, partial.body)
        self.assertEqual(result.response_status, 200)
        self.assertEqual(result.error.stage, "transport")
        self.assertEqual(result.error.message, "read interrupted")
        self.assertIsNone(result.choice)

    def test_plain_timeout_is_recorded_and_never_retried(self):
        calls = []
        def transport(request):
            calls.append(request)
            raise TimeoutError("timed out")
        result = adapter.attempt(self.game, transport)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.error.type, "TimeoutError")
        self.assertIsNone(result.response_body)
        self.assertIsNone(result.choice)

    def test_terminal_game_is_rejected_before_transport(self):
        self.game.apply("left")
        calls = []
        with self.assertRaises(ValueError):
            adapter.attempt(self.game, lambda request: calls.append(request))
        self.assertEqual(calls, [])

    def test_transport_contract_rejects_text_and_nonresponse_objects(self):
        with self.assertRaises(TypeError):
            adapter.TransportResponse(200, "already decoded")
        with self.assertRaises(TypeError):
            adapter.TransportResponse(True, b"{}")
        result = adapter.attempt(self.game, lambda request: wire(response()))
        self.assertEqual(result.error.stage, "transport")
        self.assertEqual(result.error.type, "TypeError")
        self.assertIsNone(result.choice)


if __name__ == "__main__":
    unittest.main()
