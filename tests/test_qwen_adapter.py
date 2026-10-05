"""Offline adversarial checks; injected transports never contact a model."""
import base64
from copy import deepcopy
import hashlib
import json
import unittest

from clef_snake.game import Game
from experiments.qwen3_snake_v1 import adapter


def response(action="right", **message_fields):
    message = {"role": "assistant", "content": json.dumps({"action": action})}
    message.update(message_fields)
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 317, "completion_tokens": 6, "total_tokens": 323}}


class QwenAdapterTests(unittest.TestCase):
    def setUp(self):
        self.game = Game("b145b41017ce4bce48bf75df6719f853")

    def assertRejected(self, value):
        with self.assertRaises(adapter.ResponseError):
            adapter.parse_response(value if isinstance(value, (str, bytes)) else json.dumps(value))

    def test_ordered_factual_payload_and_original_question(self):
        before = self.game.snapshot()
        original = self.game.request()
        request = adapter.build_request(self.game)
        payload = json.loads(request["messages"][1]["content"])
        self.assertEqual(list(payload), ["state", "questions"])
        self.assertEqual(list(payload["state"]), list(original["state"]))
        self.assertEqual(payload, {"state": original["state"], "questions": original["questions"]})
        self.assertEqual(request["messages"][1]["content"], json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        self.assertEqual(list(payload["questions"]["move"]["criteria"]),
                         ["up", "down", "left", "right"])
        self.assertFalse(payload["state"]["move_analysis"]["left"]["safe"])
        self.assertEqual(before, self.game.snapshot())

    def test_frozen_neutral_decoding_schema_and_fresh_context(self):
        request = adapter.build_request(self.game)
        settings = {key: value for key, value in request.items()
                    if key not in ("model", "messages", "response_format")}
        self.assertEqual(settings, {
            "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": 0,
            "max_tokens": 32, "stream": False, "cache_prompt": False,
            "frequency_penalty": 0, "presence_penalty": 0, "repeat_penalty": 1,
        })
        self.assertEqual(request["model"], "snake-qwen3-v1")
        self.assertEqual(adapter.ENDPOINT_PATH, "/v1/chat/completions")
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        schema = request["response_format"]["json_schema"]
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertEqual(schema["name"], "snake_action_v1")
        self.assertIs(schema["strict"], True)
        self.assertEqual(schema["schema"], {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["up", "down", "left", "right"]}},
            "required": ["action"], "additionalProperties": False})
        request["messages"].append({"role": "assistant", "content": "old history"})
        schema["schema"]["properties"]["action"]["enum"].clear()
        fresh = adapter.build_request(self.game)
        self.assertEqual(len(fresh["messages"]), 2)
        self.assertEqual(len(fresh["response_format"]["json_schema"]["schema"]
                             ["properties"]["action"]["enum"]), 4)

    def test_all_actions_parse_including_collision_without_override(self):
        for action in ("up", "down", "left", "right"):
            self.assertEqual(adapter.parse_response(json.dumps(response(action)))["action"], action)
        before = self.game.snapshot()
        calls = []
        def transport(wire):
            calls.append(wire)
            return json.dumps(response("left"))
        result = adapter.decide(self.game, transport)
        self.assertIsNone(result["error"])
        self.assertEqual(result["action"], "left")
        self.assertEqual(len(calls), 1)
        self.assertEqual(before, self.game.snapshot())
        self.game.apply(result["action"])
        self.assertFalse(self.game.alive)
        self.assertEqual(self.game.steps, 0)

    def test_duplicate_keys_rejected_inside_and_outside_content(self):
        self.assertRejected(response(content='{"action":"right","action":"left"}'))
        raw = json.dumps(response()).replace('"finish_reason": "stop"',
                                             '"finish_reason":"length","finish_reason":"stop"')
        self.assertRejected(raw)
        raw = json.dumps(response()).replace('"total_tokens": 323',
                                             '"total_tokens":323,"total_tokens":323')
        self.assertRejected(raw)

    def test_schema_violations_and_nonfinite_values_rejected(self):
        for content in ('{"action":"right","why":"food"}', '{}', '[]',
                        '{"action":"Right"}', '{"action":null}', '{"action":["up"]}',
                        '```json\n{"action":"up"}\n```', '{"action":"up"} extra',
                        '{"action":NaN}', '{"action":Infinity}', '{"action":-Infinity}'):
            with self.subTest(content=content):
                self.assertRejected(response(content=content))
        self.assertRejected(json.dumps(response()).replace('"prompt_tokens": 317', '"prompt_tokens": NaN'))
        self.assertRejected(json.dumps(response()).replace('"prompt_tokens": 317', '"prompt_tokens": 1e999'))
        self.assertRejected(b"\xff")
        self.assertRejected("{broken")

    def test_truncation_multiple_choices_tool_calls_refusals_and_reasoning(self):
        for reason in ("length", "tool_calls", "content_filter", None):
            value = response()
            value["choices"][0]["finish_reason"] = reason
            self.assertRejected(value)
        value = response()
        value["choices"].append(deepcopy(value["choices"][0]))
        self.assertRejected(value)
        for fields in ({"tool_calls": []}, {"function_call": {}}, {"refusal": "No"},
                       {"reasoning_content": "Think"}, {"reasoning": "Think"},
                       {"reasoning_text": "Think"}, {"role": "user"}, {"content": None}):
            with self.subTest(fields=fields):
                self.assertRejected(response(**fields))
        value = response()
        value["choices"][0]["index"] = False
        self.assertRejected(value)

    def test_token_usage_must_be_complete_consistent_and_within_budget(self):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            for bad in (-1, True, 1.0, "1", None):
                value = response()
                value["usage"][field] = bad
                self.assertRejected(value)
            value = response()
            del value["usage"][field]
            self.assertRejected(value)
        value = response()
        value["usage"].update(completion_tokens=33, total_tokens=350)
        self.assertRejected(value)
        value = response()
        value["usage"]["total_tokens"] = 324
        self.assertRejected(value)
        value["usage"] = {"prompt_tokens": 317, "completion_tokens": 32, "total_tokens": 349}
        self.assertEqual(adapter.parse_response(json.dumps(value))["usage"], value["usage"])

    def test_wire_hash_and_raw_evidence_are_lossless_without_retry(self):
        raw = json.dumps(response()).encode("utf-8")
        result = adapter.decide(self.game, lambda wire: raw)
        self.assertEqual(base64.b64decode(result["raw_response"]["base64"]), raw)
        wire = adapter.serialize_request(self.game)
        self.assertEqual(result["request_json"].encode("utf-8"), wire)
        self.assertEqual(result["request_sha256"], hashlib.sha256(wire).hexdigest())
        json.dumps(result, allow_nan=False)
        for raw in (b"\xff", '{"choices":[]}'):
            calls = []
            def invalid(wire):
                calls.append(wire)
                return raw
            result = adapter.decide(self.game, invalid)
            self.assertEqual(len(calls), 1)
            self.assertIsNone(result["action"])
            self.assertEqual(result["error"]["stage"], "response")
            if isinstance(raw, bytes):
                self.assertEqual(base64.b64decode(result["raw_response"]["base64"]), raw)
            else:
                self.assertEqual(result["raw_response"]["text"], raw)
        self.assertTrue(self.game.alive)
        self.assertEqual(self.game.steps, 0)

    def test_transport_and_terminal_failures_are_explicit_and_do_not_apply(self):
        calls = []
        def failed(wire):
            calls.append(wire)
            raise TimeoutError("controlled fixture timeout")
        result = adapter.decide(self.game, failed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["error"]["stage"], "transport")
        self.assertIsNone(result["action"])
        self.assertEqual(self.game.steps, 0)
        self.game.apply("left")
        result = adapter.decide(self.game, failed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["error"]["stage"], "request")


if __name__ == "__main__":
    unittest.main()
