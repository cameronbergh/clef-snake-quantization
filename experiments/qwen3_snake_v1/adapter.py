"""Offline request/response boundary for native llama.cpp chat completions.

The server applies its pinned native model chat template. This adapter does not
load a model, call HTTP, retry, apply moves, or imitate CLEF head probabilities.
Callers supply a transport, persist each complete decision record, and apply its
valid action exactly once. A non-null error is a technical stop, never a loss or
a substitute move. There is no response truncation or hidden history.
"""
import base64
from copy import deepcopy
import hashlib
import json
import math

from clef_snake.game import Game

ENDPOINT_PATH = "/v1/chat/completions"
MODEL_ALIAS = "snake-qwen3-v1"
MAX_TOKENS = 32
ACTIONS = ("up", "down", "left", "right")
SYSTEM_PROMPT = (
    "You control Snake using the supplied factual state and move question. "
    "Select exactly one of the four actions, including an unsafe action if "
    "that is your choice. Return only a JSON object with the single key "
    '"action" and one of "up", "down", "left", or "right". '
    "Do not include explanation, reasoning, Markdown, or text outside that "
    "JSON object."
)
ACTION_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": list(ACTIONS)}},
    "required": ["action"],
    "additionalProperties": False,
}


class ResponseError(ValueError):
    """Response violates the frozen action or completion contract."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def build_request(game):
    """Return a new ordered request; all factual fields and actions are retained."""
    if not isinstance(game, Game) or not game.alive:
        raise ValueError("A live Game is required")
    original = game.request()
    return {
        "model": MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _json({
                "state": original["state"], "questions": original["questions"],
            })},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "snake_action_v1", "strict": True,
            "schema": deepcopy(ACTION_SCHEMA),
        }},
        "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0,
        "seed": 0, "max_tokens": MAX_TOKENS, "stream": False,
        "cache_prompt": False, "frequency_penalty": 0,
        "presence_penalty": 0, "repeat_penalty": 1,
    }


def serialize_request(game):
    """Exact UTF-8 wire bytes; no sorting or chat template substitution."""
    return _json(build_request(game)).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ResponseError("Nonfinite JSON value: " + value)


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ResponseError("Nonfinite JSON number: " + value)
    return number


def _load(raw):
    if not isinstance(raw, (str, bytes)):
        raise ResponseError("Expected a raw JSON string or UTF-8 bytes")
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant,
                          parse_float=_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseError("Malformed JSON or UTF-8: " + str(exc)) from exc


def _object(value, label):
    if not isinstance(value, dict):
        raise ResponseError(label + " must be an object")
    return value


def parse_response(raw):
    """Accept one stopped, nonreasoning completion containing only one action."""
    response = _object(_load(raw), "Response")
    if response.get("error") is not None:
        raise ResponseError("Server returned an error")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ResponseError("Exactly one completion choice is required")
    choice = _object(choices[0], "Choice")
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise ResponseError("Completion choice index must be zero")
    if choice.get("finish_reason") != "stop":
        raise ResponseError("Completion must finish with stop; no partial action accepted")
    message = _object(choice.get("message"), "Message")
    if message.get("role") != "assistant":
        raise ResponseError("Completion role must be assistant")
    if message.get("tool_calls") is not None or message.get("function_call") is not None:
        raise ResponseError("Tool or function calls are forbidden")
    if message.get("refusal") not in (None, ""):
        raise ResponseError("Refusal is not an action")
    for key in ("reasoning", "reasoning_content", "reasoning_text"):
        if message.get(key) not in (None, ""):
            raise ResponseError("Reasoning output is forbidden")
    content = message.get("content")
    if not isinstance(content, str):
        raise ResponseError("Assistant content must be a JSON string")
    action = _object(_load(content), "Action")
    if set(action) != {"action"} or action["action"] not in ACTIONS:
        raise ResponseError("Content must contain only one of the four exact actions")
    usage = _object(response.get("usage"), "Usage")
    counts = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if type(value) is not int or value < 0:
            raise ResponseError("Usage counts must be nonnegative integers")
        counts[key] = value
    if counts["completion_tokens"] > MAX_TOKENS:
        raise ResponseError("Completion exceeds the frozen token budget")
    if counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]:
        raise ResponseError("Usage total is inconsistent")
    return {"action": action["action"], "usage": counts}


def decide(game, transport):
    """Record one injected transport call without applying any action.

    transport(wire_bytes) must return the complete response as str or bytes.
    Bytes are retained losslessly as base64, including invalid UTF-8. Records are
    JSON serializable. HTTP headers/status, timing and model provenance belong
    to the eventual runner and are not fabricated by this offline adapter.
    """
    record = {"request": None, "request_json": None, "request_sha256": None,
              "raw_response": None, "action": None, "usage": None, "error": None}
    stage = "request"
    try:
        record["request"] = build_request(game)
        wire = _json(record["request"]).encode("utf-8")
        record["request_json"] = wire.decode("utf-8")
        record["request_sha256"] = hashlib.sha256(wire).hexdigest()
        stage = "transport"
        raw = transport(wire)
        stage = "response"
        if isinstance(raw, bytes):
            record["raw_response"] = {
                "type": "bytes", "base64": base64.b64encode(raw).decode("ascii"),
            }
        elif isinstance(raw, str):
            record["raw_response"] = {"type": "text", "text": raw}
        record.update(parse_response(raw))
    except Exception as exc:
        record["error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    return record
