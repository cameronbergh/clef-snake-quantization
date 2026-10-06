"""Offline boundary for Kev's native /v1/systemone decision interface.

No HTTP client, model loader, retry, persistence or game movement lives here.
An injected transport receives exact request bytes and returns TransportResponse.
The caller must durably save Attempt.to_record() before applying a valid choice.
Errors are technical stops, never substitute actions or completed game losses.
Contract checks do not establish checkpoint identity or numerical model parity.
"""
import base64
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Optional

from clef_snake.game import Game

ENDPOINT_PATH = "/v1/systemone"
MODEL_ALIAS = "kev-latest"
ACTIONS = ("up", "down", "left", "right")
# Kev rounds four probabilities to four decimals after choosing the raw argmax.
PROBABILITY_SUM_TOLERANCE = 0.00021
MAX_INPUT_TOKENS = 8192


class ResponseError(ValueError):
    """A raw response violates the native four-choice contract."""


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes

    def __post_init__(self):
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise TypeError("HTTP status must be an integer from 100 to 599")
        if not isinstance(self.body, bytes):
            raise TypeError("Response body must be bytes, retained without decoding")


class TransportFailure(Exception):
    """Transport error carrying any response bytes received before failure.

    Supply response=TransportResponse(status, partial_body) when available. An
    ordinary exception is also recorded, but cannot retain bytes it withholds.
    """

    def __init__(self, message, response=None):
        if response is not None and not isinstance(response, TransportResponse):
            raise TypeError("response must be TransportResponse or None")
        super().__init__(message)
        self.response = response


@dataclass(frozen=True)
class AttemptError:
    stage: str
    type: str
    message: str


@dataclass(frozen=True)
class Attempt:
    request_body: bytes
    request_sha256: str
    response_status: Optional[int] = None
    response_body: Optional[bytes] = None
    choice: Optional[str] = None
    error: Optional[AttemptError] = None

    def to_record(self):
        """JSON-safe lossless evidence, including malformed or partial bodies."""
        return {
            "endpoint_path": ENDPOINT_PATH,
            "request_base64": base64.b64encode(self.request_body).decode("ascii"),
            "request_sha256": self.request_sha256,
            "response_status": self.response_status,
            "response_base64": (base64.b64encode(self.response_body).decode("ascii")
                                if self.response_body is not None else None),
            "choice": self.choice,
            "error": asdict(self.error) if self.error is not None else None,
        }


def build_request(game):
    """Deep-copy the ordered game request, changing only the serving alias."""
    if not isinstance(game, Game) or not game.alive:
        raise ValueError("A live Game is required")
    request = deepcopy(game.request())
    request["model"] = MODEL_ALIAS
    return request


def make_request(game):
    """Exact UTF-8 wire bytes; no sorting, feature removal or prompt rewriting."""
    return json.dumps(build_request(game), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ResponseError("Nonfinite JSON constant: " + value)


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ResponseError("Nonfinite JSON number: " + value)
    return number


def _object(value, name):
    if not isinstance(value, dict):
        raise ResponseError(name + " must be an object")
    return value


def parse_response(raw):
    """Validate native JSON and preserve Kev's pre-rounding winning choice.

    Displayed probabilities can tie after rounding; never reselect the first
    displayed maximum. output_tokens counts serialized answers in Kev, rather
    than generated tokens, and is therefore allowed to be nonzero.
    """
    if not isinstance(raw, bytes):
        raise ResponseError("Raw response must be UTF-8 bytes")
    try:
        response = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_constant=_constant, parse_float=_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseError("Malformed JSON or UTF-8: " + str(exc)) from exc
    response = _object(response, "Response")
    if response.get("error") is not None:
        raise ResponseError("Server returned an error")
    if response.get("model") != MODEL_ALIAS:
        raise ResponseError("Response model alias does not match request")
    if "truncated" in response and response["truncated"] is not False:
        raise ResponseError("Truncated input is forbidden")
    answers = _object(response.get("answers"), "Answers")
    if set(answers) != {"move"}:
        raise ResponseError("Exactly one move answer is required")
    answer = _object(answers["move"], "Move answer")
    if answer.get("type") != "choice":
        raise ResponseError("Move answer must be a native choice")
    probabilities = _object(answer.get("probabilities"), "Probabilities")
    if set(probabilities) != set(ACTIONS):
        raise ResponseError("Exactly the four directional probabilities are required")
    if any(type(p) not in (int, float) or not 0 <= p <= 1 or not math.isfinite(p)
           for p in probabilities.values()):
        raise ResponseError("Probabilities must be finite numbers in [0, 1]")
    if abs(math.fsum(probabilities.values()) - 1) > PROBABILITY_SUM_TOLERANCE:
        raise ResponseError("Probability sum is outside the rounding tolerance")
    selected = answer.get("choice")
    if not isinstance(selected, str) or selected not in ACTIONS:
        raise ResponseError("Choice must be one of the four exact directions")
    if probabilities[selected] != max(probabilities.values()):
        raise ResponseError("Selected choice is not a displayed probability maximum")
    usage = _object(response.get("usage"), "Usage")
    for key in ("input_tokens", "output_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ResponseError("Usage counts must be nonnegative integers")
    if not 0 < usage["input_tokens"] <= MAX_INPUT_TOKENS:
        raise ResponseError("Complete input exceeds the proposed 8192-token row limit")
    if "state_tokens" in usage or "state_tokens_used" in usage:
        if any(type(usage.get(key)) is not int or usage[key] < 0
               for key in ("state_tokens", "state_tokens_used")):
            raise ResponseError("State token counts must both be nonnegative integers")
        if usage["state_tokens"] != usage["state_tokens_used"]:
            raise ResponseError("State token accounting reports truncation")
    return selected


def attempt(game, transport):
    """Make exactly one injected transport call; return evidence without moving.

    Invalid/terminal input raises before transport. Once transport is attempted,
    exceptions, HTTP errors and invalid responses return an Attempt with no
    choice. The eventual runner owns timing, provenance and durable logging.
    """
    wire = make_request(game)
    digest = hashlib.sha256(wire).hexdigest()
    response = None
    stage = "transport"
    try:
        response = transport(wire)
        if not isinstance(response, TransportResponse):
            response = None
            raise TypeError("Transport must return TransportResponse")
        stage = "http"
        if not 200 <= response.status < 300:
            raise ResponseError("HTTP status " + str(response.status))
        stage = "response"
        selected = parse_response(response.body)
    except Exception as exc:
        if isinstance(exc, TransportFailure):
            response = exc.response
        return Attempt(wire, digest,
                       response.status if response is not None else None,
                       response.body if response is not None else None,
                       error=AttemptError(stage, type(exc).__name__, str(exc)))
    return Attempt(wire, digest, response.status, response.body, choice=selected)
