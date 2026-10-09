"""CLEF-compatible four-choice adapter for the maze environment.

Preserves the official decision head and native argmax behavior: the adapter only
changes the question text and state schema relative to Snake. It never loads
weights or performs inference; `choice()` verifies a recorded model response
offline, mirroring `clef_snake.benchmark.choice`.

Unsupported in this implementation task: model acquisition/loading, the
inference server, and any live campaign runner. Complete offline game checks
(requirements: this package only) need no weights.
"""
from clef_snake.catalog import OFFICIAL_REPO, OFFICIAL_REV
from .maze import DIRECTIONS


def build_request(maze):
    """Deterministic structured request; schema fixed across precision conditions."""
    return maze.request()


def verify_provenance(value):
    """Assert a response came from the official repo/revision with no fallback."""
    assert value["repo"] == OFFICIAL_REPO and value["revision"] == OFFICIAL_REV, value
    assert value["fallback"] is False and value["safety_override"] is False, value


def choice(response):
    """Extract the native argmax action from a model response, verifying provenance.

    Mirrors the Snake driver's contract: probabilities cover exactly the four
    actions, and the recorded choice is the argmax (ties broken by the server,
    never overridden here).
    """
    verify_provenance(response["provenance"])
    answer = response["answers"]["move"]
    probabilities = answer["probabilities"]
    assert set(probabilities) == set(DIRECTIONS), probabilities.keys()
    selected = answer["choice"]
    assert selected in DIRECTIONS, selected
    assert probabilities[selected] == max(probabilities.values()), probabilities
    return selected
