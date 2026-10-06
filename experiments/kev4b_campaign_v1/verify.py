"""Verify the prospective campaign bundle offline; never load a tokenizer/model."""
import json
from pathlib import Path

from .runner import validate_bundle
from .audit import read_frozen


def verify(repo=None):
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    plan = validate_bundle(root)
    independent = read_frozen(root)
    if independent["freeze_sha256"] != plan.freeze_sha256:
        raise ValueError("Independent freeze readers disagree")
    return {
        "status": "verified_prospective_preparation",
        "freeze_sha256": plan.freeze_sha256,
        "frozen_files": len(plan.input_hashes),
        "paired_seeds": len(plan.seeds),
        "planned_games": len(plan.games),
        "execution_authorized": False,
        "tokenizer_context_validation": "pending_external_report",
        "model_loads": 0,
        "forward_calls": 0,
        "games_executed": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
