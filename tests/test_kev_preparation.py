"""Offline integrity and refusal checks, never tokenizer/model validation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from clef_snake.game import Game
from experiments.kev4b_v1.adapter import make_request
from experiments.kev4b_v1.verify import validate_plans, validate_token_count, verify, verify_fixtures

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "experiments/kev4b_v1"


class KevPreparationTests(unittest.TestCase):
    def setUp(self):
        read = lambda name: json.loads((FOLDER / name).read_text())
        self.protocol = read("protocol.json")
        self.storage = read("storage-plan.json")
        self.artifacts = read("artifact-plan.json")
        self.fixtures = read("preflight-cases.json")

    def test_entire_preparation_is_offline_and_campaign_unfrozen(self):
        result = verify(ROOT)
        self.assertEqual(result["model_calls_made_by_verifier"], 0)
        self.assertFalse(result["main_campaign_frozen"])
        self.assertFalse(result["runtime_parity_verified"])

    def test_native_request_matches_independent_discovery_fixtures(self):
        for case in self.fixtures["cases"]:
            with self.subTest(case=case["id"]):
                expected = deepcopy(case["request"])
                game = Game(case["seed"])
                state = expected["state"]
                game.body = deepcopy(state["body"])
                game.food = state["food"][:]
                game.direction = state["direction"]
                expected["model"] = "kev-latest"
                self.assertEqual(make_request(game), json.dumps(expected,
                    ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())

    def test_mutated_source_fixture_is_rejected(self):
        self.fixtures["cases"][0]["request"]["state"]["move_analysis"]["left"]["safe"] = True
        with self.assertRaises(ValueError):
            verify_fixtures(ROOT, self.fixtures)

    def test_synthetic_token_admission_boundaries_not_actual_token_counts(self):
        for value in (1, 8192):
            self.assertEqual(validate_token_count(value), value)
        for value in (0, 8193, -1, True, 1.0, "8192", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_token_count(value)

    def test_budget_expansion_or_early_campaign_is_rejected(self):
        mutations = [lambda p: p["preflight"].update(max_forward_calls=65),
                     lambda p: p["preflight"].update(max_wall_seconds=901),
                     lambda p: p["preflight"]["phases"][0].update(calls=9),
                     lambda p: p["main_campaign"].update(seeds=["premature"]),
                     lambda p: p["authorization"]["scope"].update(scored_games=1),
                     lambda p: p["validation"].update(runtime_parity_verified=True),
                     lambda p: p["runtime_proposal"]["trusted_launch_environment"].update(KEV_PREFIX_CACHE="4")]
        for change in mutations:
            plan = deepcopy(self.protocol)
            change(plan)
            with self.assertRaises(ValueError):
                validate_plans(plan, self.storage, self.artifacts)

    def test_external_storage_and_exact_inventory_are_enforced(self):
        for change in [lambda s: s.update(internal=True),
                       lambda s: s["environment"].update(HF_HOME="/tmp/fallback"),
                       lambda s: s["environment"].update(TMPDIR=s["planned_root"] + "/../escape"),
                       lambda s: s["allocations_bytes"].update(temporary_download_and_conversion=100*1024**3)]:
            plan = deepcopy(self.storage)
            change(plan)
            with self.assertRaises(ValueError):
                validate_plans(self.protocol, plan, self.artifacts)
        self.artifacts["sources"][0]["artifacts"][0]["bytes"] += 1
        with self.assertRaises(ValueError):
            validate_plans(self.protocol, self.storage, self.artifacts)


if __name__ == "__main__":
    unittest.main()
