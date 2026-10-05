import copy
import json
from pathlib import Path
import unittest

from experiments.qwen3_snake_v1.verify import validate_design, verify

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "experiments/qwen3_snake_v1"


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((PREP / "protocol.json").read_text())
        self.seeds = json.loads((PREP / "seed-manifest.json").read_text())
        self.schedule = json.loads((PREP / "schedule.json").read_text())
        self.published = set()
        for f in (ROOT / "data").glob("*/seed-manifest.json"):
            self.published.update(json.loads(f.read_text())["seeds"])

    def test_entire_frozen_preparation_and_external_budget(self):
        result = verify(ROOT)
        self.assertEqual(result["planned_games"], 90)
        self.assertFalse(result["execution_authorized"])

    def test_reusing_discovery_or_preflight_seed_is_rejected(self):
        for replacement in [next(iter(self.published)), self.seeds["preflight_seeds"][0]]:
            seeds = copy.deepcopy(self.seeds)
            seeds["seeds"][0] = replacement
            with self.assertRaises(ValueError):
                validate_design(self.protocol, seeds, self.schedule, self.published)

    def test_duplicate_or_reordered_games_are_rejected(self):
        for mutation in ["duplicate", "swap"]:
            schedule = copy.deepcopy(self.schedule)
            order = schedule["execution_order"]
            if mutation == "duplicate":
                order[1] = order[0].copy()
            else:
                order[0], order[1] = order[1], order[0]
            with self.assertRaises(ValueError):
                validate_design(self.protocol, self.seeds, schedule, self.published)

    def test_changed_decoder_does_not_silently_enter_same_protocol(self):
        self.protocol["request"]["decoding"]["temperature"] = 0.5
        with self.assertRaises(ValueError):
            validate_design(self.protocol, self.seeds, self.schedule, self.published)


if __name__ == "__main__":
    unittest.main()
