import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.qwen3_snake_v1.storage import check_storage, validate_volume


class ExternalStorageTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((Path(__file__).resolve().parents[1] /
            "experiments/qwen3_snake_v1/storage-plan.json").read_text())
        self.info = {"Internal": False, "VolumeName": "Models",
                     "MountPoint": "/Volumes/Models"}

    def test_selected_external_volume_and_exact_free_space_gate(self):
        gate = self.plan["launch_gate_free_bytes"]
        result = validate_volume(self.info, True, gate, self.plan)
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["remaining_after_reservations_bytes"],
                         gate - self.plan["persistent_phase_required_free_bytes"])
        with self.assertRaises(ValueError):
            validate_volume(self.info, True, gate - 1, self.plan)

    def test_internal_x9_missing_and_unknown_volumes_fail_closed(self):
        for overrides in [{"Internal": True}, {"Internal": None},
                          {"VolumeName": "Crucial X9"},
                          {"MountPoint": "/tmp/Models"}]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_volume(dict(self.info, **overrides), True, 10**12, self.plan)
        with self.assertRaises(ValueError):
            validate_volume(self.info, False, 10**12, self.plan)

    def test_absent_mount_does_not_query_disk_or_create_fallback(self):
        with patch("os.path.ismount", return_value=False), \
                patch("subprocess.check_output") as query, \
                patch.object(Path, "mkdir") as mkdir:
            with self.assertRaises(ValueError):
                check_storage(self.plan)
            query.assert_not_called()
            mkdir.assert_not_called()

    def test_cache_escape_rejected_before_disk_query(self):
        plan = copy.deepcopy(self.plan)
        plan["runtime_storage_environment"]["HF_HOME"] = "/tmp/internal-cache"
        with patch("os.path.ismount", return_value=True), \
                patch("subprocess.check_output") as query:
            with self.assertRaises(ValueError):
                check_storage(plan)
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
