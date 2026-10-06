"""Synthetic filesystem/journal tests: no harness imports, models or network."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.kev4b_v1 import retry_budget as budget


class RetryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "explicit-offline-fix-retry"
        self.rows = {}
        self.pins = {}
        self.patch = patch.object(budget, "EXPECTED_PRIOR_JOURNAL_SHA256", self.pins)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.make_history()

    @staticmethod
    def row(event, when, **fields):
        return {"event": event, "monotonic": when, "unix_time": 1700000000 + when, **fields}

    def make_history(self, elapsed=180.856):
        self.rows.clear()
        for index, run in enumerate(budget.PRIOR_RUNS):
            (self.root / run).mkdir(exist_ok=True)
            start, end = (1000, 1011.2) if index == 0 else (1134.9, 1000 + elapsed)
            supervisor = [self.row("supervisor_start", start, deadline=1900,
                                   harness_sha256="a" * 64),
                          self.row("resource_sample", start + 1, current_rss_bytes=1000),
                          self.row("worker_exit", end, returncode=1, wall_seconds=end - start)]
            events = [self.row("worker_start", start + .1,
                               limits={"seconds": 900, "decisions": 64,
                                       "planned_decisions": 60, "games": 0}),
                      self.row("verified_file", start + .2, path="config.json", bytes=2),
                      self.row("fatal", end - .1, decisions_reserved=0, games=0)]
            self.rows[(run, "supervisor.jsonl")] = supervisor
            self.rows[(run, "events.jsonl")] = events
        self.seal()

    def write(self, key, *, resequence=True):
        rows = self.rows[key]
        if resequence:
            for sequence, row in enumerate(rows, 1):
                row["sequence"] = sequence
        raw = ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode()
        (self.root / key[0] / key[1]).write_bytes(raw)
        self.pins[key] = hashlib.sha256(raw).hexdigest()

    def seal(self, *, bind=True, resequence=True):
        original, continuation = budget.PRIOR_RUNS
        for name in ("supervisor.jsonl", "events.jsonl"):
            self.write((original, name), resequence=resequence)
        if bind:
            self.rows[(continuation, "supervisor.jsonl")][0]["prior_setup"] = {
                "path": str(self.root / original),
                "supervisor_sha256": self.pins[(original, "supervisor.jsonl")],
                "events_sha256": self.pins[(original, "events.jsonl")],
                "prior_harness_sha256": "a" * 64,
                "original_deadline": 1900, "prior_decisions_reserved": 0,
            }
        for name in ("supervisor.jsonl", "events.jsonl"):
            self.write((continuation, name), resequence=resequence)

    def claim(self):
        return budget.claim_retry_budget(self.root, self.output)

    def reject(self, message=None):
        with self.assertRaises(budget.RetryBudgetError) as caught:
            self.claim()
        if message:
            self.assertIn(message, str(caught.exception))
        self.assertFalse((self.root / budget.CLAIM_NAME).exists())
        self.assertFalse(self.output.exists())

    def test_full_span_and_gap_charged_not_summed_worker_durations(self):
        claim = self.claim()
        self.assertAlmostEqual(claim["prior_elapsed_seconds"], 180.856)
        self.assertEqual(claim["prior_charged_seconds"], 181)
        self.assertEqual(claim["remaining_seconds"], 719)
        self.assertEqual(claim["prior_decisions"], 0)
        self.assertEqual(claim["remaining_decisions"], 64)
        self.assertEqual(claim["planned_decisions"], 60)
        self.assertEqual(len(claim["evidence"]["journals"]), 4)
        self.assertEqual(json.loads((self.root / budget.CLAIM_NAME).read_text()), claim)
        self.assertFalse(self.output.exists())

    def test_integer_rounding_and_fractional_boundary(self):
        for elapsed, charged, remaining in ((180, 180, 719), (181, 181, 719),
                                             (181.000001, 182, 718), (898.1, 899, 1)):
            with self.subTest(elapsed=elapsed):
                self.make_history(elapsed)
                claim = self.claim()
                self.assertEqual((claim["prior_charged_seconds"], claim["remaining_seconds"]),
                                 (charged, remaining))
                (self.root / budget.CLAIM_NAME).unlink()  # synthetic fixture reset only

    def test_exhausted_and_invalid_time(self):
        for elapsed in (899.1, 900, 901, -1):
            with self.subTest(elapsed=elapsed):
                self.make_history(elapsed)
                self.reject()

    def test_one_use_even_if_output_never_created(self):
        self.claim()
        with self.assertRaisesRegex(budget.RetryBudgetError, "already claimed"):
            self.claim()
        with self.assertRaises(budget.RetryBudgetError):
            budget.claim_retry_budget(self.root, self.root / "different-retry")

    def test_exclusive_claim_concurrent_callers(self):
        def attempt(number):
            try:
                budget.claim_retry_budget(self.root, self.root / f"retry-{number}")
                return True
            except budget.RetryBudgetError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, (1, 2)))
        self.assertEqual(sum(results), 1)

    def test_tampered_bytes_fail_pinned_hash(self):
        path = self.root / budget.PRIOR_RUNS[1] / "events.jsonl"
        path.write_bytes(path.read_bytes().replace(b"config.json", b"tamper.json"))
        self.reject("hash mismatch")

    def test_continuation_bindings_required(self):
        second = self.rows[(budget.PRIOR_RUNS[1], "supervisor.jsonl")][0]
        for key, value in (("events_sha256", "b" * 64), ("supervisor_sha256", "b" * 64),
                           ("prior_harness_sha256", "b" * 64), ("original_deadline", 1901),
                           ("prior_decisions_reserved", 1), ("path", "/wrong")):
            with self.subTest(key=key):
                old = second["prior_setup"][key]
                second["prior_setup"][key] = value
                self.seal(bind=False)
                self.reject()
                second["prior_setup"][key] = old

    def test_deadline_not_reset_or_extended(self):
        for run in budget.PRIOR_RUNS:
            with self.subTest(run=run):
                row = self.rows[(run, "supervisor.jsonl")][0]
                row["deadline"] = 1901
                self.seal()
                self.reject("deadline")
                row["deadline"] = 1900

    def test_unaccounted_attempt_or_other_entry(self):
        for is_directory in (True, False):
            path = self.root / "unaccounted-attempt"
            path.mkdir() if is_directory else path.write_text("unknown")
            self.reject("Unaccounted")
            path.rmdir() if is_directory else path.unlink()

    def test_output_must_be_absent_direct_child(self):
        (self.root / "existing").mkdir()
        for output in (Path("relative"), self.root / "existing", self.root / "nested" / "retry",
                       self.root.parent / "outside", self.root / budget.CLAIM_NAME):
            with self.subTest(output=output), self.assertRaises(budget.RetryBudgetError):
                budget.claim_retry_budget(self.root, output)

    def test_no_decision_or_backbone_events(self):
        key = (budget.PRIOR_RUNS[1], "events.jsonl")
        for event in ("decision_start", "raw_decision", "backbone_pass_start", "backbone_pass_end"):
            with self.subTest(event=event):
                self.rows[key].insert(-1, self.row(event, 1160, number=1))
                self.seal()
                self.reject("decision/backbone")
                self.rows[key].pop(-2)

    def test_count_type_and_zero_required(self):
        fatal = self.rows[(budget.PRIOR_RUNS[1], "events.jsonl")][-1]
        for field in ("decisions_reserved", "games"):
            for value in (1, -1, True, 0.0, "0", None):
                with self.subTest(field=field, value=value):
                    fatal[field] = value
                    self.seal()
                    self.reject()
            fatal[field] = 0

    def test_terminal_records_required(self):
        for name in ("supervisor.jsonl", "events.jsonl"):
            with self.subTest(name=name):
                key = (budget.PRIOR_RUNS[1], name)
                end = self.rows[key].pop()
                self.seal()
                self.reject()
                self.rows[key].append(end)

    def test_success_exit_not_eligible(self):
        exit_row = self.rows[(budget.PRIOR_RUNS[1], "supervisor.jsonl")][-1]
        for value in (0, -9, True, 1.0):
            with self.subTest(value=value):
                exit_row["returncode"] = value
                self.seal()
                self.reject()

    def test_noncontiguous_and_boolean_sequences(self):
        row = self.rows[(budget.PRIOR_RUNS[1], "events.jsonl")][1]
        for value in (1, 3, True, 2.0):
            with self.subTest(value=value):
                row["sequence"] = value
                self.seal(resequence=False)
                self.reject("sequence")

    def test_nonfinite_times_and_nested_numbers(self):
        row = self.rows[(budget.PRIOR_RUNS[1], "events.jsonl")][1]
        for field, value in (("monotonic", float("nan")), ("unix_time", float("inf")),
                             ("bytes", float("-inf")), ("monotonic", True)):
            with self.subTest(field=field, value=value):
                old = row[field]
                row[field] = value
                self.seal()
                self.reject()
                row[field] = old

    def test_partial_json_and_duplicate_keys(self):
        key = (budget.PRIOR_RUNS[1], "events.jsonl")
        path = self.root / key[0] / key[1]
        original = path.read_bytes()
        for raw in (original[:-1], original + b"{\n",
                    original.replace(b'"sequence":1', b'"sequence":1,"sequence":1', 1)):
            with self.subTest(raw_size=len(raw)):
                path.write_bytes(raw)
                self.pins[key] = hashlib.sha256(raw).hexdigest()
                self.reject()

    def test_symlinked_journal_and_run_rejected(self):
        run = self.root / budget.PRIOR_RUNS[1]
        path = run / "events.jsonl"
        moved = run / "saved-events.jsonl"
        path.rename(moved)
        path.symlink_to(moved)
        self.reject("symlink")


if __name__ == "__main__":
    unittest.main()
