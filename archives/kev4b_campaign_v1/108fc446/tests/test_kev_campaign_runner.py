"""Synthetic offline campaign tests. No model, tokenizer, network or real run."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from clef_snake.game import Game
from experiments.kev4b_v1 import adapter
from experiments.kev4b_campaign_v1 import runner


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def synthetic_plan(*, seed_count=1, cap=3):
    seeds = tuple({"seed_round": i, "seed": f"{i:032x}"} for i in range(1, seed_count + 1))
    orders = list(itertools.permutations(runner.CONDITIONS))
    games = []
    for index, seed in enumerate(seeds):
        for variant in orders[index % 6]:
            games.append({"sequence": len(games) + 1, "game_id": f"synthetic-{index + 1:02}-{variant}",
                          **seed, "variant": variant})
    limits = {**runner.LIMITS, "successful_move_cap": cap}
    protocol = {"campaign_id": runner.CAMPAIGN_ID, "conditions": list(runner.CONDITIONS),
                "inference_authorized_by_preparation": False, "limits": limits,
                "warmups": {"fixture_path": "experiments/kev4b_v1/preflight-cases.json",
                            "case_ids": ["bf16-round-01-first", "bf16-round-01-middle"]},
                "storage": {"campaign_subdirectory": "campaigns/kev4b_campaign_v1"}}
    fixtures = []
    for i, name in enumerate(protocol["warmups"]["case_ids"]):
        request = Game(f"{i + 1000:032x}").request()
        fixtures.append({"id": name, "request": request, "request_sha256": runner.digest(runner.wire(request))})
    return runner.CampaignPlan(protocol, seeds, tuple(games), tuple(fixtures), "f" * 64, {})


class SyntheticBackend:
    synthetic = True

    def __init__(self, journal, *, choice="left", fail_on=None, malformed=False, clock=None, delayed=False):
        self.journal, self.choice = journal, choice
        self.fail_on, self.malformed = fail_on, malformed
        self.clock, self.delayed = clock, delayed
        self.calls = 0
        self.loads, self.unloads = [], 0
        self.last_response_sequence = None
        self.after_evaluate = None

    def verify_inputs(self):
        return {"synthetic": True, "weights_loaded": False}

    def load(self, variant, *, game_id):
        self.loads.append((game_id, variant))
        return {"synthetic": True}

    def evaluate(self, raw, context):
        self.calls += 1
        if context["call_id"] != self.calls:
            raise AssertionError("Native synthetic call count differs")
        self.journal.write("decision_start", **context, decision_id=self.calls,
                           request_sha256=runner.digest(raw))
        self.journal.write("native_forward_start", **context)
        if self.fail_on == self.calls:
            self.journal.write("decision_error", **context, message="synthetic failure")
            raise adapter.TransportFailure("synthetic failure", response=adapter.TransportResponse(503, b"partial native error"))
        if self.delayed and context["phase"] == "game":
            self.clock.now += 121
        p = {key: 0.7 if key == self.choice else 0.1 for key in adapter.ACTIONS}
        response = runner.wire({"model": "kev-latest", "answers": {"move": {
            "type": "choice", "choice": self.choice, "probabilities": p}},
            "usage": {"input_tokens": 10, "output_tokens": 10}})
        if self.malformed and context["phase"] == "game":
            response = b'{"broken":'
        self.last_response_sequence = self.journal.write("native_response", **context,
                                                         response_sha256=runner.digest(response))
        if self.after_evaluate:
            self.after_evaluate()
        return adapter.TransportResponse(200, response)

    def unload(self):
        self.unloads += 1
        return {"synthetic": True, "unchanged": True}


class CampaignRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()

    def setup_run(self, plan=None, **backend_options):
        plan = plan or synthetic_plan()
        journal = runner.Journal(self.folder / "runner.jsonl", synthetic=True)
        native = runner.Journal(self.folder / "native.jsonl", synthetic=True)
        self.addCleanup(journal.close)
        self.addCleanup(native.close)
        backend = SyntheticBackend(native, **backend_options)
        guard = runner.CampaignGuard(plan.protocol["limits"], clock=backend_options.get("clock") or __import__("time").monotonic)
        campaign = runner.CampaignRunner(plan, backend, journal, guard, synthetic=True)
        return campaign, backend, guard

    def rows(self, name="runner.jsonl"):
        return [json.loads(line) for line in (self.folder / name).read_text().splitlines()]

    def test_counterbalanced_schedule_and_180_warmups_are_kept(self):
        plan = synthetic_plan(seed_count=30)
        campaign, backend, guard = self.setup_run(plan)
        summary = campaign.run()
        self.assertEqual(backend.loads, [(g["game_id"], g["variant"]) for g in plan.games])
        self.assertEqual(backend.unloads, 90)
        self.assertEqual(summary, {"games_completed": 90, "decisions_reserved": 270})
        rows = self.rows()
        attempts = [row for row in rows if row["event"] == "attempt"]
        self.assertEqual(sum(row["phase"] == "warmup" for row in attempts), 180)
        self.assertEqual([row["call_id"] for row in attempts], list(range(1, 271)))
        self.assertEqual([row["sequence"] for row in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(row["synthetic"] for row in rows + self.rows("native.jsonl")))

    def test_unsafe_choice_is_applied_and_collision_is_one_attempt(self):
        campaign, backend, guard = self.setup_run()
        with patch.object(Game, "apply", autospec=True, side_effect=Game.apply) as apply:
            campaign.run()
        self.assertEqual(apply.call_count, 3)  # Never apply the six warmups.
        completed = [row for row in self.rows() if row["event"] == "game_complete"]
        self.assertTrue(all(row["attempts"] == 1 and row["steps"] == 0 for row in completed))
        self.assertTrue(all(row["end_reason"] == "collision" and not row["censored"] for row in completed))

    def test_alive_cap_is_censored_and_attempts_do_not_exceed_cap(self):
        campaign, backend, guard = self.setup_run(choice="right")
        campaign.run()
        completed = [row for row in self.rows() if row["event"] == "game_complete"]
        self.assertEqual(guard.calls, 3 * (2 + 3))
        self.assertTrue(all(row["steps"] == row["attempts"] == 3 and row["alive"]
                            and row["end_reason"] == "cap" and row["censored"] for row in completed))

    def test_attempt_is_fsynced_before_game_apply(self):
        campaign, backend, guard = self.setup_run()
        real_fsync, original_apply = os.fsync, Game.apply
        count_after_native = []
        with patch.object(os, "fsync", wraps=real_fsync) as fsync:
            backend.after_evaluate = lambda: count_after_native.append(fsync.call_count)
            def checked_apply(game, choice):
                self.assertEqual(self.rows()[-1]["event"], "attempt")
                self.assertGreater(fsync.call_count, count_after_native[-1])
                return original_apply(game, choice)
            with patch.object(Game, "apply", checked_apply):
                campaign.run()

    def test_partial_error_is_retained_and_stops_all_games_without_score(self):
        campaign, backend, guard = self.setup_run(fail_on=3)
        with self.assertRaises(runner.CampaignError):
            campaign.run()
        rows = self.rows()
        self.assertEqual(guard.calls, 3)
        self.assertEqual(len(backend.loads), 1)
        self.assertFalse(any(r["event"] in {"transition", "game_complete", "run_complete"} for r in rows))
        attempt = [r for r in rows if r["event"] == "attempt"][-1]
        self.assertEqual(attempt["attempt"]["response_status"], 503)
        self.assertEqual(attempt["attempt"]["response_sha256"], runner.digest(b"partial native error"))
        self.assertEqual(attempt["attempt"]["error"]["stage"], "transport")
        self.assertTrue(any(r["event"] == "run_error" for r in rows))

    def test_malformed_response_is_technical_failure_not_a_collision(self):
        campaign, backend, guard = self.setup_run(malformed=True)
        with self.assertRaises(runner.CampaignError):
            campaign.run()
        self.assertEqual(guard.calls, 3)
        self.assertFalse(any(r["event"] == "game_complete" for r in self.rows()))
        self.assertEqual([r for r in self.rows() if r["event"] == "attempt"][-1]["attempt"]["error"]["stage"], "response")

    def test_warmup_failure_stops_before_gameplay_and_counts_reserved_call(self):
        campaign, backend, guard = self.setup_run(fail_on=1)
        with self.assertRaises(runner.CampaignError):
            campaign.run()
        self.assertEqual(guard.calls, 1)
        self.assertFalse(any(r["event"] == "transition" for r in self.rows()))

    def test_decision_timeout_records_attempt_but_does_not_apply_move(self):
        campaign, backend, guard = self.setup_run(clock=Clock(), delayed=True)
        with self.assertRaisesRegex(runner.CampaignError, "Per-decision"):
            campaign.run()
        self.assertEqual(guard.calls, 3)
        self.assertFalse(any(r["event"] in {"transition", "game_complete"} for r in self.rows()))

    def test_budget_exhausted_by_warmups_prevents_next_native_call(self):
        plan = synthetic_plan()
        plan.protocol["limits"]["max_decision_evaluations"] = 2
        campaign, backend, guard = self.setup_run(plan)
        with self.assertRaisesRegex(runner.CampaignError, "budget exhausted"):
            campaign.run()
        self.assertEqual(backend.calls, 2)
        self.assertEqual(len([r for r in self.rows() if r["event"] == "call_reserved"]), 2)

    def test_real_runner_cannot_be_constructed_without_approval(self):
        campaign, backend, guard = self.setup_run()
        with self.assertRaisesRegex(runner.CampaignError, "separate approval"):
            runner.CampaignRunner(campaign.plan, backend, campaign.journal, guard)
        backend.synthetic = False
        with self.assertRaisesRegex(runner.CampaignError, "explicitly synthetic"):
            runner.CampaignRunner(campaign.plan, backend, campaign.journal, guard, synthetic=True)

    def test_existing_evidence_path_is_not_overwritten(self):
        journal = runner.Journal(self.folder / "existing.jsonl", synthetic=True)
        journal.write("preserved", value=1)
        journal.close()
        with self.assertRaises(FileExistsError):
            runner.Journal(self.folder / "existing.jsonl", synthetic=True)

    def test_approval_must_match_exact_freeze_and_limits(self):
        plan = synthetic_plan()
        document = {"schema_version": 1, "campaign_id": runner.CAMPAIGN_ID,
                    "freeze_sha256": plan.freeze_sha256, "execution_authorized": True,
                    "one_use_id": "synthetic-approval", "limits": plan.protocol["limits"],
                    "context_report_sha256": "c" * 64}
        path = self.folder / "approval.json"
        for key, wrong in (("execution_authorized", False), ("freeze_sha256", "a" * 64),
                           ("limits", {}), ("one_use_id", "../escape"), ("context_report_sha256", "d" * 64)):
            bad = {**document, key: wrong}
            path.write_bytes(runner.wire(bad))
            with self.subTest(key=key), self.assertRaises(runner.CampaignError):
                runner.validate_approval(path, plan, self.folder, context_report_sha256="c" * 64)
        path.write_bytes(runner.wire(document))
        self.assertEqual(runner.validate_approval(path, plan, self.folder,
            context_report_sha256="c" * 64)["approval_sha256"], runner.file_digest(path))
        with self.assertRaisesRegex(runner.CampaignError, "context report"):
            runner.validate_approval(path, plan, self.folder)

    def test_resource_samples_require_live_rss_and_final_exited_state(self):
        journal = runner.Journal(self.folder / "supervisor.jsonl", synthetic=True)
        self.addCleanup(journal.close)
        guard = runner.CampaignGuard(runner.LIMITS)
        storage = {"root_bytes": 1234, "evidence_bytes": 321, "free_bytes": 10 * 1024**3}
        with patch.object(guard, "check_storage", return_value=storage) as check:
            for phase in ("initial", "periodic"):
                for rss in (None, 0, True, runner.LIMITS["max_rss_bytes"] + 1):
                    with self.subTest(phase=phase, rss=rss), self.assertRaises(runner.CampaignError):
                        runner.record_resource_sample(journal, guard, sample_phase=phase,
                            worker_state="running", current_rss_bytes=rss)
            runner.record_resource_sample(journal, guard, sample_phase="initial",
                worker_state="running", current_rss_bytes=1024)
            runner.record_resource_sample(journal, guard, sample_phase="final",
                worker_state="exited", current_rss_bytes=None)
            with self.assertRaises(runner.CampaignError):
                runner.record_resource_sample(journal, guard, sample_phase="final",
                    worker_state="running", current_rss_bytes=1024)
        self.assertEqual(check.call_count, 2)
        rows = self.rows("supervisor.jsonl")
        self.assertEqual([r["sample_phase"] for r in rows], ["initial", "final"])
        self.assertTrue(all(row["storage"] == storage for row in rows))

    def test_storage_measurement_tolerates_atomic_pending_file_rename(self):
        pending = self.folder / "control.pending"
        final = self.folder / "control.json"
        pending.write_bytes(b"durable-control")
        def observed_files(_):
            pending.replace(final)
            yield pending
            yield final
        with patch.object(Path, "rglob", side_effect=observed_files):
            self.assertEqual(runner.directory_bytes(self.folder), len(b"durable-control"))

    def test_initial_resource_ack_is_bound_to_claim_and_deadline(self):
        guard = runner.CampaignGuard(runner.LIMITS, output_dir=self.folder, clock=Clock())
        claim = {"worker_nonce": "unit-test-only"}
        ready = {"worker_nonce": claim["worker_nonce"], "deadline": guard.deadline,
                 "resource_sample_sequence": 2}
        path = self.folder / "supervisor-ready.json"
        with patch.dict(os.environ, {"KEV_CAMPAIGN_SUPERVISOR_PID": str(os.getppid())}):
            path.write_bytes(runner.wire(ready))
            runner.await_initial_resources(guard, claim)
            for key, wrong in (("worker_nonce", "other"), ("deadline", guard.deadline + 1),
                               ("resource_sample_sequence", 0)):
                path.write_bytes(runner.wire({**ready, key: wrong}))
                with self.subTest(key=key), self.assertRaisesRegex(runner.CampaignError, "acknowledgment"):
                    runner.await_initial_resources(guard, claim)

    def test_initial_resource_ack_wait_cannot_exceed_deadline(self):
        clock = Clock()
        guard = runner.CampaignGuard(runner.LIMITS, output_dir=self.folder, clock=clock)
        def expire(_):
            clock.now = guard.deadline + 1
        with patch.dict(os.environ, {"KEV_CAMPAIGN_SUPERVISOR_PID": str(os.getppid())}), \
                patch.object(runner.time, "sleep", side_effect=expire):
            with self.assertRaisesRegex(runner.CampaignError, "Whole-campaign deadline"):
                runner.await_initial_resources(guard, {"worker_nonce": "unit-test-only"})

    def test_imports_and_context_fixture_construction_load_no_framework(self):
        code = "from experiments.kev4b_campaign_v1 import runner, context_check; context_check.synthetic_fixtures(); import sys; assert not any(n == p or n.startswith(p+'.') for n in sys.modules for p in ('kev','torch','transformers','tokenizers','mlx','mlx_lm'))"
        result = subprocess.run([sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def context_report_fixture(self):
        """Real frozen source metadata; artificial token counts only, no runtime."""
        from experiments.kev4b_campaign_v1 import audit
        repo = Path(__file__).resolve().parents[1]
        fixture_repo = self.folder / "frozen-source"
        names = {str(path.relative_to(repo)) for path in (repo / runner.PREFIX).iterdir()
                 if path.suffix in {".py", ".json"} and path.name != "freeze.json"}
        names.update(runner.EXCLUDED_MANIFESTS)
        names.update({"clef_snake/game.py", "clef_snake/__init__.py", "experiments/__init__.py"})
        names.update("experiments/kev4b_v1/" + name for name in (
            "__init__.py", "adapter.py", "local_resolver.py", "preflight.py", "preflight-cases.json",
            "artifact-plan.json", "provenance.json", "storage-plan.json", "preflight-results-2026-10-06.json"))
        hashes = {}
        for name in sorted(names):
            target = fixture_repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((repo / name).read_bytes())
            hashes[name] = runner.file_digest(target)
        (fixture_repo / runner.PREFIX / "freeze.json").write_bytes(runner.wire({"files": hashes}))
        # Both production readers independently validate complete real metadata.
        # Reuse that result when the gate asks for the real module's repo path;
        # never create a freeze or tokenization report in the actual checkout.
        frozen = audit.read_frozen(fixture_repo)
        plan = runner.validate_bundle(fixture_repo)
        frozen_read = patch.object(audit, "read_frozen", return_value=frozen)
        frozen_read.start()
        self.addCleanup(frozen_read.stop)
        fixtures = audit.context_fixtures()
        rows = []
        for case in fixtures:
            record = audit.native_record(case["request"])
            rows.append({"case_id": case["case_id"], "body_length": case["body_length"],
                         "legacy_tail_food_duplicate": case["legacy_tail_food_duplicate"],
                         "request_sha256": audit.canonical_sha(case["request"]),
                         "state_tokens": 3, "branch_tokens": 14, "complete_row_tokens": 17,
                         "decision_position": 16, "option_positions": [6, 9, 12, 15],
                         "state_truncated": False, "would_require_internal_passes": 2,
                         "native_encoding_sha256": "e" * 64,
                         "native_record_sha256": audit.canonical_sha(record),
                         "rendered_state_utf8_bytes": len(record["state"].encode("utf-8"))})
        protocol = frozen["protocol"]
        report = {"schema_version": 1, "mode": "native_tokenizer_only", "status": "passed",
                  "case_count": 113, "utility_sha256": hashes[runner.PREFIX + "context_check.py"],
                  "fixtures_sha256": audit.canonical_sha(fixtures), "cases": rows,
                  "provenance": {"base_repository": "Qwen/Qwen3.5-4B-Base",
                      "base_revision": protocol["source_pins"]["Qwen/Qwen3.5-4B-Base"],
                      "tokenizer_files_sha256": audit.CONTEXT_TOKENIZER_HASHES,
                      "kev_code_revision": protocol["source_pins"]["upstream_code"],
                      "kev_files_sha256": {"kev/__init__.py": audit.digest(b""),
                          **{key: frozen["source_hashes"][key] for key in ("kev/api.py", "kev/model.py")}},
                      "game_source_sha256": protocol["game"]["source_sha256"]},
                  "versions": {key: protocol["runtime_identity"]["packages"].get(key, "synthetic-test-only")
                      for key in ("transformers", "tokenizers", "torch", "pydantic")},
                  "truncated_requests": 0, "model_loads": 0, "forward_calls": 0, "games": 0,
                  "guard_counters": {key: 0 for key in ("network_attempts", "weight_file_open_attempts",
                      "model_load_attempts", "forward_call_attempts")}, "limits": {"complete_row_tokens": 8192},
                  "maximum_complete_row_tokens": 17, "maximum_state_tokens": 3,
                  "maximum_projected_internal_passes": 2}
        path = self.folder / plan.protocol["context_validation"]["report_relative_path"]
        path.parent.mkdir(parents=True)
        return repo, plan, path, report

    def test_context_report_gate_requires_pinned_complete_clean_native_report(self):
        repo, plan, path, report = self.context_report_fixture()
        with self.assertRaisesRegex(runner.CampaignError, "missing"):
            runner.validate_context_report(repo, plan, self.folder)
        # Exercise structural validation using artificial tokens only. This file
        # is temporary, cannot authorize execution, and is never given to audit.
        path.write_bytes(runner.wire(report))
        with patch("experiments.kev4b_campaign_v1.context_check.tokenize_fixtures",
                   side_effect=AssertionError("Tokenizer path must not execute")):
            result = runner.validate_context_report(repo, plan, self.folder)
        self.assertEqual(result["sha256"], runner.file_digest(path))
        mutations = [("mode", "fixtures_only"), ("status", "not_executed"), ("synthetic", True),
                     ("case_count", 112), ("utility_sha256", "0" * 64),
                     ("fixtures_sha256", "0" * 64), ("provenance", {}),
                     ("games", 1), ("forward_calls", True), ("guard_counters", {}),
                     ("maximum_complete_row_tokens", 18), ("versions", {})]
        for key, value in mutations:
            path.write_bytes(runner.wire({**report, key: value}))
            with self.subTest(key=key), self.assertRaises(runner.CampaignError):
                runner.validate_context_report(repo, plan, self.folder)

    def test_context_report_rejects_truncation_overflow_and_misaligned_options(self):
        repo, plan, path, report = self.context_report_fixture()
        for key, wrong in (("state_truncated", True), ("complete_row_tokens", 8193),
                           ("option_positions", [6, 6, 12, 15]), ("state_tokens", True),
                           ("would_require_internal_passes", 1), ("request_sha256", "0" * 64),
                           ("native_record_sha256", "0" * 64), ("rendered_state_utf8_bytes", 1),
                           ("option_positions", [6, 9, 12, 14])):
            bad = deepcopy(report)
            bad["cases"][0][key] = wrong
            path.write_bytes(runner.wire(bad))
            with self.subTest(key=key), self.assertRaises(runner.CampaignError):
                runner.validate_context_report(repo, plan, self.folder)

    def test_bad_context_versions_or_record_hash_stop_before_claim_or_spawn(self):
        repo, plan, path, report = self.context_report_fixture()
        args = SimpleNamespace(repo=repo, root=self.folder,
            output=self.folder / "campaigns/kev4b_campaign_v1/unit-test-only",
            approval=self.folder / "approval.json", volume_uuid="unit-test-only")
        for field in ("versions", "native_record_sha256"):
            bad = deepcopy(report)
            if field == "versions":
                bad.pop("versions")
            else:
                bad["cases"][0][field] = "0" * 64
            path.write_bytes(runner.wire(bad))
            with self.subTest(field=field), patch.object(runner, "ASSET_ROOT", self.folder), \
                    patch.object(sys, "prefix", str(self.folder / "venv")), \
                    patch.object(runner.subprocess, "Popen", side_effect=AssertionError("No worker allowed")) as spawn:
                with self.assertRaisesRegex(runner.CampaignError, "Independent context report validation failed"):
                    runner.launch(args, plan)
                spawn.assert_not_called()
            self.assertFalse((self.folder / "campaign-claims").exists())
            self.assertFalse(args.output.exists())

    def test_guard_blocks_until_watchdog_ack_is_for_exact_call(self):
        clock = Clock()
        guard = runner.CampaignGuard(runner.LIMITS, output_dir=self.folder,
                                      execution_authorized=True, clock=clock)
        def acknowledge():
            self.assertIsNotNone(guard.active)
            (self.folder / "watchdog-ack.json").write_bytes(runner.wire(guard.active))
        with patch.object(guard, "_notify_supervisor", side_effect=acknowledge):
            guard.reserve({"call_id": 1})
        self.assertEqual(guard.calls, 1)
        self.assertEqual(runner.read_json(self.folder / "control.json")["active_decision"], guard.active)

    def test_guard_times_out_without_supervisor_ack(self):
        clock = Clock()
        guard = runner.CampaignGuard(runner.LIMITS, output_dir=self.folder,
                                      execution_authorized=True, clock=clock)
        def no_ack():
            clock.now += 121
        with patch.object(guard, "_notify_supervisor", side_effect=no_ack):
            with self.assertRaisesRegex(runner.CampaignError, "Per-decision"):
                guard.reserve({"call_id": 1})
        self.assertEqual(guard.calls, 1)

    def test_import_origin_check_refuses_another_checkout(self):
        with self.assertRaisesRegex(runner.CampaignError, "Executing runner"):
            runner.require_module_origins(self.folder)

    def frozen_fixture_bundle(self):
        """A separate synthetic source tree; files are hashed but never imported."""
        plan = synthetic_plan(seed_count=30, cap=500)
        files = {
            runner.PREFIX + name: b"# synthetic source; never executed\n"
            for name in ("runner.py", "native.py", "audit.py", "verify.py", "context_check.py", "__init__.py")
        }
        files.update({name: b"# synthetic source; never executed\n" for name in (
            "clef_snake/game.py", "clef_snake/__init__.py", "experiments/__init__.py",
            "experiments/kev4b_v1/__init__.py", "experiments/kev4b_v1/adapter.py",
            "experiments/kev4b_v1/local_resolver.py", "experiments/kev4b_v1/preflight.py")})
        files.update({"experiments/kev4b_v1/" + name: b"{}\n"
                      for name in ("artifact-plan.json", "provenance.json", "storage-plan.json")})
        previous = [f"{i:032x}" for i in range(10000, 10051)]
        for path, values in zip(runner.EXCLUDED_MANIFESTS, (previous[:15], previous[:15], previous[15:])):
            files[path] = runner.wire({"seeds": values})
        exclusions = [{"path": path, "sha256": runner.digest(files[path]),
                       "seed_count": 36 if "qwen3" in path else 15} for path in runner.EXCLUDED_MANIFESTS]
        files.update({runner.PREFIX + "protocol.json": runner.wire(plan.protocol),
                      runner.PREFIX + "analysis-plan.json": b"{}\n",
                      runner.PREFIX + "seed-manifest.json": runner.wire({"seeds": plan.seeds,
                          "excluded_manifests": exclusions, "excluded_distinct_seed_count": 51}),
                      runner.PREFIX + "schedule.json": runner.wire({"games": plan.games}),
                      "experiments/kev4b_v1/preflight-cases.json": runner.wire({"cases": plan.warmups})})
        for name, value in files.items():
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        hashes = {name: runner.digest(value) for name, value in files.items()}
        freeze_path = self.folder / runner.PREFIX / "freeze.json"
        freeze_path.write_bytes(runner.wire({"files": hashes}))
        return hashes, freeze_path

    def test_bundle_hash_tamper_is_rejected(self):
        hashes, freeze_path = self.frozen_fixture_bundle()
        valid = runner.validate_bundle(self.folder)
        self.assertEqual(len(valid.games), 90)
        target = self.folder / runner.PREFIX / "schedule.json"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(runner.CampaignError, "Frozen file changed"):
            runner.validate_bundle(self.folder)

    def test_invalid_schedule_still_rejected_when_hashes_match(self):
        hashes, freeze_path = self.frozen_fixture_bundle()
        relative = runner.PREFIX + "schedule.json"
        target = self.folder / relative
        document = runner.read_json(target)
        document["games"][1]["variant"] = document["games"][0]["variant"]
        target.write_bytes(runner.wire(document))
        hashes[relative] = runner.file_digest(target)
        freeze_path.write_bytes(runner.wire({"files": hashes}))
        with self.assertRaisesRegex(runner.CampaignError, "all three conditions"):
            runner.validate_bundle(self.folder)

    def test_duplicate_json_schema_keys_are_rejected(self):
        path = self.folder / "ambiguous.json"
        path.write_text('{"execution_authorized":false,"execution_authorized":true}')
        with self.assertRaisesRegex(runner.CampaignError, "Duplicate JSON"):
            runner.read_json(path)

    def test_equal_order_counts_do_not_allow_reordered_seed_blocks(self):
        hashes, freeze_path = self.frozen_fixture_bundle()
        relative = runner.PREFIX + "schedule.json"
        target = self.folder / relative
        document = runner.read_json(target)
        first_order = [game["variant"] for game in document["games"][:3]]
        second_order = [game["variant"] for game in document["games"][3:6]]
        for game, variant in zip(document["games"][:6], second_order + first_order):
            game["variant"] = variant
        target.write_bytes(runner.wire(document))
        hashes[relative] = runner.file_digest(target)
        freeze_path.write_bytes(runner.wire({"files": hashes}))
        with self.assertRaisesRegex(runner.CampaignError, "Prescribed permutation"):
            runner.validate_bundle(self.folder)

    def test_seed_overlap_refused_even_with_updated_matching_hashes(self):
        hashes, freeze_path = self.frozen_fixture_bundle()
        relative = runner.PREFIX + "seed-manifest.json"
        target = self.folder / relative
        document = runner.read_json(target)
        document["seeds"][0]["seed"] = runner.read_json(self.folder / runner.EXCLUDED_MANIFESTS[0])["seeds"][0]
        target.write_bytes(runner.wire(document))
        hashes[relative] = runner.file_digest(target)
        freeze_path.write_bytes(runner.wire({"files": hashes}))
        with self.assertRaisesRegex(runner.CampaignError, "overlap earlier"):
            runner.validate_bundle(self.folder)


if __name__ == "__main__":
    unittest.main()
