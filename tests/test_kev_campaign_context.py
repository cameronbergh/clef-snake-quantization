"""Offline synthetic-state and gate checks: no tokenizer/model import."""
from copy import deepcopy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from clef_snake.game import Game
from experiments.kev4b_campaign_v1 import context_check as check


class ContextFixtureTests(unittest.TestCase):
    def test_cycle_covers_board_and_closes(self):
        cycle = check.hamiltonian_cycle()
        self.assertEqual(len(cycle), 144)
        self.assertEqual(set(cycle), {(x, y) for x in range(12) for y in range(12)})
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            self.assertEqual(sum(abs(x-y) for x, y in zip(a, b)), 1)

    def test_deterministic_fixtures_have_valid_adjacency_and_food(self):
        cases = check.synthetic_fixtures()
        self.assertEqual(len(cases), 113)
        self.assertEqual(check.wire(cases), check.wire(check.synthetic_fixtures()))
        self.assertEqual(len({x['case_id'] for x in cases}), len(cases))
        self.assertEqual({x['body_length'] for x in cases}, set(check.LENGTHS) | {145})
        for case in cases:
            state = case['request']['state']
            body = [tuple(p) for p in state['body']]
            self.assertEqual(len(body), case['body_length'])
            self.assertEqual(state['head'], list(body[0]))
            self.assertEqual(list(case['request']['questions']['move']['criteria']), list(check.ACTIONS))
            for a, b in zip(body, body[1:]):
                self.assertEqual(sum(abs(x-y) for x, y in zip(a, b)), 1)
            if len(body) < 144:
                self.assertEqual(len(set(body)), len(body))
                self.assertNotIn(tuple(state['food']), set(body))
            elif len(body) == 144:
                self.assertEqual(len(set(body)), 144)
                self.assertEqual(state['food'], [0, 0])
            else:
                self.assertTrue(case['legacy_tail_food_duplicate'])
                self.assertEqual(body[0], (0, 0))
                self.assertEqual(body[-1], (0, 0))
                self.assertEqual(len(set(body)), 144)
                self.assertEqual(state['safe_moves'], [])

    def test_legacy_fixture_preserves_single_tail_food_transition(self):
        # One deterministic source-semantics assertion, not a gameplay loop.
        cycle = check.hamiltonian_cycle()
        game = Game.__new__(Game)
        game.body = [list(p) for p in cycle[1:] + cycle[:1]]
        game.direction, game.score, game.steps, game.alive = 'left', 140, 498, True
        game.seed, game.food_events = 'synthetic-context-test-only', []
        game.food = game.spawn()
        self.assertEqual(game.food, [0, 0])
        self.assertEqual(game.food_events[-1]['rank'], -1)
        game.apply('left')
        expected = deepcopy(check.synthetic_fixtures()[-1]['request'])
        actual = game.request()
        actual['model'] = 'kev-latest'
        self.assertEqual(actual, expected)
        self.assertEqual((game.score, game.steps, len(game.body)), (141, 499, 145))

    def test_offline_gate_requires_all_flags_and_no_bytecode(self):
        check.verify_environment(dict(check.OFFLINE_ENV), bytecode_disabled=True)
        for name in check.OFFLINE_ENV:
            env = dict(check.OFFLINE_ENV)
            env.pop(name)
            with self.assertRaises(ValueError):
                check.verify_environment(env, bytecode_disabled=True)
        with self.assertRaises(ValueError):
            check.verify_environment(dict(check.OFFLINE_ENV), bytecode_disabled=False)

    def test_default_cli_imports_no_model_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixtures.json'
            code = "from experiments.kev4b_campaign_v1.context_check import main; import sys; main(['--report',sys.argv[1]]); assert not any(n == p or n.startswith(p+'.') for n in sys.modules for p in ('kev','torch','transformers','tokenizers','mlx'))"
            result = subprocess.run([sys.executable, '-B', '-c', code, str(path)],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(path.read_text())
            self.assertEqual(report['mode'], 'fixtures_only')
            self.assertEqual(report['case_count'], 113)
            self.assertFalse(report['tokenizer_loaded'])
            with self.assertRaises(FileExistsError):
                check.main(['--report', str(path)])

    def test_explicit_tokenization_paths_are_required_before_loading(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            check.main(['--tokenize'])

    def test_guard_refuses_network_and_weight_file_operations(self):
        # Audit events only: no socket or model file is actually opened.
        with check.tokenizer_only_guard() as counts:
            with self.assertRaises(RuntimeError):
                sys.audit('socket.connect', None, ('example.invalid', 443))
            with self.assertRaises(RuntimeError):
                sys.audit('open', '/tmp/not-a-real-weight.safetensors', 'r', 0)
        self.assertEqual(counts['network_attempts'], 1)
        self.assertEqual(counts['weight_file_open_attempts'], 1)

    def test_guard_refuses_model_constructor_before_its_body(self):
        # A tiny fake definition exercises the guard without importing Kev.
        namespace = {'__name__': 'kev.model'}
        exec('class DecisionModel:\n def __init__(self):\n  raise AssertionError("body must not execute")', namespace)
        with check.tokenizer_only_guard() as counts:
            with self.assertRaises(RuntimeError):
                namespace['DecisionModel']()
        self.assertEqual(counts['model_load_attempts'], 1)

    def test_encoding_summary_checks_whole_row_and_markers(self):
        case = check.synthetic_fixtures()[0]
        markers = [100, 101, 102, 103, 104]
        ids = [100, 7, 8, 101, 102, 9, 103, 102, 10, 103, 102, 11, 103, 102, 12, 103, 104]
        enc = {'ids': ids, 'state_truncated': False, 'option_isolation': False,
               'state_tokens': 3, 'opt_idx': [[6, 9, 12, 15]], 'decide_idx': [16]}
        meta = [{'keys': list(check.ACTIONS)}]
        result = check.summarize_encoding(case, enc, ids[:3], [{'ids': ids[3:]}], meta, markers)
        self.assertEqual(result['complete_row_tokens'], 17)
        self.assertEqual(result['option_positions'], [6, 9, 12, 15])
        for field, value in [('state_truncated', True), ('state_tokens', 4)]:
            invalid = deepcopy(enc)
            invalid[field] = value
            with self.assertRaises(ValueError):
                check.summarize_encoding(case, invalid, ids[:3], [{'ids': ids[3:]}], meta, markers)
        invalid = deepcopy(enc)
        invalid['ids'][6] = 999
        with self.assertRaises(ValueError):
            check.summarize_encoding(case, invalid, ids[:3], [{'ids': ids[3:]}], meta, markers)


if __name__ == '__main__':
    unittest.main()
