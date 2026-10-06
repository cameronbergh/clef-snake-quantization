"""Independent offline audit regressions. All campaign traces are explicit fakes.

No production runner, adapter, Game, native helper, tokenizer or model is imported.
Synthetic forward records below are test fixtures, never research observations.
"""
import ast
import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.kev4b_campaign_v1 import audit as a

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'experiments/kev4b_campaign_v1'


def frozen_fixture():
    protocol = json.loads((PREP / 'protocol.json').read_text())
    schedule = json.loads((PREP / 'schedule.json').read_text())['games']
    cases = json.loads((ROOT / protocol['warmups']['fixture_path']).read_text())['cases'][:2]
    files = {protocol['warmups']['fixture_path']: '1' * 64,
             'experiments/kev4b_campaign_v1/context_check.py': a.digest((PREP / 'context_check.py').read_bytes())}
    warmups = {}
    for c in cases:
        request = deepcopy(c['request']); request['model'] = 'kev-latest'
        warmups[c['id']] = a.wire(request)
    return {'protocol': protocol, 'schedule': schedule, 'warmups': warmups,
            'freeze': {'files': files}, 'freeze_sha256': 'a' * 64,
            'source_hashes': {r['path']: r['sha256'] for r in json.loads(
                (ROOT / 'experiments/kev4b_v1/provenance.json').read_bytes())['upstream_code']['files']},
            'acquisition_sha256': '3' * 64}


def context_report_fixture(frozen):
    """Explicitly fabricated counts: tests receipt validation, never tokenization."""
    cases = a.context_fixtures(); p = frozen['protocol']; rows = []
    for case in cases:
        record = a.native_record(case['request'])
        rows.append({'case_id': case['case_id'], 'body_length': case['body_length'],
                     'legacy_tail_food_duplicate': case['legacy_tail_food_duplicate'],
                     'request_sha256': a.canonical_sha(case['request']),
                     'native_encoding_sha256': '0' * 64, 'state_tokens': 100, 'branch_tokens': 10,
                     'complete_row_tokens': 110, 'option_positions': [102, 104, 106, 108],
                     'decision_position': 109, 'state_truncated': False, 'would_require_internal_passes': 2,
                     'native_record_sha256': a.canonical_sha(record),
                     'rendered_state_utf8_bytes': len(record['state'].encode('utf-8'))})
    return {'schema_version': 1, 'mode': 'native_tokenizer_only', 'status': 'passed',
            'utility_sha256': frozen['freeze']['files']['experiments/kev4b_campaign_v1/context_check.py'],
            'fixtures_sha256': a.canonical_sha(cases), 'case_count': 113, 'cases': rows,
            'provenance': {'base_repository': 'Qwen/Qwen3.5-4B-Base',
                           'base_revision': p['source_pins']['Qwen/Qwen3.5-4B-Base'],
                           'tokenizer_files_sha256': a.CONTEXT_TOKENIZER_HASHES,
                           'kev_code_revision': p['source_pins']['upstream_code'],
                           'kev_files_sha256': {'kev/__init__.py': a.digest(b''),
                                               **{k: frozen['source_hashes'][k] for k in ('kev/api.py', 'kev/model.py')}},
                           'game_source_sha256': p['game']['source_sha256']},
            'versions': {k: p['runtime_identity']['packages'].get(k, 'synthetic-test-only')
                         for k in ('transformers', 'tokenizers', 'torch', 'pydantic')},
            'maximum_complete_row_tokens': 110, 'maximum_state_tokens': 100, 'maximum_projected_internal_passes': 2,
            'truncated_requests': 0, 'model_loads': 0, 'forward_calls': 0, 'games': 0,
            'guard_counters': {k: 0 for k in ('network_attempts', 'weight_file_open_attempts',
                                             'model_load_attempts', 'forward_call_attempts')},
            'limits': {'complete_row_tokens': 8192}}


def inventories():
    head = json.loads((ROOT / 'experiments/kev4b_v1/preflight-results-2026-10-06.json').read_text())['head']['inventory']
    linear = {f'm{i}': {'shape': [1, 64], 'dtype': 'mlx.core.bfloat16'} for i in range(247)}
    linear['m247'] = {'shape': [(3569090560 - 247 * 64) // 64, 64], 'dtype': 'mlx.core.bfloat16'}
    def tensor(shape, dtype='mlx.core.bfloat16'):
        return {'dtype': dtype, 'shape': shape, 'elements': math.prod(shape), 'canonical_value_sha256': '4' * 64}
    tensors = {n + '.weight': tensor(v['shape']) for n, v in linear.items()}
    tensors.update({f'fixed{i}': tensor([1]) for i in range(178)})
    before = {'tensors': tensors, 'linears': linear, 'quantized': {}, 'head': head}
    variants = {'bf16': before}
    for variant, bits in zip(a.VARIANTS[1:], (8, 4)):
        after = deepcopy(before); after['linears'] = {}
        for name, module in linear.items():
            out, width = module['shape']
            after['quantized'][name] = {'bits': bits, 'group_size': 64, 'mode': 'affine', 'class': 'QuantizedLinear'}
            after['tensors'][name + '.weight'] = tensor([out, width * bits // 32], 'mlx.core.uint32')
            for suffix in ('.scales', '.biases'):
                after['tensors'][name + suffix] = tensor([out, width // 64])
        variants[variant] = after
    return before, variants


def evidence_fixture(games=90):
    """Handwritten synthetic two-warmup/one-neck-collision episode protocol."""
    frozen = frozen_fixture(); before, variants = inventories()
    events, seq, tick = [], {'runner': 0, 'native': 0}, 100.0
    def emit(source, event, **fields):
        nonlocal tick
        tick += .001; seq[source] += 1
        row = {'sequence': seq[source], 'event': event, 'synthetic': True,
               'monotonic': tick, 'unix_time': 0, **fields}
        events.append((source, row)); return row
    def bytes_fields(prefix, value):
        return {prefix + '_base64': base64.b64encode(value).decode(), prefix + '_sha256': a.digest(value)}
    p = frozen['protocol']; pins = {k: v for k, v in p['source_pins'].items() if k != 'upstream_code'}
    identity = {'pins': pins, 'code_revision': p['source_pins']['upstream_code'], 'whole_source_git_clean': True,
                'source_hashes': frozen['source_hashes'], 'versions': p['runtime_identity']['packages'],
                'fixture_sha256': '1' * 64, 'acquisition_sha256': '3' * 64,
                'verified_files': [{'repo_id': 'synthetic', 'path': str(i), 'bytes': 1, 'sha256': 'f' * 64} for i in range(29)]}
    emit('runner', 'run_start', campaign_id='kev4b_campaign_v1', freeze_sha256=frozen['freeze_sha256'],
         input_hashes=frozen['freeze']['files'], limits=p['limits'], planned_games=90, retries=0, resume=False, approval=None)
    emit('native', 'inputs_verified', **identity)
    emit('runner', 'inputs_verified', metadata=identity)
    calls = 0
    for g in frozen['schedule'][:games]:
        game = a.ReferenceBoard(g['seed']); game_context = {k: g[k] for k in ('game_id', 'variant')}
        emit('runner', 'game_start', game_sequence=g['sequence'], **{k: v for k, v in g.items() if k != 'sequence'},
             initial_snapshot=game.snapshot(), food_events=deepcopy(game.food_events))
        emit('runner', 'load_start', **game_context)
        emit('native', 'inputs_verified', **identity)
        if calls == 0:
            emit('native', 'runtime', versions=p['runtime_identity']['packages'], python=p['runtime_identity']['python'],
                 platform='Darwin-arm64', network_allowed=False)
        emit('native', 'load_start', **game_context)
        emit('native', 'checkpoint_metadata', **game_context, exact_native_temperature=a.HEAD_TEMPERATURE,
             head_file_sha256=a.HEAD_FILE_SHA256)
        cache = p['storage']['asset_root'] + '/hf/hub'
        resolver = {'kind': 'pinned_local_snapshot_resolution_v1', 'cache_root': cache,
                    'pinned_base_identifier': 'Qwen/Qwen3.5-4B-Base@' + pins['Qwen/Qwen3.5-4B-Base'],
                    'snapshots': {key: {'revision': val, 'path': cache + '/models--' + key.replace('/', '--') + '/snapshots/' + val}
                                  for key, val in pins.items()}, 'network_calls': 0, 'unknown_input_policy': 'refuse_without_fallback'}
        emit('native', 'local_snapshot_resolution', **game_context, resolver=resolver)
        emit('native', 'native_inventory', **game_context, inventory=before, inventory_sha256=a.canonical_sha(before))
        after = variants[g['variant']]
        if g['variant'] != 'bf16':
            bits = 8 if g['variant'] == a.VARIANTS[1] else 4
            emit('native', 'quantize_start', **game_context, bits=bits, group_size=64, mode='affine', quantize_input=False)
            emit('native', 'quantized_inventory', **game_context, inventory=after, inventory_sha256=a.canonical_sha(after),
                 derivation='fresh_native_merge_then_in_memory_quantization', persisted_variant=False, export_reload_validated=False)
        loaded = {**game_context, 'inventory_sha256': a.canonical_sha(after), 'head': after['head'],
                  'retained_backbone_tensors': 178, 'export_reload_validated': False}
        emit('native', 'load_complete', **loaded); emit('runner', 'load_complete', **game_context, metadata=loaded)
        for phase, index in [('warmup', 1), ('warmup', 2), ('game', 1)]:
            calls += 1
            case = p['warmups']['case_ids'][index - 1] if phase == 'warmup' else None
            context = {'call_id': calls, 'game_id': g['game_id'], 'phase': phase,
                       'game_attempt': 1 if phase == 'game' else None,
                       'warmup_index': index if phase == 'warmup' else None, 'fixture_id': case}
            request = frozen['warmups'][case] if case else a.wire(game.request())
            emit('runner', 'call_reserved', **context, **bytes_fields('request', request))
            native_context = {**context, 'decision_id': calls, 'variant': g['variant']}
            emit('native', 'decision_start', **native_context, **bytes_fields('request', request))
            enc = {'ids': list(range(11)), 'seg': [0] + [1]*10, 'pos': list(range(11)),
                   'opt': [-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, -2], 'option_isolation': False,
                   'decide_idx': [10], 'opt_idx': [[3, 5, 7, 9]], 'labels': [0], 'state_tokens': 1, 'state_truncated': False}
            emit('native', 'encoded', **native_context, native_record=a.native_record(json.loads(request)),
                 native_meta=[{'id': 'move', 'type': 'choice', 'keys': list(a.ACTIONS)}], encoding=enc, complete_row_tokens=11)
            emit('native', 'native_forward_start', **native_context, expected_backbone_passes=2)
            passes = []
            for j, length in enumerate((1, 10), 1):
                item = {'pass_number': j, 'input_shape': [1, length], 'cache_supplied': True,
                        'native_stage': 'state_prefix' if j == 1 else 'question_branch'}
                passes.append(item)
                emit('native', 'backbone_pass_start', **native_context, **item)
                emit('native', 'backbone_pass_result', **native_context, **item, outcome='returned')
            probabilities = [.25001, .25, .25002, .24997]  # All displayed probabilities tie; LEFT wins raw.
            emit('native', 'raw_decision', **native_context, logits=[[math.log(x) for x in probabilities]],
                 probabilities=[probabilities], selected_index=2, choice='left', internal_backbone_passes=passes)
            response = a.wire({'model': 'kev-latest', 'answers': {'move': {'type': 'choice', 'choice': 'left',
                              'confidence': round((max(probabilities)/sum(probabilities)-.25)/.75, 4),
                              'probabilities': dict(zip(a.ACTIONS, [.25]*4))}}, 'truncated': False,
                              'usage': {'input_tokens': 11, 'output_tokens': 10, 'state_tokens': 1, 'state_tokens_used': 1}})
            terminal = emit('native', 'native_response', **native_context, status=200, choice='left', **bytes_fields('response', response))
            attempt = {'endpoint_path': '/v1/systemone', **bytes_fields('request', request), 'response_status': 200,
                       **bytes_fields('response', response), 'choice': 'left', 'error': None}
            emit('runner', 'attempt', **context, attempt=attempt, native_response_sequence=terminal['sequence'])
            if phase == 'game':
                before_board = game.snapshot(); game.apply('left')
                emit('runner', 'transition', **context, move='left', before=before_board, after=game.snapshot(),
                     new_food_events=[], food_events=deepcopy(game.food_events))
        emit('runner', 'unload_start', **game_context)
        unloaded = {**game_context, 'inventory': after, 'inventory_sha256': a.canonical_sha(after),
                    'head_after': after['head'], 'unchanged': True}
        emit('native', 'unloaded', **unloaded); emit('runner', 'unload_complete', **game_context, metadata=unloaded)
        emit('runner', 'game_complete', **game_context, seed_round=g['seed_round'], seed=g['seed'],
             score=0, steps=0, attempts=1, alive=False, end_reason='collision', censored=False,
             final_snapshot=game.snapshot(), food_events=deepcopy(game.food_events))
    if games == 90:
        emit('runner', 'run_complete', games_completed=90, decisions_reserved=calls, warmups=180)
    return frozen, events


def run(events, frozen, allow=True):
    engine = a.CampaignAudit(frozen, allow_synthetic=allow)
    for source, row in events:
        engine.feed(source, row)
    return engine.result()


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen, cls.events = evidence_fixture(90)

    def test_complete_fake_is_explicitly_synthetic_never_real_pass(self):
        report = run(self.events, self.frozen)
        self.assertTrue(report['complete']); self.assertTrue(report['evidence_valid'])
        self.assertFalse(report['passed']); self.assertFalse(report['real'])
        self.assertEqual(report['status'], 'synthetic_complete')
        self.assertEqual(report['counts']['completed_games'], 90)
        self.assertEqual(report['counts']['reserved_decisions'], 270)
        self.assertEqual(report['counts']['warmup_reservations'], 180)
        self.assertEqual(report['counts']['internal_backbone_passes'], 540)
        with self.assertRaisesRegex(a.EvidenceError, 'Synthetic'):
            run(self.events, self.frozen, allow=False)

    def test_partial_does_not_emit_scores(self):
        frozen, events = evidence_fixture(1)
        report = run(events, frozen)
        self.assertFalse(report['passed']); self.assertFalse(report['complete'])
        self.assertEqual(report['status'], 'incomplete'); self.assertNotIn('games', report)
        self.assertEqual(report['counts']['completed_games'], 1)

    def mutate(self, name, fn, source=None):
        events = list(self.events)
        index = next(i for i, (s, r) in enumerate(events) if r['event'] == name and (source is None or s == source))
        row = deepcopy(events[index][1]); fn(row); events[index] = (events[index][0], row)
        return events

    def test_request_field_order_and_fact_tampering_rejected(self):
        for mutation in ('order', 'feature'):
            def change(r):
                request = json.loads(base64.b64decode(r['request_base64']))
                if mutation == 'order': request = dict(reversed(list(request.items())))
                else: request['state']['move_analysis']['up']['open_space'] += 1
                raw = a.wire(request); r.update(request_base64=base64.b64encode(raw).decode(), request_sha256=a.digest(raw))
            with self.assertRaises(a.EvidenceError):
                run(self.mutate('call_reserved', change), self.frozen)

    def test_native_rounding_tie_must_keep_unrounded_argmax(self):
        with self.assertRaisesRegex(a.EvidenceError, 'argmax'):
            run(self.mutate('raw_decision', lambda r: r.update(choice='up', selected_index=0)), self.frozen)

    def test_transition_without_durable_attempt_rejected(self):
        index = next(i for i, (_, r) in enumerate(self.events) if r['event'] == 'transition')
        events = list(self.events); events[index - 1], events[index] = events[index], events[index - 1]
        with self.assertRaises(a.EvidenceError): run(events, self.frozen)

    def test_extra_forward_retry_and_internal_pass_rejected(self):
        for event in ('native_forward_start', 'backbone_pass_start'):
            index = next(i for i, (_, r) in enumerate(self.events) if r['event'] == event)
            events = self.events[:index+1] + [self.events[index]] + self.events[index+1:]
            with self.assertRaises(a.EvidenceError): run(events, self.frozen)

    def test_collision_not_success_and_mutated_food_rejected(self):
        for change in (lambda r: r['after'].update(steps=1), lambda r: r['food_events'][0].update(rank=999)):
            with self.assertRaises(a.EvidenceError): run(self.mutate('transition', change), self.frozen)

    def test_wrong_native_response_bytes_and_sequence_rejected(self):
        for change in (lambda r: r['attempt'].update(response_sha256='0'*64),
                       lambda r: r.update(native_response_sequence=0)):
            with self.assertRaises(a.EvidenceError): run(self.mutate('attempt', change), self.frozen)

    def test_truncated_tokens_and_changed_readout_order_rejected(self):
        for change in (lambda r: r['encoding'].update(state_truncated=True),
                       lambda r: r['encoding'].update(opt_idx=[[5,3,7,9]])):
            with self.assertRaises(a.EvidenceError): run(self.mutate('encoded', change), self.frozen)

    def test_retained_head_and_packed_scope_mutations_rejected(self):
        before, variants = inventories()
        for variant in a.VARIANTS:
            a.verify_inventory(before, variants[variant], variant, synthetic=True)
        bad = deepcopy(variants[a.VARIANTS[1]]); bad['tensors']['fixed0']['canonical_value_sha256'] = '0'*64
        with self.assertRaises(a.EvidenceError): a.verify_inventory(before, bad, a.VARIANTS[1], synthetic=True)
        bad = deepcopy(variants[a.VARIANTS[1]]); bad['head']['temperature'] = 2.41
        with self.assertRaises(a.EvidenceError): a.verify_inventory(before, bad, a.VARIANTS[1], synthetic=True)
        bad = deepcopy(variants[a.VARIANTS[1]]); del bad['tensors']['m0.scales']
        with self.assertRaises(a.EvidenceError): a.verify_inventory(before, bad, a.VARIANTS[1], synthetic=True)
        with self.assertRaises(a.EvidenceError): a.verify_inventory(before, before, 'bf16')

    def test_torn_duplicate_key_nonfinite_and_noncontiguous_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runner.jsonl'
            for raw in (b'{"sequence":1}', b'{"sequence":1,"sequence":1,"event":"x","monotonic":1}\n',
                        b'{"sequence":1,"event":"x","monotonic":NaN}\n',
                        b'{"sequence":2,"event":"x","monotonic":1}\n'):
                path.write_bytes(raw)
                with self.assertRaises(a.EvidenceError): list(a.journal(path, 'runner'))

    def test_independent_reference_matches_saved_native_request_fixtures(self):
        cases = json.loads((ROOT / 'experiments/kev4b_v1/preflight-cases.json').read_text())['cases']
        for case in cases:
            request = deepcopy(case['request']); request['model'] = 'kev-latest'
            state = request['state']; board = a.ReferenceBoard('fixture')
            board.body = [tuple(p) for p in state['body']]; board.food = tuple(state['food'])
            board.direction = state['direction']
            self.assertEqual(a.wire(board.request()), a.wire(request), case['id'])

    def test_food_seed_and_tail_vacating_facts(self):
        board = a.ReferenceBoard('b145b41017ce4bce48bf75df6719f853')
        self.assertEqual(board.food, (0,3))
        self.assertEqual(board.request()['state']['move_analysis']['left']['reason'], 'neck')
        board.body = [(1,1),(1,2),(0,2),(0,1)]; board.food = (9,9)
        board.apply('left'); self.assertTrue(board.alive); self.assertEqual(board.body[0], (0,1))
        self.assertEqual(a.ReferenceBoard.open_space((0,0), {(1,0),(0,1)}), 1)

    def test_full_board_fallback_and_tail_eating_preserved(self):
        board = a.ReferenceBoard('full-board-fixture')
        board.body = [(x,y) for y in range(12) for x in range(12)]
        board._food(); self.assertEqual(board.food, (0,0)); self.assertEqual(board.food_events[-1]['rank'], -1)
        # Full board, adjacent vacating tail is also the legacy fallback food.
        board.body = [(1,0)] + [p for p in board.body if p not in ((1,0),(0,0))] + [(0,0)]
        board.apply('left')
        self.assertTrue(board.alive); self.assertEqual(len(board.body),145)
        self.assertEqual(board.score,1); self.assertEqual(board.steps,1)
        self.assertEqual(board.food,(0,0)); self.assertEqual(board.food_events[-1]['rank'],-1)
        board.apply('up'); self.assertFalse(board.alive); self.assertEqual(board.steps,1)

    def test_auditor_has_no_production_or_model_imports(self):
        tree = ast.parse((PREP / 'audit.py').read_text())
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imported += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any(x and (x.startswith(('clef_snake','experiments','torch','mlx','transformers','kev')))
                             for x in imported))

    def test_technical_forward_failure_is_accounted_and_never_a_score(self):
        end = next(i for i, (_, row) in enumerate(self.events) if row['event'] == 'native_forward_start')
        events = self.events[:end + 1]
        first_attempt = next(row for _, row in self.events if row['event'] == 'attempt')
        native_context = {key: self.events[end][1][key] for key in (*a.CONTEXT, 'decision_id', 'variant')}
        now = self.events[end][1]['monotonic']
        metadata = {'synthetic': True, 'sequence': 999, 'monotonic': now + .001, 'unix_time': 0}
        events.append(('native', {**metadata, 'event': 'decision_error', **native_context,
                                 'response_base64': None, 'response_sha256': None,
                                 'error_type': 'SyntheticFailure', 'message': 'Test only'}))
        failed = deepcopy(first_attempt); failed.update(monotonic=now + .002, native_response_sequence=None)
        failed['attempt'].update(response_status=None, response_base64=None, response_sha256=None,
                                 choice=None, error={'stage': 'transport', 'type': 'SyntheticFailure', 'message': 'Test only'})
        events.append(('runner', failed))
        events.append(('runner', {**metadata, 'monotonic': now + .003, 'event': 'run_error'}))
        report = run(events, self.frozen)
        self.assertEqual(report['status'], 'technical_failure')
        self.assertEqual(report['counts']['reserved_decisions'], 1)
        self.assertEqual(report['counts']['native_forwards'], 1)
        self.assertEqual(report['counts']['failed_attempts'], 1)
        self.assertNotIn('games', report)
        self.assertFalse(report['passed'])
        events.append(next(item for item in self.events if item[1]['event'] == 'call_reserved'))
        with self.assertRaises(a.EvidenceError): run(events, self.frozen)

    def test_attempt_at_cap_is_forbidden_but_cap_is_alive(self):
        board = a.ReferenceBoard('synthetic-cap')
        board.steps = 499; board.food = (11,11)
        board.apply('up')
        self.assertEqual(board.steps, 500); self.assertTrue(board.alive)
        with self.assertRaises(a.EvidenceError): board.apply('up')

    def test_supervisor_requires_matching_approval_claim_and_successful_exit(self):
        from types import SimpleNamespace
        context_raw = a.wire(context_report_fixture(self.frozen))
        approval = {'schema_version': 1, 'campaign_id': 'kev4b_campaign_v1',
                    'execution_authorized': True, 'freeze_sha256': self.frozen['freeze_sha256'],
                    'limits': self.frozen['protocol']['limits'], 'one_use_id': 'test-only-fixture',
                    'context_report_sha256': a.digest(context_raw)}
        raw = a.wire(approval)
        evidence = {'approval_sha256': a.digest(raw), 'one_use_id': approval['one_use_id'],
                    'freeze_sha256': approval['freeze_sha256'], 'limits': approval['limits'],
                    'context_report_sha256': a.digest(context_raw)}
        claim = {'schema_version': 1, 'approval': evidence, 'deadline': 200}
        claim_raw = a.wire(claim)
        engine = SimpleNamespace(frozen=self.frozen, approval={**evidence, 'claim_sha256': a.digest(claim_raw)},
                                 start_time=110, first_native_load_time=112, end_time=180, finished=True)
        rows = [{'sequence': 1, 'synthetic': False, 'monotonic': 100, 'event': 'supervisor_start',
                 'freeze_sha256': approval['freeze_sha256'], 'approval': evidence,
                 'claim_sha256': a.digest(claim_raw), 'deadline': 200, 'limits': approval['limits']},
                {'sequence': 2, 'synthetic': False, 'monotonic': 111, 'event': 'resource_sample',
                 'sample_phase': 'initial', 'worker_state': 'running', 'current_rss_bytes': 1024,
                 'storage': {'free_bytes': 8589934592, 'root_bytes': 1000, 'evidence_bytes': 100}},
                {'sequence': 3, 'synthetic': False, 'monotonic': 181, 'event': 'resource_sample',
                 'sample_phase': 'final', 'worker_state': 'exited', 'current_rss_bytes': None,
                 'storage': {'free_bytes': 8589934592, 'root_bytes': 2000, 'evidence_bytes': 200}},
                {'sequence': 4, 'synthetic': False, 'monotonic': 182, 'event': 'worker_exit', 'returncode': 0}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'approval.json').write_bytes(raw); (root / 'claim.json').write_bytes(claim_raw)
            (root / 'context-report.json').write_bytes(context_raw)
            def save(): (root / 'supervisor.jsonl').write_bytes(b''.join(a.wire(r) + b'\n' for r in rows))
            save(); a.verify_supervisor(root, engine)
            valid = deepcopy(rows)
            # Keep sequence numbers contiguous after each mutation: these fail
            # resource completeness checks, not merely JSONL ordering checks.
            cases = [
                [valid[0], valid[-1]],
                [valid[0], valid[1], valid[-1]],
                [valid[0], valid[2], valid[-1]],
            ]
            for broken in cases:
                rows[:] = deepcopy(broken)
                for i, row in enumerate(rows, 1): row['sequence'] = i
                save()
                with self.assertRaises(a.EvidenceError): a.verify_supervisor(root, engine)
            changes = [
                (1, lambda r: r.update(current_rss_bytes=None)),
                (1, lambda r: r.update(current_rss_bytes=0)),
                (1, lambda r: r.update(current_rss_bytes=True)),
                (1, lambda r: r.update(current_rss_bytes=51539607553)),
                (1, lambda r: r.update(monotonic=113)),
                (1, lambda r: r.pop('storage')),
                (1, lambda r: r['storage'].pop('free_bytes')),
                (1, lambda r: r['storage'].update(free_bytes=8589934591)),
                (1, lambda r: r['storage'].update(root_bytes=60129542145)),
                (2, lambda r: r['storage'].update(evidence_bytes=17179869185)),
                (2, lambda r: r.update(monotonic=179)),
                (2, lambda r: r.update(current_rss_bytes=1)),
                (2, lambda r: r.update(worker_state='running')),
            ]
            for index, change in changes:
                rows[:] = deepcopy(valid); change(rows[index]); save()
                with self.assertRaises(a.EvidenceError): a.verify_supervisor(root, engine)
            rows[:] = deepcopy(valid)
            rows[-1]['returncode'] = 1; save()
            with self.assertRaises(a.EvidenceError): a.verify_supervisor(root, engine)
            rows[-1]['returncode'] = 0; save()
            (root / 'context-report.json').write_bytes(context_raw + b'\n')
            with self.assertRaisesRegex(a.EvidenceError, 'context report hash'): a.verify_supervisor(root, engine)
            (root / 'context-report.json').write_bytes(context_raw)
            (root / 'approval.json').write_bytes(a.wire({**approval, 'execution_authorized': False}))
            with self.assertRaises(a.EvidenceError): a.verify_supervisor(root, engine)

    def test_context_report_requires_complete_pinned_untruncated_zero_activity_evidence(self):
        report = context_report_fixture(self.frozen)
        a.verify_context_report(report, self.frozen)
        cases = a.context_fixtures()
        self.assertEqual(len(cases), 113)
        self.assertEqual(a.canonical_sha(cases), '3b396a5614e19bed464920cba99d581d32c48d45defe94b0ae568c5fdafe877d')
        self.assertEqual(cases[-1]['body_length'], 145)
        self.assertTrue(cases[-1]['legacy_tail_food_duplicate'])
        self.assertEqual(cases[-1]['request']['state']['safe_moves'], [])
        mutations = [
            lambda r: r.update(mode='fixtures_only'),
            lambda r: r.update(status='pending'),
            lambda r: r.update(synthetic=True),
            lambda r: r.update(synthetic=0),
            lambda r: r.update(case_count=112),
            lambda r: r.update(utility_sha256='0' * 64),
            lambda r: r.update(fixtures_sha256='0' * 64),
            lambda r: r['cases'].pop(),
            lambda r: r['cases'].reverse(),
            lambda r: r['provenance'].update(base_revision='0' * 40),
            lambda r: r['versions'].update(transformers='changed'),
            lambda r: r.update(model_loads=1),
            lambda r: r.update(forward_calls=1),
            lambda r: r.update(games=1),
            lambda r: r.update(truncated_requests=False),
            lambda r: r['guard_counters'].pop('weight_file_open_attempts'),
            lambda r: r['guard_counters'].update(network_attempts=1),
            lambda r: r['cases'][0].update(request_sha256='0' * 64),
            lambda r: r['cases'][0].update(complete_row_tokens=8193),
            lambda r: r['cases'][0].update(state_tokens=True),
            lambda r: r['cases'][0].update(state_truncated=True),
            lambda r: r['cases'][0].update(option_positions=[102, 104, 104, 108]),
            lambda r: r['cases'][0].update(decision_position=108),
            lambda r: r['cases'][0].update(would_require_internal_passes=3),
            lambda r: r['cases'][0].update(native_record_sha256='0' * 64),
            lambda r: r['cases'][0].update(rendered_state_utf8_bytes=0),
            lambda r: r['cases'][0].update(native_encoding_sha256='invalid'),
            lambda r: r.update(maximum_complete_row_tokens=111),
        ]
        for i, mutate in enumerate(mutations):
            changed = deepcopy(report); mutate(changed)
            with self.subTest(mutation=i), self.assertRaises(a.EvidenceError):
                a.verify_context_report(changed, self.frozen)

    def test_real_native_events_require_complete_memory_measurements(self):
        engine = a.CampaignAudit(self.frozen)
        engine.synthetic = False
        memory = {'rss_peak_bytes': 1000, 'mlx_active_bytes': 300,
                  'mlx_peak_bytes': 400, 'mlx_cache_bytes': 500}
        for event in ('runtime', 'load_start', 'native_inventory', 'quantized_inventory',
                      'load_complete', 'raw_decision', 'unloaded'):
            with self.subTest(event=event):
                engine.memory({'event': event, 'memory': memory}, 'native')
                with self.assertRaisesRegex(a.EvidenceError, 'mandatory memory'):
                    engine.memory({'event': event}, 'native')
                for key in memory:
                    missing = dict(memory); del missing[key]
                    with self.assertRaisesRegex(a.EvidenceError, 'mandatory memory'):
                        engine.memory({'event': event, 'memory': missing}, 'native')
        for change in ({'rss_peak_bytes': 0}, {'rss_peak_bytes': 51539607553},
                       {'mlx_active_bytes': 51539607553}, {'mlx_peak_bytes': 299},
                       {'mlx_cache_bytes': -1}, {'mlx_peak_bytes': False},
                       {'mlx_active_bytes': 1.5}, {'mlx_cache_bytes': None}):
            with self.subTest(change=change), self.assertRaises(a.EvidenceError):
                engine.memory({'event': 'raw_decision', 'memory': {**memory, **change}}, 'native')
        # Runner lifecycle events contain matching nested native metadata, not
        # another invented top-level measurement. Synthetic tests are explicit.
        engine.memory({'event': 'load_start'}, 'runner')
        engine.synthetic = True
        engine.memory({'event': 'raw_decision'}, 'native')

    def test_real_source_payload_inventory_rejects_synthetic_hashes(self):
        identity = next(row for source, row in self.events if source == 'native' and row['event'] == 'inputs_verified')
        with self.assertRaisesRegex(a.EvidenceError, 'inventory differs'):
            a.verify_inputs_event(identity, self.frozen)

    def test_freeze_detects_changed_source_and_nonpaired_schedule(self):
        names = ['protocol.json', 'seed-manifest.json', 'schedule.json', 'analysis-plan.json',
                 'audit.py', 'runner.py', 'native.py', 'context_check.py']
        paths = ['experiments/kev4b_campaign_v1/' + name for name in names]
        paths += ['clef_snake/game.py', 'experiments/kev4b_v1/adapter.py', 'experiments/kev4b_v1/local_resolver.py',
                  'experiments/kev4b_v1/preflight.py', 'experiments/kev4b_v1/provenance.json',
                  'experiments/kev4b_v1/artifact-plan.json', 'experiments/kev4b_v1/preflight-results-2026-10-06.json',
                  'experiments/kev4b_v1/preflight-cases.json', 'data/2026-10-05/seed-manifest.json',
                  'data/2026-10-05-fiveway/seed-manifest.json', 'experiments/qwen3_snake_v1/seed-manifest.json']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); files = {}
            for relative in paths:
                dest = root / relative; dest.parent.mkdir(parents=True, exist_ok=True)
                raw = (ROOT / relative).read_bytes(); dest.write_bytes(raw); files[relative] = a.digest(raw)
            freeze = root / 'experiments/kev4b_campaign_v1/freeze.json'
            freeze.write_bytes(a.wire({'files': files}))
            parsed = a.read_frozen(root)
            self.assertEqual(len(parsed['schedule']), 90)
            for required in ('experiments/kev4b_campaign_v1/context_check.py',
                             'data/2026-10-05-fiveway/seed-manifest.json',
                             'experiments/qwen3_snake_v1/seed-manifest.json'):
                digest = files.pop(required); freeze.write_bytes(a.wire({'files': files}))
                with self.assertRaises(a.EvidenceError): a.read_frozen(root)
                files[required] = digest
            schedule_key = 'experiments/kev4b_campaign_v1/schedule.json'
            schedule_file = root / schedule_key
            saved_schedule = schedule_file.read_bytes()
            schedule = json.loads(saved_schedule)
            # Retain balanced condition counts while swapping the prescribed
            # order for two seed blocks. Counts alone cannot reject this.
            for offset in range(3):
                first, second = schedule['games'][offset], schedule['games'][offset + 3]
                first['variant'], second['variant'] = second['variant'], first['variant']
            schedule_file.write_bytes(a.wire(schedule)); files[schedule_key] = a.digest(schedule_file.read_bytes())
            freeze.write_bytes(a.wire({'files': files}))
            with self.assertRaisesRegex(a.EvidenceError, 'permutation order'): a.read_frozen(root)
            schedule_file.write_bytes(saved_schedule); files[schedule_key] = a.digest(saved_schedule)
            manifest_key = 'experiments/kev4b_campaign_v1/seed-manifest.json'
            manifest_file = root / manifest_key; saved_manifest = manifest_file.read_bytes()
            manifest = json.loads(saved_manifest)
            prior = json.loads((root / 'experiments/qwen3_snake_v1/seed-manifest.json').read_bytes())
            manifest['seeds'][0]['seed'] = prior['preflight_seeds'][0]
            manifest_file.write_bytes(a.wire(manifest)); files[manifest_key] = a.digest(manifest_file.read_bytes())
            freeze.write_bytes(a.wire({'files': files}))
            with self.assertRaisesRegex(a.EvidenceError, 'reused earlier'): a.read_frozen(root)
            manifest_file.write_bytes(saved_manifest); files[manifest_key] = a.digest(saved_manifest)
            analysis_key = 'experiments/kev4b_campaign_v1/analysis-plan.json'
            digest = files.pop(analysis_key)
            freeze.write_bytes(a.wire({'files': files}))
            with self.assertRaisesRegex(a.EvidenceError, 'analysis-plan.json'): a.read_frozen(root)
            files[analysis_key] = digest
            freeze.write_bytes(a.wire({'files': files}))
            analysis_file = root / analysis_key
            original = analysis_file.read_bytes(); analysis_file.write_bytes(original + b'\n')
            with self.assertRaisesRegex(a.EvidenceError, 'Frozen file changed'): a.read_frozen(root)
            analysis_file.write_bytes(original)
            game = root / 'clef_snake/game.py'; game.write_bytes(game.read_bytes() + b'\n')
            with self.assertRaisesRegex(a.EvidenceError, 'Frozen file changed'): a.read_frozen(root)
            files['clef_snake/game.py'] = a.digest(game.read_bytes())
            schedule_file = root / 'experiments/kev4b_campaign_v1/schedule.json'
            schedule = json.loads(schedule_file.read_bytes()); schedule['games'][1]['variant'] = schedule['games'][0]['variant']
            schedule_file.write_bytes(a.wire(schedule))
            files['experiments/kev4b_campaign_v1/schedule.json'] = a.digest(schedule_file.read_bytes())
            freeze.write_bytes(a.wire({'files': files}))
            with self.assertRaisesRegex(a.EvidenceError, 'three conditions'): a.read_frozen(root)


if __name__ == '__main__':
    unittest.main()
