"""Portable, standard-library audit of privacy-reviewed Kev campaign evidence.

The unchanged frozen auditor reconstructs every decision and board transition.
This layer validates public hashes, path aliases and redacted supervisor receipts.
It never imports a runner, adapter, tokenizer or model framework.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import re

from experiments.kev4b_campaign_v1 import audit as frozen

ARCHIVE = Path('archives/kev4b_campaign_v1/108fc446')
FREEZE_SHA = '1d9e83c0213d1db7263ff4f21a45c51ccf0a037c9b8213283f817b9df2dea79c'
SOURCE_COMMIT = '108fc4464c5012793b82efe8200ab7438d17255e'
ALIAS = '$ASSET_ROOT'
PUBLICATION_FILES = ('experiments/kev4b_publication_v1/__init__.py',
                     'experiments/kev4b_publication_v1/audit.py',
                     'experiments/kev4b_publication_v1/export.py',
                     'experiments/kev4b_publication_v1/analysis.py',
                     'requirements-kev-analysis.txt')
PRIVATE = re.compile(rb'/Users/|/Volumes/|file://|(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|Bearer [A-Za-z0-9._-]{20,}')
PRIVATE_KEYS = {'authorization_provenance', 'worker_nonce', 'local_thread_id',
                'local_user_turn_id', 'supervisor_pid', 'worker_pid'}


def check_public(value):
    frozen.need(not PRIVATE.search(frozen.wire(value)), 'Private path or credential pattern in public evidence')
    _check_fields(value)


def _check_fields(value):
    if isinstance(value, dict):
        frozen.need(not PRIVATE_KEYS.intersection(value), 'Private control-plane field in public evidence')
        if 'one_use_id' in value:
            ident = value['one_use_id']
            frozen.need(isinstance(ident, str) and ident.startswith('sha256:')
                        and frozen.hexhash(ident[7:]), 'Unredacted one-use identity')
        for key in ('request', 'response'):
            if key + '_base64' in value:
                raw = frozen.body(value, key, optional=True)
                frozen.need(raw is None or not PRIVATE.search(raw), 'Private content in decoded wire bytes')
        for child in value.values():
            if isinstance(child, (dict, list)):
                _check_fields(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                _check_fields(child)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_hashes(root):
    listed = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        relative = Path(name)
        frozen.need(frozen.hexhash(expected) and name not in listed and not relative.is_absolute()
                    and '..' not in relative.parts and name != 'SHA256SUMS', 'Unsafe or duplicate hash entry')
        path = root / relative
        frozen.need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Hash path escapes dataset')
        frozen.need(sha(path) == expected, 'Public file hash mismatch: ' + name)
        listed[name] = expected
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
    frozen.need(set(listed) == actual, 'Hash manifest must cover exactly every public dataset file')


class CompressedJournal:
    def __init__(self, path):
        self.path, self.name = path, path.name

    def open(self, mode):
        return gzip.open(self.path, mode)


def journal(root, source, preparation):
    for now, _, row in frozen.journal(CompressedJournal(root / (source + '.jsonl.gz')), source):
        check_public(row)
        if source == 'native' and row['event'] == 'local_snapshot_resolution':
            resolver = row['resolver']
            cache = ALIAS + '/hf/hub'
            frozen.need(resolver['cache_root'] == cache, 'Public cache alias changed')
            resolver['cache_root'] = preparation['protocol']['storage']['asset_root'] + '/hf/hub'
            for ident, snapshot in resolver['snapshots'].items():
                expected = cache + '/models--' + ident.replace('/', '--') + '/snapshots/' + snapshot['revision']
                frozen.need(snapshot['path'] == expected, 'Public snapshot alias changed')
                snapshot['path'] = resolver['cache_root'] + expected[len(cache):]
        yield now, source, row


def verify_supervisor(root, engine, preparation):
    receipt = frozen.strict_json((root / 'authorization-receipt.json').read_bytes())
    check_public(receipt)
    frozen.need(receipt['schema_version'] == 1 and receipt['publication_only'] is True
                and receipt['execution_authorized'] is False, 'Public receipt must not authorize execution')
    approval = receipt['original_approval']
    claim_sha = receipt['original_claim_sha256']
    frozen.need(frozen.hexhash(approval['approval_sha256']) and frozen.hexhash(claim_sha)
                and approval['freeze_sha256'] == FREEZE_SHA
                and approval['limits'] == preparation['protocol']['limits'], 'Original receipt commitment mismatch')
    frozen.need(engine.approval == {**approval, 'claim_sha256': claim_sha}, 'Runner receipt commitment differs')
    raw = (root / 'context-report.json').read_bytes()
    context = frozen.strict_json(raw)
    check_public(context)
    frozen.need(frozen.digest(raw) == approval['context_report_sha256'], 'Context bytes changed')
    frozen.verify_context_report(context, preparation)
    deadline = receipt['deadline']
    started = initial = final = exited = False
    for _, _, row in journal(root, 'supervisor', preparation):
        frozen.need(row.get('synthetic') is False and not exited, 'Invalid public supervisor lifecycle')
        event = row['event']
        if event == 'supervisor_start':
            frozen.need(not started and row['approval'] == approval and row['claim_sha256'] == claim_sha
                        and row['freeze_sha256'] == FREEZE_SHA and row['limits'] == approval['limits']
                        and row['deadline'] == deadline, 'Supervisor receipt binding changed')
            frozen.need(frozen.finite(deadline) and 0 < deadline-row['monotonic'] <= 43200
                        and row['monotonic'] <= engine.start_time, 'Supervisor deadline/start ordering changed')
            started = True
        elif event == 'resource_sample':
            frozen.need(started and not final and row['monotonic'] <= deadline, 'Resource sample outside lifecycle')
            storage, rss = row['storage'], row['current_rss_bytes']
            frozen.need(all(frozen.integer(storage.get(k)) for k in ('free_bytes', 'root_bytes', 'evidence_bytes'))
                        and storage['free_bytes'] >= 8589934592 and storage['root_bytes'] <= 60129542144
                        and storage['evidence_bytes'] <= 17179869184, 'Storage measurements missing or out of bounds')
            phase = row['sample_phase']
            if phase in ('initial', 'periodic'):
                frozen.need(row['worker_state'] == 'running' and frozen.integer(rss, 1)
                            and rss <= 51539607552, 'Running resource sample lacks bounded RSS')
                if phase == 'initial':
                    frozen.need(not initial and engine.first_native_load_time is not None
                                and row['monotonic'] <= engine.first_native_load_time, 'Late/repeated initial resource sample')
                    initial = True
                else:
                    frozen.need(initial, 'Periodic resource sample precedes initial sample')
            elif phase == 'final':
                frozen.need(initial and row['worker_state'] == 'exited' and rss is None
                            and engine.end_time <= row['monotonic'], 'Invalid final resource sample')
                final = True
            else:
                raise frozen.EvidenceError('Unknown resource phase')
        elif event == 'worker_exit':
            frozen.need(started and initial and final and row['returncode'] == 0 and engine.finished
                        and engine.end_time <= row['monotonic'] <= deadline, 'Missing successful worker exit')
            exited = True
        else:
            raise frozen.EvidenceError('Unexpected supervisor event')
    frozen.need(started and exited, 'Incomplete supervisor evidence')
    return row['sequence']


def verify_summary(value, samples):
    values = sorted(samples)
    n = len(values)
    expected = {'n':n, 'mean':math.fsum(values)/n,
                'median':values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2,
                'min':values[0], 'max':values[-1]}
    frozen.need(set(value)==set(expected) and type(value['n']) is int and value['n']==n,
                'Derived summary shape/count differs')
    for key in ('mean','median','min','max'):
        frozen.need(frozen.finite(value[key]) and math.isclose(value[key],expected[key],rel_tol=1e-12,abs_tol=1e-12),
                    'Derived summary differs: '+key)


def audit(root, *, repo=None, hashes=True):
    root = Path(root)
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    if hashes:
        verify_hashes(root)
    preparation = frozen.read_frozen(repo / ARCHIVE)
    frozen.need(preparation['freeze_sha256'] == FREEZE_SHA, 'Unexpected preparation freeze')
    auditor_path = 'experiments/kev4b_campaign_v1/audit.py'
    frozen.need(sha(Path(frozen.__file__)) == preparation['freeze']['files'][auditor_path], 'Imported auditor differs from frozen source')
    engine = frozen.CampaignAudit(preparation)
    latency, loads, quantize = defaultdict(list), defaultdict(list), defaultdict(list)
    memory, journal_rows = {}, {}
    for _, source, row in heapq.merge(journal(root, 'runner', preparation),
                                     journal(root, 'native', preparation), key=lambda r: r[0]):
        engine.feed(source, row)
        journal_rows[source] = row['sequence']
        if source == 'native':
            v,event = row.get('variant'),row['event']
            if event == 'raw_decision' and row['phase']=='game':
                latency[v].append(row['forward_wall_seconds_including_instrumentation']*1000)
            if event == 'native_inventory':loads[v].append(row['native_load_and_merge_seconds'])
            if event == 'quantized_inventory':quantize[v].append(row['quantize_seconds'])
            for key,value in row.get('memory',{}).items():memory[key]=max(memory.get(key,0),value)
    report = engine.result()
    frozen.need(report['passed'] and report['complete'] and report['real'], 'Incomplete or synthetic campaign')
    journal_rows['supervisor'] = verify_supervisor(root, engine, preparation)
    for name in ('original-audit.json', 'results.json', 'provenance.json', 'privacy-review.json',
                 'seed-manifest.json', 'campaign-seed-manifest.json', 'schedule.json', 'analysis-plan.json'):
        check_public(frozen.strict_json((root / name).read_bytes()))
    original = frozen.strict_json((root / 'original-audit.json').read_bytes())
    results = frozen.strict_json((root / 'results.json').read_bytes())
    frozen.need(original['passed'] is True and original['freeze_sha256'] == FREEZE_SHA
                and original['counts'] == report['counts'] and original['games'] == report['games'], 'Original audit summary differs')
    frozen.need(results['games'] == report['games'] and results['counts'] == report['counts'], 'Published score rows differ from reconstructed games')
    frozen.exact(results['games'],report['games'],'Published game values/types/order differ')
    frozen.exact(original['games'],report['games'],'Original summary values/types/order differ')
    frozen.need(results['dataset_kind']=='kev_native_campaign_v1' and results['paired_seeds']==30
                and results['state']=='complete' and results['native_memory_peak_bytes']==memory,
                'Published result identity or memory differs')
    for v in frozen.VARIANTS:
        g = [r for r in report['games'] if r['variant']==v]
        summary = results['aggregate_by_condition'][v]
        frozen.need(summary['games']==30 and summary['caps']==sum(r['censored'] for r in g)
                    and summary['collisions']==sum(not r['alive'] for r in g)
                    and summary['attempts_total']==sum(r['attempts'] for r in g), 'Derived condition counts differ')
        for key, samples in (('food',[r['score'] for r in g]),('successful_moves',[r['steps'] for r in g]),
                             ('native_forward_latency_ms',latency[v]),('fresh_load_and_merge_seconds',loads[v])):
            verify_summary(summary[key],samples)
        if quantize[v]:verify_summary(summary['quantization_seconds'],quantize[v])
        else:frozen.need(summary['quantization_seconds'] is None,'BF16 has no quantization timing')
    provenance = frozen.strict_json((root / 'provenance.json').read_bytes())
    frozen.need(provenance['source_freeze_commit'] == SOURCE_COMMIT and provenance['freeze_sha256'] == FREEZE_SHA
                and provenance['frozen_files'] == preparation['freeze']['files'], 'Publication source provenance differs')
    frozen.need(provenance['journal_rows']==journal_rows,'Public journal row counts differ')
    frozen.need(provenance['publication_source_sha256']=={name:sha(repo/name) for name in PUBLICATION_FILES},
                'Versioned publication source changed')
    prep = repo / ARCHIVE / 'experiments/kev4b_campaign_v1'
    for public, source in (('campaign-seed-manifest.json', 'seed-manifest.json'),
                           ('schedule.json', 'schedule.json'), ('analysis-plan.json', 'analysis-plan.json')):
        frozen.need((root / public).read_bytes() == (prep / source).read_bytes(), 'Frozen trial design copy changed')
    manifest = frozen.strict_json((root / 'seed-manifest.json').read_bytes())
    frozen.need(manifest['seeds'] == [r['seed'] for r in preparation['manifest']['seeds']], 'Flat seed index changed')
    report.update(audit_kind='portable_redacted_evidence_v1', hash_manifest_verified=hashes,
                  private_approval_bytes_reverified=False)
    report['limitations'].append('Original private approval/claim bytes were verified locally before export; public data retain their hashes and a one-use identity commitment, not personal authorization history or an executable approval.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    result = audit(args.dataset)
    result.pop('games')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
