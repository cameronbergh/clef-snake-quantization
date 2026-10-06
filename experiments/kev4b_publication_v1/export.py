"""Allowlisted export of a completed campaign, preserving original evidence."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from experiments.kev4b_campaign_v1 import audit as original
from . import audit as portable

INPUT_JOURNALS = ('runner', 'native', 'supervisor')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def redact(value, *, asset_root, one_use_id, counts):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key == 'one_use_id':
                original.need(child == one_use_id, 'Unexpected one-use ID')
                result[key] = 'sha256:' + original.digest(child.encode())
                counts['one_use_identity_commitments'] += 1
            else:
                result[key] = redact(child, asset_root=asset_root, one_use_id=one_use_id, counts=counts)
        return result
    if isinstance(value, list):
        return [redact(v, asset_root=asset_root, one_use_id=one_use_id, counts=counts) for v in value]
    if isinstance(value, str) and value.startswith(asset_root + '/'):
        counts['asset_path_aliases'] += 1
        return portable.ALIAS + value[len(asset_root):]
    return value


def summary(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


def export(root, output, *, original_repo, repo):
    root, output, repo = Path(root), Path(output), Path(repo)
    original.need(not output.exists(), 'Export destination must be new')
    report = original.audit(root, repo=Path(original_repo))
    original.need(report['passed'] and report['complete'] and report['real'], 'Original campaign must pass complete independent audit')
    preparation = original.read_frozen(repo / portable.ARCHIVE)
    original.need(report['freeze_sha256'] == preparation['freeze_sha256'] == portable.FREEZE_SHA, 'Export freeze mismatch')
    approval_raw, claim_raw = (root/'approval.json').read_bytes(), (root/'claim.json').read_bytes()
    approval, claim = original.strict_json(approval_raw), original.strict_json(claim_raw)
    ident = approval['one_use_id']; asset_root = preparation['protocol']['storage']['asset_root']
    counts = defaultdict(int)
    source_hashes = {name: portable.sha(root/name) for name in
                     ('runner.jsonl', 'native.jsonl', 'supervisor.jsonl', 'approval.json', 'claim.json', 'context-report.json')}
    output.mkdir(parents=True)
    lat, loads, quant = defaultdict(list), defaultdict(list), defaultdict(list)
    memory, event_counts = {}, {}
    for source in INPUT_JOURNALS:
        lines = 0
        with (root/(source+'.jsonl')).open('rb') as inp, (output/(source+'.jsonl.gz')).open('wb') as out:
            with gzip.GzipFile(filename='', fileobj=out, mode='wb', mtime=0, compresslevel=9) as compressed:
                for raw in inp:
                    original.need(raw.endswith(b'\n'), 'Torn original journal')
                    row = original.strict_json(raw)
                    public = redact(row, asset_root=asset_root, one_use_id=ident, counts=counts)
                    portable.check_public(public)
                    for prefix in ('request', 'response'):
                        if prefix+'_base64' in row:
                            original.need(row[prefix+'_base64'] == public[prefix+'_base64']
                                          and row[prefix+'_sha256'] == public[prefix+'_sha256'], 'Wire evidence was altered')
                    compressed.write(original.wire(public)+b'\n'); lines += 1
                    if source == 'native':
                        v = row.get('variant'); event = row['event']
                        if event == 'raw_decision' and row['phase'] == 'game':
                            lat[v].append(1000*row['forward_wall_seconds_including_instrumentation'])
                        if event == 'native_inventory':loads[v].append(row['native_load_and_merge_seconds'])
                        if event == 'quantized_inventory':quant[v].append(row['quantize_seconds'])
                        for key, value in row.get('memory', {}).items():memory[key] = max(memory.get(key, 0), value)
                    if source == 'supervisor' and row['event'] == 'worker_exit':terminal = row
                    if source == 'supervisor' and row['event'] == 'supervisor_start':start = row
        event_counts[source] = lines
    receipt = {'schema_version': 1, 'publication_only': True, 'execution_authorized': False,
               'original_approval': redact(claim['approval'], asset_root=asset_root, one_use_id=ident, counts=counts),
               'original_claim_sha256': original.digest(claim_raw), 'deadline': claim['deadline'],
               'note': 'Commitments to a consumed private approval/claim, locally audited before export. This is not an execution approval.'}
    portable.check_public(receipt)
    dump(output/'authorization-receipt.json', receipt)
    context_raw = (root/'context-report.json').read_bytes()
    portable.check_public(original.strict_json(context_raw))
    (output/'context-report.json').write_bytes(context_raw)
    dump(output/'original-audit.json', report)
    prep = repo / portable.ARCHIVE / 'experiments/kev4b_campaign_v1'
    for public, source in (('campaign-seed-manifest.json','seed-manifest.json'),
                           ('schedule.json','schedule.json'), ('analysis-plan.json','analysis-plan.json')):
        (output/public).write_bytes((prep/source).read_bytes())
    dump(output/'seed-manifest.json', {'schema_version':1,'seeds':[r['seed'] for r in preparation['manifest']['seeds']],
                                     'note':'Flat index of the exact paired manifest; see campaign-seed-manifest.json for original design metadata.'})
    provenance = {'schema_version':1,'source_freeze_commit':portable.SOURCE_COMMIT,
                  'freeze_sha256':portable.FREEZE_SHA,'frozen_files':preparation['freeze']['files'],
                  'source_pins':preparation['protocol']['source_pins'],'original_file_sha256':source_hashes,
                  'original_journal_bytes':{name:(root/name).stat().st_size for name in source_hashes if name.endswith('.jsonl')},
                  'export_pipeline':'experiments.kev4b_publication_v1','journal_rows':event_counts,
                  'publication_source_sha256':{name:portable.sha(repo/name) for name in portable.PUBLICATION_FILES},
                  'journal_serialization':'All rows retained in order; JSON compacted without sorting keys; deterministic gzip, mtime=0.',
                  'changes':dict(counts), 'raw_requests_responses':'Exact base64 bytes and original SHA256 retained; never redacted.',
                  'private_files_excluded':['approval.json','claim.json','control.json','watchdog-ack.json','stdout.txt','stderr.txt'],
                  'receipt_policy':'Private identifiers replaced by SHA256 commitments; private approval/claim bytes are excluded.'}
    dump(output/'provenance.json', provenance)
    dump(output/'privacy-review.json', {'schema_version':1,'passed':True,'compressed_logs_decoded_and_checked':True,
                                      'decoded_request_response_bytes_checked':True,'redactions':dict(counts),
                                      'policy':'Explicit file allowlist; fail on unrecognized personal paths, credential patterns or private control-plane fields.',
                                      'original_journals_preserved':True})
    aggregate = {}
    for variant in original.VARIANTS:
        games = [g for g in report['games'] if g['variant']==variant]
        aggregate[variant] = {'games':len(games),'food':summary([g['score'] for g in games]),
                              'successful_moves':summary([g['steps'] for g in games]),
                              'attempts_total':sum(g['attempts'] for g in games),
                              'caps':sum(g['censored'] for g in games),'collisions':sum(not g['alive'] for g in games),
                              'native_forward_latency_ms':summary(lat[variant]),
                              'fresh_load_and_merge_seconds':summary(loads[variant]),
                              'quantization_seconds':summary(quant[variant]) if quant[variant] else None}
    dump(output/'results.json', {'schema_version':1,'dataset_kind':'kev_native_campaign_v1','state':'complete',
                                'campaign_id':'kev4b_campaign_v1','paired_seeds':30,'counts':report['counts'],
                                'games':report['games'],'aggregate_by_condition':aggregate,'native_memory_peak_bytes':memory,
                                'started_unix_time':start['unix_time'],'finished_unix_time':terminal['unix_time'],
                                'campaign_wall_seconds':terminal['monotonic']-start['monotonic'],
                                'latency_note':'Native forward wall time includes instrumentation, excludes loading/conversion and warmups; encountered states differ along trajectories.'})
    (output/'README.md').write_text('# Kev 4B prospective Snake campaign\n\n90 complete games, 30 fresh paired seeds, BF16 / affine8 / affine4; 180 warmups. All three conditions use the same native MLX runtime and unchanged FP32 pointer head. No improvement criterion was met. See ../../docs/KEV_CAMPAIGN_RESULTS.md and ../../analysis/2026-10-06-kev4b-campaign-v1/.\n\nEvery runner/native/supervisor row is retained in deterministic gzip. Requests, responses, logits, token IDs, inventories and trajectories retain exact values and order. Only resolver asset paths are aliased and one-use IDs committed with SHA256. Original journals are preserved outside Git; their hashes are in provenance.json. Private control/approval/history files are excluded. authorization-receipt.json is not an execution approval.\n\nAudit without weights, tokenizer or third-party packages:\n\n```sh\npython -B -m experiments.kev4b_publication_v1.audit data/2026-10-06-kev4b-campaign-v1\n```\n\nThe original local audit verified private approval bytes. The portable audit rechecks all gameplay, numerical, tensor, source, context and resource evidence, but cannot recheck private authorization history or prove that hardware produced the logged values.\n')
    proof = portable.audit(output, repo=repo, hashes=False)
    # Record the final proof, then verify that exact report after all file hashes exist.
    proof['hash_manifest_verified'] = True
    dump(output/'portable-audit.json', proof)
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{portable.sha(p)}  {p.relative_to(output)}\n' for p in files))
    original.need(portable.audit(output, repo=repo) == proof, 'Final audit differs from recorded proof')
    original.need(source_hashes == {name:portable.sha(root/name) for name in source_hashes}, 'Original evidence changed during export')
    return {'passed':True,'games':90,'journal_rows':event_counts,'redactions':dict(counts),
            'compressed_bytes':sum((output/(s+'.jsonl.gz')).stat().st_size for s in INPUT_JOURNALS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--original-repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run,args.out,original_repo=args.original_repo,repo=Path(__file__).resolve().parents[2]),indent=2))


if __name__ == '__main__':main()
