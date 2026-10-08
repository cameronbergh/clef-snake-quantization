# SPDX-License-Identifier: Apache-2.0
"""Portable allowlisted export; only exact public upstream fixture exemption."""
import argparse,gzip,json,shutil
from pathlib import Path
from experiments.classifier_benchmark_v2_v1.common import ROOT,cases,sha,write,verify_freeze
PUBLIC_AUTH_EXAMPLE='Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.7fLxQx5vBd4Tq8ZcWq0uHf3Rexample'

def inspect_public_payload(payload):
    # cases() verifies both immutable upstream file bytes and its canonical lock.
    assert sum(PUBLIC_AUTH_EXAMPLE in c['request']['state'] for c in cases())==1
    occurrences=payload.count(PUBLIC_AUTH_EXAMPLE.encode());scrubbed=payload.replace(PUBLIC_AUTH_EXAMPLE.encode(),b'<known public synthetic upstream example>')
    forbidden=[b'/' + b'Users/',b'/' + b'home/',b'Authorization:' + b' Bearer',b'sk-' + b'proj-']
    for needle in forbidden:assert needle not in scrubbed,'private content marker'
    return occurrences

def export(source,destination):
    source=Path(source);dest=Path(destination);f=verify_freeze();assert json.loads((source/'handoff.json').read_text())['phase']=='inference_complete'
    dest.mkdir(parents=True,exist_ok=False)
    for name in list(f['files'])+['freeze.json']:
        target=dest/'preparation'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    for condition in f['condition_order']:
        target=dest/'records'/f'{condition}.jsonl.gz';target.parent.mkdir(exist_ok=True)
        b=(source/condition/'records.jsonl').read_bytes()
        with target.open('wb') as file:
            with gzip.GzipFile(filename='',mode='wb',fileobj=file,mtime=0) as zipped:zipped.write(b)
        target=dest/'statuses'/f'{condition}.json';target.parent.mkdir(exist_ok=True);shutil.copyfile(source/condition/'status.json',target)
    shutil.copyfile(source/'supervisor.jsonl',dest/'supervisor.jsonl')
    publication=dest/'publication';publication.mkdir()
    for name in ['export.py','README.md','audit.py','analysis.py']:shutil.copyfile(Path(__file__).parent/name,publication/name)
    inspection={}
    for path in sorted(dest.rglob('*')):
        if path.is_file():
            b=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
            inspection[str(path.relative_to(dest))]=inspect_public_payload(b)
    provenance={'version':'classifier_benchmark_publication_v1','inference_version':'classifier_benchmark_v2_v1','freeze_content_sha256':f['freeze_content_sha256'],'publication_export_source_sha256':sha(Path(__file__)),'scope':'publication-only narrow known-public-fixture exemption; frozen requests/inference/scoring/analysis unchanged','private_local_config_and_worker_logs_excluded':True,'worker_log_sha256':{c:sha(source/c/'worker.log') for c in f['condition_order']},'plain_and_compressed_payload_inspection':'pass','public_synthetic_auth_example_occurrences':{k:v for k,v in inspection.items() if v},'no_request_or_response_redaction':True,'audit_inventory_repair':'Exclude only the root hashes.json manifest; include nested upstream case lock manifests','publication_audit_source_sha256':sha(Path(__file__).parent/'audit.py'),'publication_analysis_wrapper_sha256':sha(Path(__file__).parent/'analysis.py')}
    write(dest/'provenance.json',provenance);inspect_public_payload((dest/'provenance.json').read_bytes())
    write(dest/'hashes.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();print(export(a.source,a.destination))
if __name__=='__main__':main()
