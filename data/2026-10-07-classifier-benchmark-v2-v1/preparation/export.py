# SPDX-License-Identifier: Apache-2.0
"""Explicit allowlisted portable evidence export; no private worker logs/config."""
import argparse,gzip,json,shutil
from pathlib import Path
from .common import ROOT,sha,write,verify_freeze

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
    write(dest/'provenance.json',{'version':'classifier_benchmark_v2_v1','freeze_content_sha256':f['freeze_content_sha256'],'scope':'only new compact benchmark; no historical datasets or model weights','private_local_config_and_worker_logs_excluded':True,'worker_log_sha256':{c:sha(source/c/'worker.log') for c in f['condition_order']}})
    # Inspect compressed and uncompressed payloads. Frozen source references the mandated
    # Models storage rule; personal paths and secrets are forbidden in evidence/log payloads.
    forbidden=[b'/' + b'Users/',b'/' + b'home/',b'Authorization:' + b' Bearer',b'sk-' + b'proj-']
    for path in dest.rglob('*'):
        if path.is_file():
            b=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
            for needle in forbidden:assert needle not in b,'private content marker in '+str(path.relative_to(dest))
    write(dest/'hashes.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();print(export(a.source,a.destination))
if __name__=='__main__':main()
