# SPDX-License-Identifier: Apache-2.0
"""Freeze exact requests, existing assets and all execution/analysis code before calls."""
import argparse,hashlib,importlib.metadata,json,os,platform,sys,tomllib
from collections import Counter
from pathlib import Path
from .common import ROOT,CONDITIONS,cases,sha,digest,write
from .supervisor import volume_check
from .support.catalog import MODELS,HEAD_SHA,HEAD_SOURCE_SHA,OFFICIAL_REV,NATIVE_REV,QUANT_REV

def main():
    p=argparse.ArgumentParser();p.add_argument('--external-root',type=Path,required=True);p.add_argument('--official',type=Path,required=True);p.add_argument('--bridge',type=Path,required=True);p.add_argument('--python',type=Path,required=True);a=p.parse_args()
    assert not (ROOT/'freeze.json').exists(),'freeze immutable once created'
    volume=volume_check(a.external_root)
    assert a.external_root.exists()
    os.environ['CLEF_OFFICIAL_DIR']=str(a.official);os.environ['CLEF_BRIDGE_LIBRARY']=str(a.bridge)
    sys.path.insert(0,str(a.official));from transformers import AutoProcessor
    from joint_schema_model import encode_record
    import gguf
    assert sha(a.official/'joint_head.safetensors')==HEAD_SHA and sha(a.official/'joint_schema_model.py')==HEAD_SOURCE_SHA
    assert sha(a.bridge)=='305382000f6e514b25db17b70eed62b5fafcee1b39f443e7427144326b1a01d3'
    processor=AutoProcessor.from_pretrained(a.official,local_files_only=True);rows=cases()
    for c in rows:
        encoded=encode_record(processor.tokenizer,c['request'],max_length=4096,processor=processor)
        unbounded=encode_record(processor.tokenizer,c['request'],max_length=16384,processor=processor)
        assert encoded.input_ids==unbounded.input_ids and 0<len(encoded.input_ids)<=2048
        c.update(input_tokens=len(encoded.input_ids),input_ids_sha256=digest(encoded.input_ids),truncated=False)
    write(ROOT/'requests.json',rows)
    raw=tomllib.loads((ROOT/'vendor/cases/v2.toml').read_text())['task']
    blob={'suite':'v2','tasks':[]}
    for t in raw:
        q={'type':t['type'],'instructions':t['question']['instructions']}
        if t['question'].get('criteria'):q['criteria']=t['question']['criteria']
        blob['tasks'].append({'id':t['id'],'type':t['type'],'question':q,'cases':[[c['state'],c['expected']] for c in t['cases']]})
    upstream_digest=hashlib.sha256(json.dumps(blob,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert upstream_digest==json.loads((ROOT/'vendor/cases/hashes.json').read_text())['v2']['sha256']
    print('Requests and upstream lock verified; hashing all existing payloads.',flush=True)
    official_files=[]
    verified=json.loads((a.official/'VERIFIED.json').read_text());assert verified['revision']==OFFICIAL_REV
    expected_official={r['file']:r for r in verified['files']}
    for path in sorted(a.official.rglob('*')):
        if not path.is_file() or path.name.startswith('.') or '__pycache__' in path.parts or '.cache' in path.parts:continue
        print('Hashing official',str(path.relative_to(a.official)),flush=True)
        entry={'file':str(path.relative_to(a.official)),'size_bytes':path.stat().st_size,'sha256':sha(path)}
        if entry['file'] in expected_official:
            original=expected_official[entry['file']];assert entry['sha256']==original['sha256'] and entry['size_bytes']==original['size']
        if path.suffix=='.safetensors':
            with path.open('rb') as file:
                n=int.from_bytes(file.read(8),'little');header=json.loads(file.read(n))
            tensor=[v for k,v in header.items() if k!='__metadata__'];entry['tensor_count']=len(tensor);entry['tensor_dtype_counts']=dict(Counter(x['dtype'] for x in tensor))
        official_files.append(entry)
    conds={'official-bf16':{'runtime':'PyTorch/MPS BF16','official_revision':OFFICIAL_REV}}
    asset_conditions={}
    for name in CONDITIONS[1:]:
        if name=='gguf-bf16':
            path=Path('/Volumes/Models/clef-flash-q6-k-l/clef-backbone-validation-BF16.gguf');expected_sha='a5b6d1725f237092d68056a1ec35eb9efcf0c74f9697e5848bc63a4651fc2191';expected_size=17920696864
        else:path=Path('/Volumes/Models/clef-flash-'+name)/MODELS[name]['file'];expected_sha=MODELS[name]['sha256'];expected_size=MODELS[name]['size']
        assert path.stat().st_size==expected_size;print('Hashing',path.name,flush=True);assert sha(path)==expected_sha
        reader=gguf.GGUFReader(path);types=Counter(t.tensor_type.name for t in reader.tensors);output=next(t for t in reader.tensors if t.name=='output.weight');embedding=next(t for t in reader.tensors if t.name=='token_embd.weight')
        entry={'file':path.name,'size_bytes':expected_size,'sha256':expected_sha,'tensor_count':len(reader.tensors),'tensor_type_counts':dict(types),'output_tensor_type':output.tensor_type.name,'input_tensor_type':embedding.tensor_type.name,'runtime':'llama.cpp/Metal raw all-token states + official MPS BF16 head'}
        if name=='gguf-bf16':assert entry['tensor_count']==427 and dict(types)=={'F32':177,'BF16':250} and output.tensor_type.name==embedding.tensor_type.name=='BF16'
        asset_conditions[name]=entry;conds[name]={'gguf':str(path),'sha256':expected_sha}
    versions={}
    for package in ['torch','transformers','safetensors','numpy','gguf','tokenizers','huggingface-hub']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not-distribution'
    assets={'official_revision':OFFICIAL_REV,'head_sha256':HEAD_SHA,'head_source_sha256':HEAD_SOURCE_SHA,'official_files':official_files,'gguf_conditions':asset_conditions,'native_revision':NATIVE_REV,'quant_revision':QUANT_REV,'bridge_sha256':sha(a.bridge),'head_parameters':121762820,'environment':{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'packages':versions},'volume':volume,'upstream_canonical_sha256':upstream_digest,'context':{'min_tokens':min(c['input_tokens'] for c in rows),'max_tokens':max(c['input_tokens'] for c in rows),'mean_tokens':sum(c['input_tokens'] for c in rows)/len(rows),'truncated_cases':0}}
    write(ROOT/'assets.json',assets)
    cfg={'official':str(a.official),'bridge':str(a.bridge),'python':str(a.python),'head_sha256':HEAD_SHA,'source_sha256':HEAD_SOURCE_SHA,'bridge_sha256':sha(a.bridge),'conditions':conds}
    write(a.external_root/'local-config.json',cfg)
    order=sorted(CONDITIONS,key=lambda k:hashlib.sha256(('classifier-benchmark-v2-v1:condition:'+k).encode()).hexdigest())
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='freeze.json'}
    freeze={'version':'classifier_benchmark_v2_v1','files':files,'condition_order':order,'case_order':'exact upstream file task/case order','scored_calls':5196,'warmup_calls':18,'assets_sha256':sha(ROOT/'assets.json'),'requests_sha256':sha(ROOT/'requests.json')};freeze['freeze_content_sha256']=digest(freeze);write(ROOT/'freeze.json',freeze)
    write(a.external_root/'handoff.json',{'phase':'frozen_preinference','scored_calls':0,'warmup_calls':0,'freeze_content_sha256':freeze['freeze_content_sha256'],'condition_order':order,'tmux_session':'clef-classifier-v2-v1'})
    print(json.dumps({'freeze':freeze['freeze_content_sha256'],'condition_order':order,'context':assets['context']},indent=2))
if __name__=='__main__':main()
