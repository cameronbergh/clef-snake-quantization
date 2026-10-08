# SPDX-License-Identifier: Apache-2.0
"""Publication audit repair: exclude only the root manifest, not nested locks.

All other original frozen audit checks are retained verbatim. No model calls.
"""
import argparse,gzip,hashlib,json,math
from collections import Counter
from pathlib import Path

def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def audit(root):
    root=Path(root);manifest=json.loads((root/'hashes.json').read_text())
    files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'hashes.json'}
    assert files==set(manifest),'incomplete hash inventory'
    for name,hash_ in manifest.items():assert h(root/name)==hash_,'payload hash: '+name
    frozen=root/'preparation';freeze=json.loads((frozen/'freeze.json').read_text())
    for name,hash_ in freeze['files'].items():assert h(frozen/name)==hash_,'frozen source: '+name
    content={k:v for k,v in freeze.items() if k!='freeze_content_sha256'}
    assert canon(content)==freeze['freeze_content_sha256'],'freeze self commitment'
    assert h(frozen/'vendor/cases/v2.toml')=='57af4d24ba8d627b12b307c2ffd4ded3c0fe3828c803dc6d70c1449d89f0ac1c'
    suite=json.loads((frozen/'vendor/cases/v2.json').read_text())['task'];cases=[]
    blob={'suite':'v2','tasks':[]}
    for t in suite:
        q={'type':t['type'],'instructions':t['question']['instructions']}
        if t['question'].get('criteria'):q['criteria']=t['question']['criteria']
        blob['tasks'].append({'id':t['id'],'type':t['type'],'question':q,'cases':[[c['state'],c['expected']] for c in t['cases']]})
    lock=hashlib.sha256(json.dumps(blob,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert lock=='f9d74c2885657a21690f10e5ca76e4a273b82e851fb61e9f6c00f9dc00d02ceb','canonical source mirror mismatch'
    for t in suite:
        q={'type':t['type'],'instructions':t['question']['instructions']}
        if t['question'].get('criteria'):q['criteria']=t['question']['criteria']
        for i,c in enumerate(t['cases']):
            row={'case_id':t['id']+':'+str(i).zfill(3),'task_id':t['id'],'primitive':t['type'],'expected':c['expected'],'request':{'model':'clef-flash','state':c['state'],'questions':{'decision':q}}}
            row['content_sha256']=canon(row);cases.append(row)
    assert len(cases)==866 and len(suite)==49
    requests=json.loads((frozen/'requests.json').read_text());assert len(requests)==866
    for a,b in zip(cases,requests):
        for k,v in a.items():assert v==b[k]
        assert 0<b['input_tokens']<=2048 and b['input_ids_sha256'] and b['truncated'] is False
    warm=[next(c for c in cases if c['primitive']==k) for k in ['choice','noul','score']]
    expected=[('warmup',c) for c in warm]+[('scored',c) for c in cases]
    rows={};boundaries={};total=0;native=Counter()
    for condition in freeze['condition_order']:
        with gzip.open(root/'records'/f'{condition}.jsonl.gz','rt') as stream:records=[json.loads(l) for l in stream]
        assert len(records)==869;scored=[];boundary=0
        for index,(r,(phase,c)) in enumerate(zip(records,expected),1):
            assert r['condition']==condition and r['phase']==phase and r['inference_id']==index
            assert r['case_id']==c['case_id'] and r['content_sha256']==c['content_sha256']
            assert json.dumps(r['request'],ensure_ascii=False)==json.dumps(c['request'],ensure_ascii=False),'exact request/field order'
            token=next(x for x in requests if x['case_id']==r['case_id'])
            assert r['input_ids_sha256']==token['input_ids_sha256']
            assert r['response']['usage']=={'input_tokens':token['input_tokens'],'output_tokens':0}
            assert r['response']['model']=='clef-flash' and set(r['response']['answers'])=={'decision'}
            q=c['request']['questions']['decision'];p=r['raw_probabilities'];typ=q['type']
            order=['true','false'] if typ=='noul' else list(q['criteria']) if typ=='choice' else [str(i) for i in range(len(q['criteria']))]
            assert set(p)==set(order) and all(type(v) is float and math.isfinite(v) and 0<=v<=1 for v in p.values()) and abs(sum(p.values())-1)<1e-5
            if typ=='noul':
                a={'type':'noul','noul':round(p['true'],4)};prediction=a['noul']>=.5;raw=p['true']>=.5
            elif typ=='choice':
                value=max(order,key=lambda k:p[k]);a={'type':'choice','choice':value,'confidence':round(p[value],4),'probabilities':{k:round(p[k],4) for k in order}};prediction=value;raw=value
            else:
                a={'type':'score','score':round(sum(i*p[k] for i,k in enumerate(order)),4),'confidence':round(max(p.values()),4),'legend':dict(zip(order,q['criteria'])),'probabilities':{k:round(p[k],4) for k in order}}
                prediction=int(max(order,key=lambda k:a['probabilities'][k]));raw=int(max(order,key=lambda k:p[k]))
            assert r['response']['answers']['decision']==a,'native answer reconstruction'
            assert r['prediction']==prediction and type(r['prediction']) is type(prediction)
            assert r['raw_prediction']==raw and type(r['raw_prediction']) is type(raw)
            assert math.isfinite(r['latency_ms']) and r['latency_ms']>0 and r['process_peak_rss_bytes']>0
            if phase=='scored':
                boundary+=prediction!=raw;scored.append({**r,'task_id':c['task_id'],'primitive':typ,'expected':c['expected'],'correct':prediction==c['expected']});native[typ]+=1
        status=json.loads((root/'statuses'/f'{condition}.json').read_text());assert status['state']=='complete' and status['records']==status['forward_started']==status['forward_completed']==869 and status['scored']==866 and status['warmup']==3
        rows[condition]=scored;boundaries[condition]=boundary;total+=len(records)
    assert total==5214 and sum(map(len,rows.values()))==5196 and native==Counter({'choice':2172,'noul':1866,'score':1158})
    events=[json.loads(l) for l in (root/'supervisor.jsonl').read_text().splitlines()]
    assert events[0]['event']=='start' and events[-1]['event']=='complete'
    assert events[0]['freeze_content_sha256']==freeze['freeze_content_sha256']
    starts=[e['condition'] for e in events if e['event']=='worker_start'];ends=[e['condition'] for e in events if e['event']=='worker_complete']
    assert starts==ends==freeze['condition_order']
    for e in events:
        if 'volume' in e:assert e['volume']['volume_uuid']=='99B50D1F-26DA-4175-BCB3-5D4852B1C5FB' and e['volume']['internal'] is False and e['volume']['free_bytes']>=8589934592
        if e['event']=='sample':
            assert e['snapshot']['worker_rss_bytes']<=42949672960 and e['snapshot']['available_memory_estimate_bytes']>=8589934592 and e['external_written_bytes']<=2147483648
    assert events[-1]['status']['scored_calls']==5196 and events[-1]['status']['warmup_calls']==18
    result={'audit':'pass','conditions':len(rows),'tasks':49,'cases_per_condition':866,'scored_calls':5196,'warmup_calls':18,'forward_started':5214,'forward_completed':5214,'native_raw_boundary_differences':boundaries,'hashes_verified':len(manifest),'freeze_content_sha256':freeze['freeze_content_sha256']}
    return result,rows

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();result,_=audit(a.root);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
