# SPDX-License-Identifier: Apache-2.0
"""Portable exact request construction, native scoring and source verification."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REV='afb83bee3b74064ae5a5d58c05b352a8d0ef7240'
SUITE_SHA='57af4d24ba8d627b12b307c2ffd4ded3c0fe3828c803dc6d70c1449d89f0ac1c'
CONDITIONS=['official-bf16','gguf-bf16','q6-k-l','q4-k-m','iq2-m','q2-k']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def cases():
    p=ROOT/'vendor/cases/v2.toml';assert sha(p)==SUITE_SHA,'suite tamper'
    tasks=json.loads(p.with_suffix('.json').read_text())['task'];out=[]
    blob={'suite':'v2','tasks':[]}
    for t in tasks:
        q={'type':t['type'],'instructions':t['question']['instructions']}
        if t['question'].get('criteria'):q['criteria']=t['question']['criteria']
        blob['tasks'].append({'id':t['id'],'type':t['type'],'question':q,'cases':[[c['state'],c['expected']] for c in t['cases']]})
    locked=hashlib.sha256(json.dumps(blob,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert locked==json.loads((ROOT/'vendor/cases/hashes.json').read_text())['v2']['sha256'],'canonical mirror tamper'
    assert len(tasks)==49 and len({t['id'] for t in tasks})==49
    for t in tasks:
        q={'type':t['type'],'instructions':t['question']['instructions']}
        if t['question'].get('criteria'):q['criteria']=t['question']['criteria']
        for i,c in enumerate(t['cases']):
            gold=c['expected'];typ=t['type']
            assert isinstance(c['state'],str)
            if typ=='noul':assert type(gold) is bool
            elif typ=='score':assert type(gold) is int and 0<=gold<len(q['criteria'])
            else:assert typ=='choice' and gold in q['criteria']
            request={'model':'clef-flash','state':c['state'],'questions':{'decision':q}}
            row={'case_id':t['id']+':'+str(i).zfill(3),'task_id':t['id'],'primitive':typ,'expected':gold,'request':request}
            row['content_sha256']=digest(row);out.append(row)
    assert len(out)==866
    assert {k:sum(c['primitive']==k for c in out) for k in ('choice','noul','score')}=={'choice':362,'noul':311,'score':193}
    return out

def labels(question):
    if question['type']=='noul':return ['true','false']
    if question['type']=='score':return [str(i) for i in range(len(question['criteria']))]
    return list(question['criteria'])

def predict(question,response):
    a=response['answers']['decision'];typ=question['type'];assert a['type']==typ
    if typ=='noul':return bool(a['noul']>=.5)
    if typ=='choice':return a['choice']
    return int(max(labels(question),key=a['probabilities'].__getitem__))

def raw_predict(question,probs):
    if question['type']=='noul':return bool(probs['true']>=.5)
    value=max(labels(question),key=probs.__getitem__)
    return int(value) if question['type']=='score' else value

def reconstruct(question,probs):
    typ=question['type'];ls=labels(question)
    if typ=='noul':return {'type':'noul','noul':round(probs['true'],4)}
    if typ=='choice':
        best=max(ls,key=probs.__getitem__)
        return {'type':'choice','choice':best,'confidence':round(probs[best],4),'probabilities':{k:round(probs[k],4) for k in ls}}
    return {'type':'score','score':round(sum(i*probs[k] for i,k in enumerate(ls)),4),'confidence':round(max(probs.values()),4),'legend':dict(zip(ls,question['criteria'])),'probabilities':{k:round(probs[k],4) for k in ls}}

def verify_freeze(root=ROOT):
    root=Path(root);f=json.loads((root/'freeze.json').read_text())
    if 'freeze_content_sha256' in f:assert digest({k:v for k,v in f.items() if k!='freeze_content_sha256'})==f['freeze_content_sha256'],'freeze self commitment'
    for p,h in f['files'].items():assert sha(root/p)==h,'freeze tamper: '+p
    return f
