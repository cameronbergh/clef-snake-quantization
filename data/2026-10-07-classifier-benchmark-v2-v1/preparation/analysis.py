# SPDX-License-Identifier: Apache-2.0
"""Frozen paired task analysis. Offline; imports no inference libraries."""
import argparse,json,math,statistics
from collections import defaultdict
from pathlib import Path
from .audit import audit
from .common import write

def auc(rows):
    pairs=sorted((r['raw_probabilities']['true'],bool(r['expected'])) for r in rows);pos=sum(y for _,y in pairs);neg=len(pairs)-pos
    if not pos or not neg:return None
    s=0;i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        s+=sum(y for _,y in pairs[i:j])*(i+1+j)/2;i=j
    return (s-pos*(pos+1)/2)/(pos*neg)

def metrics(rows):
    if not rows:return None
    correct=[int(r['correct']) for r in rows];errors=[];nll=[];conf=[];bins=[{'lo':i/10,'hi':(i+1)/10,'count':0,'confidence_sum':0.,'correct_sum':0} for i in range(10)]
    for r in rows:
        p=r['raw_probabilities'];gold='true' if r['primitive']=='noul' and r['expected'] else 'false' if r['primitive']=='noul' else str(r['expected'])
        errors.append(sum((v-int(k==gold))**2 for k,v in p.items()));nll.append(-math.log(max(1e-12,p[gold])))
        c=max(p.values());b=bins[min(9,int(c*10))];b['count']+=1;b['confidence_sum']+=c;b['correct_sum']+=r['correct'];conf.append(c)
    reliability=[];ece=0
    for b in bins:
        n=b['count'];a=b['correct_sum']/n if n else None;c=b['confidence_sum']/n if n else None
        if n:ece+=n/len(rows)*abs(a-c)
        reliability.append({'lo':b['lo'],'hi':b['hi'],'count':n,'accuracy':a,'confidence':c})
    result={'n':len(rows),'correct':sum(correct),'accuracy':statistics.mean(correct),'multiclass_brier':statistics.mean(errors),'clipped_nll':statistics.mean(nll),'top_confidence_ece_10bins':ece,'reliability':reliability,'native_raw_boundary_differences':sum(r['prediction']!=r['raw_prediction'] for r in rows)}
    if all(r['primitive']=='score' for r in rows):
        e=[abs(r['prediction']-r['expected']) for r in rows];result.update(ordinal_mae=statistics.mean(e),within_one=statistics.mean(int(x<=1) for x in e))
    if all(r['primitive']=='noul' for r in rows):
        e=[r['raw_probabilities']['true']-int(r['expected']) for r in rows];result.update(auc=auc(rows),binary_brier=statistics.mean(x*x for x in e),probability_mae=statistics.mean(abs(x) for x in e))
    return result

def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]);out=[0.]*len(ps);last=0
    for rank,i in enumerate(order):last=max(last,(len(ps)-rank)*ps[i]);out[i]=min(1,last)
    return out

def analyze(root,out):
    import numpy as np
    evidence,records=audit(root);summary={'audit':evidence,'interpretation':'Fixed public synthetic suite, not held-out or a validated difficulty scale. Task bootstrap is descriptive; latency is not globally contention-controlled. Primary contrasts versus same-runtime BF16 GGUF; official MPS BF16 is secondary.','conditions':{},'primary_macro_contrasts':[],'secondary_official_macro_contrasts':[],'pairing':{},'analysis':{'bootstrap_unit':'49 paired task vectors','samples':100000,'rng':'NumPy PCG64 seed 20261007','p_method':'centered paired bootstrap, two-sided +1 correction','primary_multiplicity':'Holm across four quant-vs-GGUF-BF16 contrasts','probability_clip':1e-12,'ece_bins':10,'versions':{'numpy':np.__version__}}}
    task_ids=list(dict.fromkeys(r['task_id'] for r in next(iter(records.values()))));vectors={};conditions=list(records)
    for condition,rows in records.items():
        tasks={t:metrics([r for r in rows if r['task_id']==t]) for t in task_ids};vectors[condition]=np.array([tasks[t]['accuracy'] for t in task_ids]);lat=[r['latency_ms'] for r in rows]
        summary['conditions'][condition]={'micro':metrics(rows),'macro_accuracy':float(vectors[condition].mean()),'tasks':tasks,'primitives':{p:metrics([r for r in rows if r['primitive']==p]) for p in ['choice','noul','score']},'latency':{'median_ms':statistics.median(lat),'mean_ms':statistics.mean(lat),'p95_ms':float(np.percentile(lat,95))},'worker':json.loads((Path(root)/'statuses'/f'{condition}.json').read_text())}
    quants=['q6-k-l','q4-k-m','iq2-m','q2-k'];rng=np.random.Generator(np.random.PCG64(20261007))
    contrasts=[(q,'gguf-bf16') for q in quants]+[(q,'official-bf16') for q in ['gguf-bf16']+quants]
    diffs=np.stack([vectors[q]-vectors[b] for q,b in contrasts]);observed=diffs.mean(axis=1);samples=np.empty((len(contrasts),100000))
    for start in range(0,100000,5000):
        indices=rng.integers(0,len(task_ids),size=(min(5000,100000-start),len(task_ids)));samples[:,start:start+len(indices)]=diffs[:,indices].mean(axis=2)
    for i,(q,b) in enumerate(contrasts):
        ci=np.percentile(samples[i],[2.5,97.5]);p=(1+int((np.abs(samples[i]-observed[i])>=abs(observed[i])-1e-15).sum()))/100001
        result={'condition':q,'baseline':b,'difference':float(observed[i]),'bootstrap_95_percentile':[float(x) for x in ci],'centered_bootstrap_p':p}
        summary['primary_macro_contrasts' if b=='gguf-bf16' else 'secondary_official_macro_contrasts'].append(result)
    adjusted=holm([r['centered_bootstrap_p'] for r in summary['primary_macro_contrasts']])
    for r,p in zip(summary['primary_macro_contrasts'],adjusted):r['holm_p']=p
    for baseline in ['gguf-bf16','official-bf16']:
        summary['pairing'][baseline]={}
        for q,rows in records.items():
            transitions={'both_correct':0,'gain':0,'loss':0,'both_wrong':0,'same_prediction':0}
            for a,b in zip(rows,records[baseline]):
                assert a['case_id']==b['case_id']
                key='both_correct' if a['correct'] and b['correct'] else 'gain' if a['correct'] else 'loss' if b['correct'] else 'both_wrong';transitions[key]+=1;transitions['same_prediction']+=a['prediction']==b['prediction']
            summary['pairing'][baseline][q]=transitions
    out=Path(out);out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',summary)
    lines=['# Compact non-game CLEF decision benchmark','', 'Complete locked v2: 49 tasks × 866 identical cases × six conditions; 5,196 scored calls plus 18 separate warmups.','', '| Condition | Micro accuracy | Equal-task macro accuracy | Median ms |','|---|---:|---:|---:|']
    for k in conditions:
        s=summary['conditions'][k];lines.append(f"| {k} | {s['micro']['accuracy']:.2%} | {s['macro_accuracy']:.2%} | {s['latency']['median_ms']:.1f} |")
    lines+=['','Primary paired macro contrasts (percentage points), versus same-runtime GGUF BF16:','', '| Condition | Difference | 95% task-bootstrap interval | Holm p |','|---|---:|---:|---:|']
    for r in summary['primary_macro_contrasts']:
        ci=r['bootstrap_95_percentile'];lines.append(f"| {r['condition']} | {100*r['difference']:+.2f} | {100*ci[0]:+.2f} to {100*ci[1]:+.2f} | {r['holm_p']:.4f} |")
    lines+=['','The public synthetic suite is not held-out confirmation, does not establish easy/hard task levels, and supports no broad or monotonic quantization claim. Runtime differences are isolated by the extra BF16 GGUF control, but kernels are not bit-identical and lexical output embeddings follow each actual GGUF. All native answers and raw head probabilities are preserved; any rounded-boundary disagreements are disclosed. Latency is descriptive with unrelated services untouched.','', 'Every task, primitive, calibration metric, reliability bin, paired correctness transition, resource receipt and boundary difference is in `summary.json`.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('out',type=Path);a=p.parse_args();s=analyze(a.data,a.out);print(json.dumps({k:{'micro':v['micro']['accuracy'],'macro':v['macro_accuracy']} for k,v in s['conditions'].items()},indent=2))
if __name__=='__main__':main()
