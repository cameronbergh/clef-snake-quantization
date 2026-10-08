# SPDX-License-Identifier: Apache-2.0
"""One isolated model, one forward per exact SystemOne request. Opt-in CLI only."""
import argparse,json,os,resource,sys,time,traceback
from pathlib import Path
from .common import ROOT,cases,sha,write,verify_freeze,digest,predict,raw_predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--local-config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    verify_freeze();cfg=json.loads(args.local_config.read_text());out=args.out
    assert out.exists() and not (out/'records.jsonl').exists();cond=cfg['conditions'][args.condition]
    os.environ['CLEF_OFFICIAL_DIR']=cfg['official'];os.environ['CLEF_BRIDGE_LIBRARY']=cfg['bridge']
    sys.path.insert(0,cfg['official'])
    import torch
    import joint_schema_model as official
    assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert sha(Path(cfg['official'])/'joint_head.safetensors')==cfg['head_sha256']
    assert sha(Path(cfg['official'])/'joint_schema_model.py')==cfg['source_sha256']
    if cond.get('gguf'):assert sha(cond['gguf'])==cond['sha256']
    assert sha(cfg['bridge'])==cfg['bridge_sha256']
    assets=json.loads((ROOT/'assets.json').read_text())
    for asset in assets['official_files']:
        if asset['file'].startswith('model-') and args.condition!='official-bf16':continue
        assert sha(Path(cfg['official'])/asset['file'])==asset['sha256'],'official asset changed'
    write(out/'status.json',{'state':'loading','condition':args.condition,'pid':os.getpid(),'forward_started':0,'forward_completed':0,'records':0})
    start=time.monotonic()
    if args.condition=='official-bf16':model,processor=official.load_release_model(cfg['official'],device='mps',dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa')
    else:
        from .support.model_bridge import load
        model,processor=load(cond['gguf'])
    assert sum(p.numel() for p in model.head.parameters())==121762820
    torch.mps.synchronize();load_s=time.monotonic()-start
    original_forward=model.forward;capture={};counts={'started':0,'completed':0}
    def captured(batch):
        counts['started']+=1
        write(out/'status.json',{'state':'running','condition':args.condition,'pid':os.getpid(),'forward_started':counts['started'],'forward_completed':counts['completed'],'records':counts['completed'],'case_id':capture['case_id']})
        logits=original_forward(batch)
        counts['completed']+=1
        assert len(logits)==1 and len(logits[0])==1
        question=batch['records'][0].questions[0]
        capture['raw_probabilities']=dict(zip(question.option_ids,logits[0][0].float().softmax(-1).tolist()))
        capture['input_ids']=batch['records'][0].input_ids
        return logits
    model.forward=captured
    rows=cases();warm=[next(c for c in rows if c['primitive']==k) for k in ['choice','noul','score']]
    tokens=json.loads((ROOT/'requests.json').read_text())
    expected_tokens={r['case_id']:r['input_ids_sha256'] for r in tokens}
    with (out/'records.jsonl').open('x') as stream:
        for index,(phase,c) in enumerate([('warmup',c) for c in warm]+[('scored',c) for c in rows],1):
            capture={'case_id':c['case_id']};t=time.perf_counter()
            with torch.inference_mode():response=official.systemone(model,processor,c['request'],max_length=4096)
            torch.mps.synchronize();ms=(time.perf_counter()-t)*1000
            assert counts['started']==counts['completed']==index<=869
            assert digest(capture['input_ids'])==expected_tokens[c['case_id']],'token mismatch/truncation'
            q=c['request']['questions']['decision'];probs=capture['raw_probabilities']
            assert all(0<=v<=1 for v in probs.values()) and abs(sum(probs.values())-1)<1e-5
            record={'condition':args.condition,'phase':phase,'inference_id':index,'case_id':c['case_id'],'content_sha256':c['content_sha256'],'request':c['request'],'response':response,'raw_probabilities':probs,'input_ids_sha256':digest(capture['input_ids']),'prediction':predict(q,response),'raw_prediction':raw_predict(q,probs),'latency_ms':ms,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            write(out/'status.json',{'state':'running','condition':args.condition,'pid':os.getpid(),'forward_started':index,'forward_completed':index,'records':index,'scored':max(0,index-3),'case_id':c['case_id']})
            if index==4 or index%100==0 or index==869:print(json.dumps({'condition':args.condition,'records':index,'scored':max(0,index-3),'latency_ms':ms}),flush=True)
    if hasattr(model,'backbone') and hasattr(model.backbone,'close'):model.backbone.close()
    del model;torch.mps.empty_cache();torch.mps.synchronize()
    write(out/'status.json',{'state':'complete','condition':args.condition,'pid':os.getpid(),'forward_started':counts['started'],'forward_completed':counts['completed'],'records':869,'scored':866,'warmup':3,'load_seconds':load_s,'elapsed_seconds':time.monotonic()-start,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

if __name__=='__main__':
    try:main()
    except BaseException:traceback.print_exc();raise
