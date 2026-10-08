# SPDX-License-Identifier: Apache-2.0
"""Fail-closed external-volume serial process supervisor; explicit opt-in execution."""
import argparse,json,os,plistlib,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
from .common import ROOT,verify_freeze,write
UUID='99B50D1F-26DA-4175-BCB3-5D4852B1C5FB'

def volume_check(root):
    info=plistlib.loads(subprocess.check_output(['diskutil','info','-plist','/Volumes/Models']))
    assert info['VolumeUUID']==UUID and info['MountPoint']=='/Volumes/Models' and info['Internal'] is False and info['BusProtocol']=='USB' and info['Writable']
    assert Path(root).resolve().is_relative_to(Path('/Volumes/Models'))
    assert shutil.disk_usage('/Volumes/Models').free>=8*1024**3
    return {'volume_uuid':info['VolumeUUID'],'internal':False,'bus':'USB','free_bytes':shutil.disk_usage('/Volumes/Models').free}

def health():
    out={}
    for port in [8765,8767,8768,8769,8770]:
        try:
            with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=3) as r:d=json.load(r)
            out[str(port)]={k:d.get(k) for k in ['state','inferences','requests'] if k in d}
        except Exception as e:out[str(port)]={'read_error_type':type(e).__name__}
    return out

def snapshot(pid=None):
    vm=subprocess.check_output(['vm_stat'],text=True);page=int(vm.split('page size of ')[1].split(' bytes')[0]);stats={}
    for line in vm.splitlines()[1:]:
        if ':' in line:
            k,v=line.split(':',1)
            try:stats[k]=int(v.strip().rstrip('.'))*page
            except ValueError:pass
    available=sum(stats.get(k,0) for k in ['Pages free','Pages inactive','Pages purgeable'])
    processes=[];rss=0
    for line in subprocess.check_output(['ps','-axo','pid=,rss=,comm='],text=True).splitlines():
        parts=line.strip().split(None,2)
        if len(parts)!=3:continue
        n,b=int(parts[0]),int(parts[1])*1024
        if n==pid:rss=b
        if b>=256*1024**2 or n==pid:processes.append({'pid':n,'rss_bytes':b,'executable':Path(parts[2]).name})
    return {'unix_time':time.time(),'available_memory_estimate_bytes':available,'worker_pid':pid,'worker_rss_bytes':rss,'vm_bytes':stats,'processes':processes,'old_clef_health':health()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--local-config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.local_config.read_text());f=verify_freeze();protocol=json.loads((ROOT/'protocol.json').read_text());volume_check(a.out)
    assert a.out.exists() and not (a.out/'supervisor.jsonl').exists(),'never silently restart a campaign'
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']:assert os.environ.get(k)=='1'
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    for k in ['HF_HOME','XDG_CACHE_HOME','TMPDIR']:assert Path(os.environ[k]).resolve().is_relative_to(a.out.resolve())
    start=time.monotonic();completed=[];proc=None;state={'phase':'starting','pid':os.getpid(),'tmux_session':'clef-classifier-v2-v1','completed_conditions':[],'scored_calls':0,'warmup_calls':0,'maximum_forward_calls':5214,'freeze_sha256':f['freeze_content_sha256']}
    write(a.out/'handoff.json',state)
    resources=(a.out/'supervisor.jsonl').open('x')
    def event(kind,**extra):resources.write(json.dumps({'event':kind,**extra},allow_nan=False)+'\n');resources.flush()
    event('start',volume=volume_check(a.out),snapshot=snapshot(),freeze_content_sha256=f['freeze_content_sha256'])
    try:
        for condition in f['condition_order']:
            volume_check(a.out);s=snapshot();assert s['available_memory_estimate_bytes']>=protocol['minimum_available_memory_bytes'],'memory guard before load'
            target=a.out/condition;target.mkdir(exist_ok=False)
            log=(target/'worker.log').open('x')
            proc=subprocess.Popen([sys.executable,'-B','-u','-m','experiments.classifier_benchmark_v2_v1.worker','--condition',condition,'--local-config',str(a.local_config),'--out',str(target)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parents[1],env=os.environ.copy())
            state.update(phase='inference',current_condition=condition,worker_pid=proc.pid);write(a.out/'handoff.json',state)
            event('worker_start',condition=condition,pid=proc.pid,snapshot=s)
            cstart=time.monotonic();last=-1
            while proc.poll() is None:
                now=time.monotonic();assert now-start<=protocol['deadline_seconds'],'aggregate deadline';assert now-cstart<=protocol['condition_timeout_seconds'],'condition deadline'
                v=volume_check(a.out);s=snapshot(proc.pid);assert s['worker_rss_bytes']<=protocol['worker_rss_cap_bytes'],'RSS guard';assert s['available_memory_estimate_bytes']>=protocol['minimum_available_memory_bytes'],'memory guard'
                used=sum(q.stat().st_size for q in a.out.rglob('*') if q.is_file());assert used<=protocol['external_write_cap_bytes'],'write budget guard'
                status=json.loads((target/'status.json').read_text()) if (target/'status.json').exists() else {}
                assert status.get('forward_started',0)<=869
                state.update(scored_calls=866*len(completed)+status.get('scored',max(0,status.get('forward_completed',0)-3)),warmup_calls=3*len(completed)+min(3,status.get('forward_completed',0)),current_status=status,external_written_bytes=used)
                write(a.out/'handoff.json',state);event('sample',condition=condition,snapshot=s,volume=v,status=status,external_written_bytes=used)
                n=status.get('scored',0)
                if n!=last and (n==1 or n//100>max(0,last)//100):print(json.dumps({'condition':condition,'scored':n,'completed_conditions':completed}),flush=True)
                last=n;time.sleep(10)
            code=proc.wait();log.close();assert code==0,'worker failed (see preserved local worker.log)'
            status=json.loads((target/'status.json').read_text());assert status['state']=='complete' and status['forward_started']==status['forward_completed']==869
            completed.append(condition);state.update(completed_conditions=completed.copy(),scored_calls=866*len(completed),warmup_calls=3*len(completed));write(a.out/'handoff.json',state)
            event('worker_complete',condition=condition,status=status,snapshot=snapshot());print(json.dumps({'complete':condition,'scored_calls':state['scored_calls']}),flush=True)
            proc=None
        state.update(phase='inference_complete',worker_pid=None,elapsed_seconds=time.monotonic()-start);write(a.out/'handoff.json',state)
        event('complete',status=state,snapshot=snapshot(),volume=volume_check(a.out));print(json.dumps(state),flush=True)
    except BaseException as error:
        if proc and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        state.update(phase='stopped_error',error_type=type(error).__name__,error=str(error));write(a.out/'handoff.json',state);event('error',status=state);raise
    finally:resources.close()
if __name__=='__main__':main()
