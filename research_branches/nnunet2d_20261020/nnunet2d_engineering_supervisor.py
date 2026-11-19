"""One-shot outer watchdog for the approved 1800 s / 100-attempt GPU cap."""
from __future__ import annotations
import json, os, signal, subprocess, sys, time
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('${RETINA_PROJECT_ROOT}')
EXP=ROOT/'experiments/nnunet2d_branch_20261020'
BRANCH=ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020'
ENV=ROOT/'envs/nnunet2d_v2'
WORKER=BRANCH/'nnunet2d_bounded_engineering.py'
ENG=EXP/'engineering'
ATT=ENG/'attempts'
LEDGER=ENG/'budget_ledger.json'
MAX_SECONDS=1800
MAX_ATTEMPTS=100

def gpu0_is_idle():
    out=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next((x.strip() for x in out.splitlines() if x.strip().startswith('0,')),None)
    if row is None: return False,{'query':out.strip(),'reason':'GPU0 row missing'}
    fields=[x.strip() for x in row.split(',')]
    used,util=int(fields[1]),int(fields[2])
    app_result=subprocess.run(['/usr/bin/nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
    if app_result.returncode==0:
        app_lines=[x.strip() for x in app_result.stdout.splitlines() if x.strip() and 'No running processes found' not in x]
        compute_rows=app_lines; graphics_rows=[]; fallback=False
    else:
        # Some driver/host combinations return a nonzero status for an empty compute-app query.
        # In that case parse the full authoritative process table; require it to be readable.
        full=subprocess.check_output(['/usr/bin/nvidia-smi'],text=True,stderr=subprocess.STDOUT)
        in_processes=False; compute_rows=[]; graphics_rows=[]
        for line in full.splitlines():
            if line.strip().startswith('| Processes:'): in_processes=True; continue
            if not in_processes: continue
            m=re.match(r'^\|\s*(\d+)\s+N/A\s+N/A\s+(\d+)\s+([CG])\s+(.+?)\s*\|?\s*$',line)
            if m and int(m.group(1))==0:
                item={'pid':int(m.group(2)),'type':m.group(3),'process':m.group(4).strip()}
                (compute_rows if item['type']=='C' else graphics_rows).append(item)
        fallback=True
    idle=used<=512 and util==0 and not compute_rows
    return idle,{'gpu0_memory_mib':used,'gpu0_utilization_pct':util,'compute_apps':compute_rows,
      'graphics_apps':graphics_rows,'compute_query_return_code':app_result.returncode,'fallback_process_table':fallback,'query':row}

def kill_group(p):
    try: os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait()

def count_attempts(log):
    path=Path(log)
    n=0
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: e=json.loads(line)
            except json.JSONDecodeError: continue
            if e.get('event')=='optimizer_attempt_start': n+=1
    return n

def main():
    if LEDGER.exists(): raise SystemExit(f'budget ledger already exists; refusing a second allocation: {LEDGER}')
    for p in (WORKER,EXP/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json',EXP/'engineering/environment_lock.json'):
        if not p.exists(): raise SystemExit(f'CPU gate artifact missing: {p}')
    idle,gpu=gpu0_is_idle()
    ledger={'status':'STARTED','approved_caps':{'gpu_window_seconds':MAX_SECONDS,'optimizer_attempts':MAX_ATTEMPTS},
            'preflight_gpu0':gpu,'started_utc':datetime.now(timezone.utc).isoformat(),
            'phases':[],'attempts_used':0,'gpu_window_elapsed_seconds':0.0,
            'official_test_accessed':False,'dev_model_calls':0,'scientific_training':False,
            'run_training_called':False}
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    if not idle:
        ledger['status']='STOPPED_GPU_BUSY'; LEDGER.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
        raise SystemExit('GPU0 busy; no CUDA context created and no update attempted')
    if ATT.exists() and any(ATT.iterdir()): raise SystemExit('engineering attempt outputs already exist; refusing overwrite')
    t0=time.monotonic()
    phases=[
      ('train50',['train50','--output',str(ATT/'train50'),'--checkpoint-out',str(ATT/'train50/checkpoint_stage50.pth'),'--state-out',str(ATT/'train50/state_stage50.pt')],50),
      ('fixture25',['fixture25','--output',str(ATT/'fixture25'),'--checkpoint-in',str(ATT/'train50/checkpoint_stage50.pth'),'--state-in',str(ATT/'train50/state_stage50.pt'),'--fixture',str(EXP/'engineering/fixtures/train_batch_fixture.pt'),'--checkpoint-out',str(ATT/'fixture25/checkpoint_continuous25.pth'),'--state-out',str(ATT/'fixture25/state_continuous25.pt')],25),
      ('split12',['split12','--output',str(ATT/'split12'),'--checkpoint-in',str(ATT/'train50/checkpoint_stage50.pth'),'--state-in',str(ATT/'train50/state_stage50.pt'),'--fixture',str(EXP/'engineering/fixtures/train_batch_fixture.pt'),'--checkpoint-out',str(ATT/'split12/checkpoint_split12.pth'),'--state-out',str(ATT/'split12/state_split12.pt')],12),
      ('resume13',['resume13','--output',str(ATT/'resume13'),'--checkpoint-in',str(ATT/'split12/checkpoint_split12.pth'),'--state-in',str(ATT/'split12/state_split12.pt'),'--fixture',str(EXP/'engineering/fixtures/train_batch_fixture.pt'),'--checkpoint-out',str(ATT/'resume13/checkpoint_resumed25.pth'),'--state-out',str(ATT/'resume13/state_resumed25.pt')],13),
    ]
    env=os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES':'0','nnUNet_raw':str(EXP/'raw'),'nnUNet_preprocessed':str(EXP/'preprocessed'),
      'nnUNet_results':str(EXP/'results'),'nnUNet_compile':'false','nnUNet_n_proc_DA':'1',
      'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    py=str(ENV/'bin/python')
    for name,args,limit in phases:
        elapsed=time.monotonic()-t0; remaining=MAX_SECONDS-elapsed
        if remaining<=20:
            ledger['status']='STOPPED_TIME_CAP_BEFORE_PHASE'; break
        if ledger['attempts_used']+limit>MAX_ATTEMPTS:
            ledger['status']='STOPPED_ATTEMPT_CAP_BEFORE_PHASE'; break
        cmd=[py,'-B',str(WORKER),*args]
        record={'name':name,'planned_optimizer_attempts':limit,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()}
        log_path=ATT/f'{name}.supervisor.log'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        with log_path.open('w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            while p.poll() is None:
                spent=time.monotonic()-t0
                if spent>=MAX_SECONDS-8:
                    timed_out=True; kill_group(p); break
                ledger['gpu_window_elapsed_seconds']=spent
                ledger['attempts_used']=sum(count_attempts(x.get('attempts_log','')) for x in ledger['phases'])+count_attempts(ATT/name/'attempts.jsonl')
                ledger['current_phase']=name; ledger['updated_utc']=datetime.now(timezone.utc).isoformat()
                LEDGER.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
                time.sleep(2)
        rc=p.returncode
        used=count_attempts(ATT/name/'attempts.jsonl')
        record.update({'return_code':rc,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start,
          'optimizer_attempts':used,'attempts_log':str(ATT/name/'attempts.jsonl'),'supervisor_log':str(log_path),
          'finished_utc':datetime.now(timezone.utc).isoformat()})
        ledger['phases'].append(record)
        ledger['attempts_used']=sum(x['optimizer_attempts'] for x in ledger['phases'])
        ledger['gpu_window_elapsed_seconds']=time.monotonic()-t0
        ledger['updated_utc']=datetime.now(timezone.utc).isoformat()
        if rc!=0 or timed_out or used!=limit:
            ledger['status']='STOPPED_PHASE_FAILURE_OR_BUDGET'; break
        LEDGER.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
    if len(ledger['phases'])==4 and ledger['attempts_used']==MAX_ATTEMPTS and all(p['return_code']==0 for p in ledger['phases']):
        ledger['status']='GPU_ENGINEERING_PHASES_COMPLETE_NEEDS_STATE_COMPARISON'
    elif ledger.get('status')=='STARTED': ledger['status']='STOPPED_INCOMPLETE'
    ledger['gpu_window_elapsed_seconds']=time.monotonic()-t0
    ledger['finished_utc']=datetime.now(timezone.utc).isoformat()
    LEDGER.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(ledger,indent=2),flush=True)
    if ledger['status']!='GPU_ENGINEERING_PHASES_COMPLETE_NEEDS_STATE_COMPARISON': raise SystemExit(2)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--preflight-only':
        idle,details=gpu0_is_idle()
        print(json.dumps({'gpu0_idle':idle,'details':details},indent=2))
        raise SystemExit(0 if idle else 1)
    main()
