"""Run the zero-update Train validation probe inside the original cumulative GPU cap."""
from __future__ import annotations
import json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('${RETINA_PROJECT_ROOT}'); EXP=ROOT/'experiments/nnunet2d_branch_20261020'
ENV=ROOT/'envs/nnunet2d_v2'; BRANCH=ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020'
ENG=EXP/'engineering'; LEDGER=ENG/'budget_ledger.json'; LIMIT=1800

def gpu_idle():
    out=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(x.strip() for x in out.splitlines() if x.strip().startswith('0,')); _,mem,util=[x.strip() for x in row.split(',')]
    apps=subprocess.run(['/usr/bin/nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
    active=[x for x in apps.stdout.splitlines() if x.strip()] if apps.returncode==0 else ['query_failed']
    return int(mem)<=512 and int(util)==0 and not active,{'query':row,'apps_returncode':apps.returncode,'apps':active}

def kill(p):
    try: os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait()

def main():
    ledger=json.loads(LEDGER.read_text(encoding='utf-8'))
    if ledger['attempts_used']!=100 or ledger['status']!='GPU_ENGINEERING_PHASES_COMPLETE_NEEDS_STATE_COMPARISON':
        raise SystemExit('original 100-attempt cap or completed phase ledger mismatch')
    previous=float(ledger['gpu_window_elapsed_seconds']); remaining=LIMIT-previous
    if remaining<=20: raise SystemExit('no remaining time for zero-update validation probe')
    idle,gpu=gpu_idle()
    probe={'gpu_preflight':gpu,'optimizer_attempts':0,'started_utc':datetime.now(timezone.utc).isoformat()}
    if not idle:
        probe['status']='SKIPPED_GPU_BUSY'; ledger['validation_probe']=probe
        LEDGER.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
        raise SystemExit('GPU0 busy; validation probe not launched')
    script=BRANCH/'nnunet2d_validation_probe.py'
    logpath=ENG/'validation_probe.supervisor.log'
    env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'0','nnUNet_raw':str(EXP/'raw'),
      'nnUNet_preprocessed':str(EXP/'preprocessed'),'nnUNet_results':str(EXP/'results'),
      'nnUNet_compile':'false','nnUNet_n_proc_DA':'1','OMP_NUM_THREADS':'1',
      'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    start=time.monotonic(); cmd=[str(ENV/'bin/python'),'-B',str(script)]; probe['command']=cmd
    with logpath.open('w',encoding='utf-8') as f:
        p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        timed=False
        while p.poll() is None:
            if time.monotonic()-start>=remaining-8: timed=True; kill(p); break
            time.sleep(1)
    duration=time.monotonic()-start; probe.update({'return_code':p.returncode,'timed_out':timed,
      'elapsed_seconds':duration,'log':str(logpath),'finished_utc':datetime.now(timezone.utc).isoformat()})
    report=ENG/'train_internal_val_probe.json'
    if report.exists(): probe['report']=str(report)
    probe['status']='PASS' if p.returncode==0 and not timed else 'FAILED'
    ledger['validation_probe']=probe
    ledger['gpu_window_elapsed_seconds']=previous+duration
    ledger['updated_utc']=datetime.now(timezone.utc).isoformat()
    LEDGER.write_text(json.dumps(ledger,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(probe,indent=2),flush=True)
    if probe['status']!='PASS': raise SystemExit(2)
    if report.exists(): print(report.read_text(encoding='utf-8'),flush=True)

if __name__=='__main__': main()
