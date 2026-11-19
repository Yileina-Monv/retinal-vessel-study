from __future__ import annotations
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time
from datetime import datetime, timezone
BASE=pathlib.Path('${RETINA_PROJECT_ROOT}')
FREEZE=BASE/'docs/nnunet2d_seed2026_100epochs_20261020'
PROTO=json.loads((FREEZE/'protocol.json').read_text())
ROOT=pathlib.Path(PROTO['paths']['scientific_root'])
PREFLIGHT=ROOT/'preflight/verification.json'
BINDING=ROOT/'execution_binding.json'
PYTHON=str(PROTO['paths']['python'])
BRANCH=pathlib.Path(PROTO['paths']['branch_code'])
LEDGER=ROOT/'budget_ledger.json'
EVENTS=ROOT/'budget_events.jsonl'
CAP=float(PROTO['budget']['cumulative_gpu_wall_cap_seconds'])

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def validate_zero_update_restart(auth, prior, *, startup_failure, ledger_sha, preflight_sha, binding_sha):
    if auth.get('status')!='USER_APPROVED_SINGLE_ZERO_UPDATE_RESTART':
        raise RuntimeError('explicit user restart approval missing')
    if (auth.get('seed'),auth.get('epochs'),auth.get('optimizer_attempt_cap'),auth.get('gpu_process_cap_seconds'))!=(2026,100,25000,36000):
        raise RuntimeError('restart approval would change frozen budget or recipe')
    if auth.get('prior_budget_ledger_sha256')!=ledger_sha:
        raise RuntimeError('ledger changed since explicit zero-update restart approval')
    if auth.get('preflight_sha256')!=preflight_sha or auth.get('execution_binding_sha256')!=binding_sha:
        raise RuntimeError('restart preflight/binding not explicitly approved')
    stages=prior.get('stages',[])
    if len(stages)!=1 or stages[0].get('exit_code')!=1 or stages[0].get('status')!='INCOMPLETE':
        raise RuntimeError('restart only covers the one failed first startup')
    if prior.get('optimizer_attempts')!=0 or prior.get('successful_updates')!=0:
        raise RuntimeError('cannot fresh-restart a trajectory that already updated')
    if startup_failure.get('model_initialized') is not False or startup_failure.get('optimizer_attempts')!=0:
        raise RuntimeError('failure did not occur before model/optimizer updates')
    spent=float(prior.get('cumulative_gpu_process_seconds',-1))
    if spent!=auth.get('prior_gpu_process_seconds') or spent<=0 or spent>=CAP:
        raise RuntimeError('restart would reset or invalidate spent GPU time')

def watchdog_stop_threshold(remaining_seconds):
    if remaining_seconds<=120:
        raise RuntimeError('not enough cumulative budget for a guarded GPU process')
    return remaining_seconds-120

def gpu_resource_gate():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    if not rows: raise RuntimeError('nvidia-smi returned no GPUs')
    row=rows[0].split(','); mem=int(row[2].strip().split()[0]); util=float(row[3].strip().split()[0])
    if mem>512 or util>0 or (apps and 'No running' not in apps): raise RuntimeError(f'GPU0 not idle: {row}; apps={apps}')
    free=int(subprocess.check_output(['df','-BG',str(BASE)],text=True).splitlines()[1].split()[3].rstrip('G'))
    if free<50: raise RuntimeError(f'free disk below frozen 50 GiB gate: {free}')
    return {'gpu0_uuid':row[1].strip(),'memory_mib':mem,'utilization_pct':util,'compute_apps':apps,'free_disk_gib':free}

def record_event(obj):
    with open(EVENTS,'a',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

def write_ledger(doc):
    tmp=LEDGER.with_suffix('.json.tmp'); tmp.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n'); os.replace(tmp,LEDGER)

def run_stage(stage, args):
    pre=json.loads(PREFLIGHT.read_text())
    if pre.get('status')!='PASS': raise RuntimeError('new scientific preflight is not PASS')
    actual_pre_sha=__import__('hashlib').sha256(PREFLIGHT.read_bytes()).hexdigest()
    if args.preflight_sha256!=actual_pre_sha: raise RuntimeError('provided preflight SHA does not match verified artifact')
    bind=json.loads(BINDING.read_text())
    if pre.get('execution_binding_sha256')!=__import__('hashlib').sha256(BINDING.read_bytes()).hexdigest(): raise RuntimeError('execution binding changed after preflight')
    if bind.get('protocol_sha256')!=sha(FREEZE/'protocol.json'):
        raise RuntimeError('scientific recipe changed after approval')
    for name,expected in bind['implementation_sha256s'].items():
        if sha(BRANCH/name)!=expected: raise RuntimeError('implementation drift: '+name)
    res=gpu_resource_gate()
    prior=json.loads(LEDGER.read_text()) if LEDGER.exists() else {'status':'PRELAUNCH','cumulative_gpu_process_seconds':0.0,'stages':[],'optimizer_attempts':0,'successful_updates':0,'internal_train_monitor_batches':0,'cap_seconds':CAP,'official_test_accessed':False}
    if prior.get('official_test_accessed') or prior.get('optimizer_attempts',0)>25000 or prior.get('cumulative_gpu_process_seconds',0)>CAP: raise RuntimeError('ledger violates frozen protocol; stop')
    spent=float(prior.get('cumulative_gpu_process_seconds',0))
    if stage=='train':
        if args.start_scientific_training is not True: raise RuntimeError('explicit --start-scientific-training required')
        restart_auth=None
        if prior.get('stages'):
            if not args.restart_authorization: raise RuntimeError('training already attempted; explicit user restart receipt required')
            auth_path=pathlib.Path(args.restart_authorization).resolve()
            if auth_path.parent!=ROOT/'restart_01': raise RuntimeError('restart receipt must be inside this run restart_01')
            restart_auth=json.loads(auth_path.read_text())
            failure=json.loads((ROOT/'startup_failure.json').read_text())
            if sha(ROOT/'startup_failure.json')!=restart_auth.get('startup_failure_sha256'):
                raise RuntimeError('original startup failure changed')
            validate_zero_update_restart(restart_auth,prior,startup_failure=failure,ledger_sha=sha(LEDGER),
                                         preflight_sha=actual_pre_sha,binding_sha=sha(BINDING))
            if list(ROOT.rglob('*.pth')) or list(ROOT.rglob('*.pt')) or (ROOT/'per_update.jsonl').exists():
                raise RuntimeError('scientific model/update artifacts exist; refusing fresh restart')
            # Atomic one-shot consumption prevents two conversations launching twice.
            with (auth_path.parent/'authorization_consumed.json').open('x',encoding='utf-8') as f:
                json.dump({'utc':datetime.now(timezone.utc).isoformat(),'authorization_sha256':sha(auth_path),
                           'prior_gpu_seconds':spent,'prior_attempts':0},f)
        if json.loads((ROOT/'training_completion.json').read_text()).get('status')=='TRAINING_COMPLETE' if (ROOT/'training_completion.json').exists() else False: raise RuntimeError('training already complete')
        model_dir=pathlib.Path(PROTO['paths']['nnUNet_results'])/'Dataset501_MAPLES/nnUNetTrainer_100epochs__nnUNetPlans_8GBtarget__2d'
        if model_dir.exists() or (ROOT/'initial_state.json').exists(): raise RuntimeError('new model output already exists; preserving, no restart')
        command=[PYTHON,'-B',str(BRANCH/'nnunet2d_scientific_train.py'),'--execute-scientific-run']
        # The child hard-stops training at 34,200s; outer watchdog retains the absolute 36,000s cap.
        deadline=CAP-spent
        if deadline<=0: raise RuntimeError('no cumulative GPU budget remains')
        phase_name='train_restart_01' if restart_auth is not None else 'train'
    else:
        if args.start_final_dev_inference is not True: raise RuntimeError('explicit --start-final-dev-inference required')
        tc=ROOT/'training_completion.json'
        if not tc.is_file() or json.loads(tc.read_text()).get('status')!='TRAINING_COMPLETE': raise RuntimeError('terminal 100epoch checkpoint is not verified; Dev remains sealed')
        if prior.get('optimizer_attempts')!=25000 or prior.get('successful_updates')!=25000 or prior.get('internal_train_monitor_batches')!=5000: raise RuntimeError('training counters incomplete')
        if (ROOT/'dev_predictions/prediction_export_verification.json').exists(): raise RuntimeError('endpoint inference already attempted; no auto re-inference')
        command=[PYTHON,'-B',str(BRANCH/'nnunet2d_scientific_infer_eval.py'),'--run-final-dev-inference']
        deadline=CAP-spent
        if deadline<=0: raise RuntimeError('GPU cap already consumed')
        phase_name='final_dev_inference'
    env=os.environ.copy()
    env.update({'PYTHONHASHSEED':'2026','CUDA_VISIBLE_DEVICES':'0','nnUNet_compile':'false','nnUNet_n_proc_DA':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
      'nnUNet_raw':PROTO['paths']['raw'],'nnUNet_preprocessed':PROTO['paths']['preprocessed'],'nnUNet_results':PROTO['paths']['nnUNet_results']})
    # Include the previous failed process in the child's own conservative time guard.
    started=time.monotonic(); env['SCIENTIFIC_PROCESS_START_MONOTONIC']=str(started-spent)
    env['SCIENTIFIC_PREFLIGHT_PATH']=str(PREFLIGHT)
    env['SCIENTIFIC_BINDING_PATH']=str(BINDING)
    stop_after=watchdog_stop_threshold(deadline)
    log=ROOT/(phase_name+'_supervisor.log')
    record_event({'event':'stage_start','stage':phase_name,'utc':datetime.now(timezone.utc).isoformat(),'gpu_preflight':res,'command':command,'deadline_seconds':deadline,'preflight_sha256':args.preflight_sha256,'execution_binding_sha256':pre['execution_binding_sha256']})
    stage_doc={'stage':phase_name,'started_utc':datetime.now(timezone.utc).isoformat(),'gpu_uuid':res['gpu0_uuid'],'command':command,'log':str(log),'cumulative_before_seconds':spent,'watchdog_deadline_seconds':deadline,'status':'RUNNING'}
    prior['status']='RUNNING'; prior['stages'].append(stage_doc); write_ledger(prior)
    with open(log,'x',encoding='utf-8') as lf:
        proc=subprocess.Popen(command,env=env,stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        stage_doc['pid']=proc.pid; write_ledger(prior)
        stop_reason=None
        try:
            while proc.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>=stop_after:
                    stop_reason='watchdog cumulative GPU process budget deadline reached'
                    try: os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                    break
                record_event({'event':'heartbeat','stage':phase_name,'utc':datetime.now(timezone.utc).isoformat(),'stage_elapsed_seconds':elapsed,'cumulative_projected_seconds':spent+elapsed,'child_pid':proc.pid})
                time.sleep(min(30,max(0.1,stop_after-elapsed)))
        except KeyboardInterrupt:
            stop_reason='supervisor interrupted; no automatic restart'
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
        code=proc.poll()
        if code is None:
            try: code=proc.wait(timeout=10)
            except subprocess.TimeoutExpired: code=-signal.SIGKILL
    elapsed=time.monotonic()-started
    spent_after=spent+elapsed
    stage_doc.update({'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_process_seconds':elapsed,'exit_code':code,'stop_reason':stop_reason,'status':'PASS' if code==0 and stop_reason is None else 'INCOMPLETE'})
    prior['cumulative_gpu_process_seconds']=spent_after
    prior['stages'][-1]=stage_doc
    if stage=='train':
        try:
            tr=json.loads((ROOT/'training_completion.json').read_text())
            prior['optimizer_attempts']=tr['attempts']; prior['successful_updates']=tr['successful_updates']; prior['internal_train_monitor_batches']=tr['internal_train_monitor_batches']
            prior['status']='TRAINING_COMPLETE' if code==0 and tr['attempts']==25000 and tr['successful_updates']==25000 and tr['internal_train_monitor_batches']==5000 else 'INCOMPLETE'
        except Exception:
            prior['optimizer_attempts']=sum(1 for _ in open(ROOT/'per_update.jsonl')) if (ROOT/'per_update.jsonl').exists() else 0
            prior['successful_updates']=None; prior['internal_train_monitor_batches']=sum(1 for _ in open(ROOT/'internal_train_monitor.jsonl')) if (ROOT/'internal_train_monitor.jsonl').exists() else 0
            prior['status']='INCOMPLETE'
    else:
        prior['status']='EVALUATION_COMPLETE' if code==0 and (ROOT/'dev_predictions/prediction_export_verification.json').exists() else 'INCOMPLETE'
    prior['updated_utc']=datetime.now(timezone.utc).isoformat(); prior['official_test_accessed']=False
    write_ledger(prior)
    record_event({'event':'stage_end','stage':phase_name,'utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':elapsed,'cumulative_gpu_process_seconds':spent_after,'exit_code':code,'stop_reason':stop_reason,'status':prior['status']})
    print(json.dumps({'status':prior['status'],'stage':phase_name,'elapsed_process_seconds':elapsed,'cumulative_gpu_process_seconds':spent_after,'exit_code':code,'stop_reason':stop_reason,'ledger':str(LEDGER),'log':str(log)},indent=2))
    return 0 if prior['status'] in ('TRAINING_COMPLETE','EVALUATION_COMPLETE') else 2

def main():
    global PREFLIGHT, BINDING
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('train','final_dev_inference'),required=True)
    ap.add_argument('--preflight-sha256',required=True)
    ap.add_argument('--start-scientific-training',action='store_true')
    ap.add_argument('--start-final-dev-inference',action='store_true')
    ap.add_argument('--preflight-file',default=str(PREFLIGHT))
    ap.add_argument('--binding-file',default=str(BINDING))
    ap.add_argument('--restart-authorization')
    a=ap.parse_args()
    PREFLIGHT=pathlib.Path(a.preflight_file).resolve()
    BINDING=pathlib.Path(a.binding_file).resolve()
    if not PREFLIGHT.is_relative_to(ROOT) or not BINDING.is_relative_to(ROOT):
        raise RuntimeError('preflight/binding must remain inside the existing scientific run')
    return run_stage(a.mode,a)
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(json.dumps({'status':'STOP','error':repr(e),'no_model_update_started':True},indent=2),file=sys.stderr)
        raise
