"""One detached train/final-inference/CPU-report job, shared GPU0 lock, no restart."""
import os,sys,json,time,signal,subprocess,secrets,shutil
from pathlib import Path
from science100_gate import STAGE,CODE,BRANCH,PHASE,GPU,protocol,validate,resources,sha,update_status
from fast_observer import atomic,Journal,AmpLimits
from progress_watchdog import ProgressWatchdog,train_progress_token,inference_progress_token
sys.path.insert(0,str(CODE))
from retinal_m0.maples_g1_safety import exclusive_lock,process_identity,owner_alive

def run_child(role,key):
    # This is the only spawn path for GPU processes. Each role may be attempted once.
    session_file=STAGE/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json')
    if session_file.exists():raise RuntimeError('stage already attempted; never restart automatically')
    nonce=secrets.token_hex(24);env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONHASHSEED='2026',NNUNET100_NONCE=nonce,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',nnUNet_compile='false',nnUNet_n_proc_DA='1',MPLCONFIGDIR=str(STAGE/'matplotlib'),nnUNet_raw=str(STAGE.parent/'raw'),nnUNet_preprocessed=str(STAGE.parent/'preprocessed'),nnUNet_results=str(STAGE/'nnunet_results'))
    env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    script=BRANCH/('science100_train.py' if role=='train' else 'science100_infer.py');flag='--authorized-train' if role=='train' else '--authorized-final-inference';started=time.monotonic()
    with (STAGE/(role+'.log')).open('x',encoding='utf-8') as out:
        p=subprocess.Popen([sys.executable,'-u','-B',str(script),flag,'--parent-pid',str(os.getpid()),'--nonce',nonce],env=env,cwd=CODE,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        owner=process_identity(p.pid);atomic(session_file,{'controller':process_identity(),'worker':owner,'nonce':nonce,'binding_sha256':key,'role':role})
        update_status(execution_started=True,active_role=role,controller_pid=os.getpid(),worker_pid=p.pid)
        watch=ProgressWatchdog(1800);last_warning=-float('inf')
        try:
            while p.poll() is None:
                # Monitoring faults only warn. No NAS mtime/wall-clock comparison may kill training.
                try:
                    path=STAGE/('budget.json' if role=='train' else 'inference_budget.json')
                    if path.exists():
                        b=json.loads(path.read_text());token=train_progress_token(b) if role=='train' else inference_progress_token(b)
                        observed=watch.observe(token,time.monotonic())
                        if observed['stop'] and time.monotonic()-last_warning>=300:
                            print('PROGRESS_WARNING_ONLY',role,observed,flush=True);last_warning=time.monotonic()
                    else:observed={'waiting_for_budget':True}
                    atomic(STAGE/'job_state.json',{'state':'RUNNING','role':role,'controller':process_identity(),'worker':owner,'elapsed_role_seconds':time.monotonic()-started,'normal_progress_wall_limit':None,'watchdog_policy':'real counters + monotonic; warning only, never SIGTERM','watchdog':observed,'time':time.time()})
                except Exception as e:
                    print('MONITOR_IO_WARNING_ONLY',role,repr(e),flush=True)
                time.sleep(5)
        except BaseException:
            if owner_alive(owner):
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    result={'role':role,'elapsed_process_seconds':time.monotonic()-started,'exit_code':p.returncode,'no_restart':True};atomic(STAGE/(role+'_process_result.json'),result)
    if p.returncode:raise RuntimeError(role+' child stopped '+str(p.returncode))
    return result

def main():
    key=validate(full=True);protocol()
    with exclusive_lock(STAGE/'SCIENCE_RUNTIME.lock'),exclusive_lock(CODE.parents[1]/'experiments/a14_gpu0.lock'):
        if (STAGE/'TRAIN_SESSION.json').exists() or (STAGE/'authorization_consumed.json').exists():raise RuntimeError('fresh scientific trajectory already attempted')
        gpu=resources();j=Journal(STAGE/'budget.json',key,100,250,50,AmpLimits(5,25))
        if j.read()!=json.loads((STAGE/'budget_migration.json').read_text())['migrated_budget']:raise RuntimeError('not the CPU-approved1000-update migrated ledger')
        atomic(STAGE/'authorization_consumed.json',{'binding_sha256':key,'release_sha256':sha(STAGE/'SCIENCE_RELEASE.json'),'controller':process_identity(),'GPU0':gpu,'time':time.time()})
        try:
            train=run_child('train',key);validate(full=True)
            completed=json.loads((STAGE/'training_completion.json').read_text())
            if completed['attempts']!=25000 or completed['internal_train_monitor_batches']!=5000 or completed['counts']['fatal'] or completed['counts']['unknown']:raise RuntimeError('training incomplete; Dev not allowed')
            # Training has exited and freed its CUDA context. Recheck the owned GPU before final inference.
            resources();infer=run_child('infer',key)
        except BaseException as exc:
            j.stop(repr(exc));atomic(STAGE/'failure.json',{'error':repr(exc),'counts':j.read(),'automatic_restart':False});atomic(STAGE/'job_state.json',{'state':'FAILED_STOP','error':repr(exc),'time':time.time()});update_status(state='FAILED_STOP',GPU_released=False,worker_pid=None,controller_pid=None,run_state='FAILED_STOP',next='Sol/user review saved failure and costs; no automatic retry');raise
    # GPU lock can be released; the remaining work is CPU-only.
    update_status(state='CONSUMED',GPU_released=False,worker_pid=None,active_role='CPU_FINALIZATION',charged_attempts=25000,next='Automatic CPU endpoint reconciliation and report; then Sol scientific review, no further GPU work')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        from science100_analysis import analyze
        endpoint=analyze();atomic(STAGE/'completion.json',{'status':'COMPLETE_PENDING_SOL_SCIENTIFIC_REVIEW','training':train,'inference':infer,'counts':j.read(),'endpoint_path':str(STAGE/'analysis/endpoint_100epochs.json'),'endpoint_sha256':sha(STAGE/'analysis/endpoint_100epochs.json'),'readout':endpoint['readout'],'Test_accessed':False})
        atomic(STAGE/'job_state.json',{'state':'COMPLETE_PENDING_SOL_REVIEW','time':time.time(),'extra_GPU_work':0});update_status(run_state='COMPLETE_PENDING_SOL_REVIEW',training_complete=True,controller_pid=None,active_role=None,scientific_review_pending=True,result_path=str(STAGE/'analysis/endpoint_100epochs.json'),next='Sol scientific review of fixed final endpoint; no auto seed/budget/threshold expansion')
        backup=STAGE/'management_before_auto_completion';backup.mkdir()
        for name in ['STATUS.md','项目全局状态与研究决策.md']:shutil.copy2(CODE/name,backup/name)
        means=endpoint['absolute_means'];d=endpoint['metrics']['dice']['NNUNET-A0'];header=f'### nnU-Net AMP修订科学100epochs已完成，待Sol科研复盘\n\n25000尝试、成功{j.read()["success"]}、受控跳步{j.read()["skip"]}、5000 Train内部监控；唯一final模型完成Dev28/.5评价。普通Dice {100*means["NNUNET"]["dice"]:.4f}% vs A0 {100*means["A0"]["dice"]:.4f}%，差{100*d["estimate"]:+.4f}pp，配对95%区间[{100*d["lower"]:+.4f},{100*d["upper"]:+.4f}]pp；完整保护成本另报，不凭Dice或Recall单项宣称整体成功。后台CPU程序应用冻结规则并归档，未冒称Sol已审阅。结果 '+str(STAGE/'analysis/report.md')+'。单seed重复Dev/流程比较，Test封存，无自动后续GPU。\n\n'
        header+='接续说明：保留V4前1000次，V5只运行剩余24000次；增强队列重新建立，非逐位精确的未中断运行。模型、优化器、AMP、LR与主RNG恢复凭据已保存。\n\n'
        for path in [CODE/'STATUS.md',CODE/'项目全局状态与研究决策.md']:
            old=path.read_text();first,_,rest=old.partition('\n');tmp=path.with_name(path.name+'.nnunet100_auto.pending')
            with tmp.open('x',encoding='utf-8') as f:f.write(first+'\n\n'+header+rest.lstrip('\n'));f.flush();os.fsync(f.fileno())
            if path.read_text()!=old:raise RuntimeError('management changed concurrently')
            os.replace(tmp,path)
    except BaseException as exc:
        atomic(STAGE/'CPU_finalization_failure.json',{'error':repr(exc),'GPU_work_complete':True,'extra_GPU_work':0});update_status(run_state='GPU_COMPLETE_CPU_FINALIZATION_FAILED',scientific_review_pending=True,next='Only CPU reconciliation repair; never rerun model inference automatically');raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true',required=True);p.parse_args();main()
