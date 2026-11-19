"""Detached conditional engineering -> CPU acceptance -> fresh scientific2000 -> CPU report."""
import os,sys,time,json,secrets,subprocess,signal,shutil,datetime
from scse_gate import STAGE as S,CODE,BRANCH as B,OUT,GPU,PHASE,PROTOCOL_SHA,protocol,validate_release,resources,sha,authority,update_status
from retinal_m0.maples_g1_safety import atomic_json,read_json,exclusive_lock,process_identity,owner_alive
from scse_core import Budget,verify_checkpoint

def header_record(header,folder):
    backup=S/folder;backup.mkdir(exist_ok=False)
    for p in [CODE/'STATUS.md',CODE/'项目全局状态与研究决策.md']:
        for _ in range(5):
            old=p.read_bytes();(backup/p.name).write_bytes(old);first,_,rest=old.decode().partition('\n');tmp=p.with_name(p.name+'.scse_record.'+str(os.getpid())+'.pending');tmp.write_text(first+'\n\n'+header+'\n\n'+rest.lstrip('\n'))
            if p.read_bytes()!=old:tmp.unlink();continue
            os.replace(tmp,p);break
        else:raise RuntimeError('concurrent record updates')
def run_child(role,track,start,end,key,wall_start):
    name=f'{role}_{track}_{start:04d}';session_path=S/('SESSION_'+name+'.json')
    if session_path.exists():raise RuntimeError('segment already attempted; no automatic retry')
    resources(idle=True);nonce=secrets.token_hex(24);env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONHASHSEED='2026',SCSE_NONCE=nonce,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128',PYTHONPATH=str(B)+os.pathsep+str(CODE))
    args={'role':role,'track':track,'start':start,'end':end};cmd=[sys.executable,'-u','-B',str(B/'scse_worker.py'),'--role',role,'--track',track,'--start',str(start),'--end',str(end),'--parent-pid',str(os.getpid()),'--nonce',nonce];began=time.monotonic()
    with (S/(name+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owner=process_identity(p.pid)
        session={'controller':process_identity(),'worker':owner,'nonce':nonce,'binding_sha256':key,'role':role,'args':args,'engineering_wall_start':wall_start};atomic_json(session_path,session);atomic_json(S/'CURRENT_CHILD.json',session)
        update_status(active_role=role,active_track=track,controller_pid=os.getpid(),worker_pid=p.pid,execution_started=True)
        try:
            while p.poll() is None:
                if role=='engineering' and time.monotonic()-wall_start>=900:raise RuntimeError('fixed engineering15minute cap')
                try:atomic_json(S/'job_state.json',{'state':'RUNNING','role':role,'track':track,'time':time.time(),'elapsed_child_seconds':time.monotonic()-began,'heartbeat_timeout_action':'warning only; no heartbeat-based termination'})
                except Exception as e:print('MONITOR_WRITE_WARNING',repr(e),flush=True)
                time.sleep(3)
        except BaseException:
            if owner_alive(owner):
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    result={'segment':name,'exit_code':p.returncode,'elapsed_process_seconds':time.monotonic()-began,'worker':owner};atomic_json(S/('PROCESS_'+name+'.json'),result)
    if p.returncode:raise RuntimeError('worker failed '+name+' code '+str(p.returncode))
    deadline=time.monotonic()+15
    while True:
        try:resources(idle=True);break
        except Exception:
            if time.monotonic()>=deadline:raise
            time.sleep(1)
    return result

def review_engineering(key,processes):
    from retinal_m0.state import state_digest
    ledger=Budget(S/'engineering_budget.json',key,global_cap=32,track_caps={'CONT':16,'SPLIT':16},forward_cap=32,gradient_cap=32);counts=ledger.snapshot()
    if any(counts[k]!=32 for k in ['charged_events','verified_successes','optimizer_call_starts','forward_calls','gradient_calls']) or any(counts[k] for k in ['failed_events','uncertain_events','failed_compute']) or counts['compute_pending'] or counts['status']!='READY':raise RuntimeError('engineering32 incomplete/failed')
    states={}
    for track in ['CONT','SPLIT']:states[track]=verify_checkpoint(S/'engineering'/track/'last.pt',expected_step=16,binding_sha=key,run_dir=S/'engineering'/track)
    if states['CONT']['digests']!=states['SPLIT']['digests']:raise RuntimeError('continuous vs new-process8+8 full-state mismatch')
    rows={m:[json.loads(x) for x in (S/'engineering'/m/'train.jsonl').read_text().splitlines()] for m in ['CONT','SPLIT']}
    keys=['pair','loss','module_gradient_L2','preclip_total_L2','amp_scale_before','amp_scale_after','state_digests']
    if len(rows['CONT'])!=16 or len(rows['SPLIT'])!=16 or any(any(a[k]!=b[k] for k in keys) for a,b in zip(rows['CONT'],rows['SPLIT'])):raise RuntimeError('engineering replay loss/gradient/RNG/state differs')
    diagnostic=read_json(S/'engineering_diagnostics.json')
    if diagnostic['status']!='PASS' or diagnostic['forwards']!=6 or diagnostic['updates']!=0 or not (S/'engineering/SPLIT/restore_0008.json').exists():raise RuntimeError('identity/AMP/restore diagnostics incomplete')
    names=list(rows['CONT'][0]['module_gradient_L2']);learned={n:any((r['module_gradient_L2'][n] or 0)>0 for r in rows['CONT']) for n in names}
    if not all(learned.values()):raise RuntimeError('a module parameter never received a nonzero gradient within16 updates')
    wall=sum(x['elapsed_process_seconds'] for x in processes)
    if wall>=900:raise RuntimeError('engineering GPU child wall cap exceeded')
    atomic_json(S/'engineering_acceptance_rows.json',rows['CONT']);review={'status':'ENGINEERING_ACCEPTED_CONDITIONAL_SCIENCE_ELIGIBLE','binding_sha256':key,'real_updates':32,'train_forwards':32,'backwards':32,'zero_update_diagnostic_forwards':6,'CONT16_vs_SPLIT8_new_process8':'all per-update values and full saved states exact','all_module_parameters_received_nonzero_gradient':learned,'peak_reserved_MiB':max(r['GPU_peak_reserved_MiB'] for rr in rows.values() for r in rr),'GPU_child_wall_seconds':wall,'Test_accessed':False,'Dev_inference':0,'scientific_effect_claim':False};atomic_json(S/'SOL_ENGINEERING_REVIEW.json',review);return review

def main():
    key=validate_release('engineering',full_sources=True)
    with exclusive_lock(S/'RUNTIME.lock'),exclusive_lock(CODE.parents[1]/'experiments/a14_gpu1.lock'):
        if (S/'authorization_consumed.json').exists():raise RuntimeError('approved run already attempted')
        atomic_json(S/'authorization_consumed.json',{'binding_sha256':key,'controller':process_identity(),'GPU1':resources(idle=True),'time':time.time()})
        try:
            started=time.monotonic();processes=[run_child('engineering',track,start,end,key,started) for track,start,end in protocol()['engineering']['segments']]
            review=review_engineering(key,processes)
            rel={'phase':PHASE,'role':'science','protocol_sha256':PROTOCOL_SHA,'binding_sha256':key,'CPU_review_sha256':sha(S/'CPU_REVIEW.json'),'GPU':GPU,'updates':2000,'Test_allowed':False,'auto_restart':False,'engineering_review_sha256':sha(S/'SOL_ENGINEERING_REVIEW.json')}
            # Conditional user authorization already includes this exact fresh run. No automatic group expansion.
            atomic_json(S/'SCIENCE_RELEASE.json',rel);update_status(state='SCIENCE_RELEASED',GPU_released=True,release_sha256=sha(S/'SCIENCE_RELEASE.json'),engineering_complete=True,engineering_review_path=str(S/'SOL_ENGINEERING_REVIEW.json'),next='One freshM0 SCSE seed2026 fixed2000, original sampling/loss; fixed0.5 endpoint, no auto additional experiment')
            header_record('### 单scSE工程32更新通过，按用户条件授权启动GPU1科学2000步\n\n6次FP32/AMP初始恒等与旁路前向、连续16及新进程8+8恢复全值一致，累计32更新、零Dev工程推理，所有新参数后续获得非零梯度。CPU自动严格复核真实日志/检查点后独立生成专用科学release；这只证明工程通过，不是Dice提升。从原M0重新开始，唯一seed2026/2000步，BCE+Dice/原8000输入不变，终点0.5，GPU0 nnU-Net继续。Test封存，所有结果待Sol科研复盘。证据 '+str(S/'SOL_ENGINEERING_REVIEW.json'),'management_before_science_start')
            validate_release('science',full_sources=True);run_child('science','SCSE',0,2000,key,time.monotonic())
        except BaseException as e:
            atomic_json(S/'failure.json',{'error':repr(e),'time':time.time(),'no_automatic_retry':True,'engineering_budget':read_json(S/'engineering_budget.json'),'science_budget':read_json(S/'science_budget.json')});update_status(state='FAILED_STOP',GPU_released=False,worker_pid=None,controller_pid=None,run_state='FAILED_STOP',next='Review actual failure/costs; no implicit reset/retry or new experiment');raise
    update_status(state='CONSUMED',GPU_released=False,worker_pid=None,active_role='CPU_ANALYSIS',training_complete=True,completed_steps=2000,charged_updates=2000)
    try:
        from scse_analysis import analyze
        endpoint=analyze();readout=endpoint['readout'];atomic_json(S/'completion.json',{'status':'COMPLETE_PENDING_SOL_SCIENTIFIC_REVIEW','endpoint':str(OUT/'analysis/endpoint_2000.json'),'readout':readout,'Test_accessed':False})
        update_status(controller_pid=None,active_role=None,run_state='COMPLETE_PENDING_SOL_REVIEW',result_path=str(OUT/'analysis/endpoint_2000.json'),scientific_review_pending=True,next='Sol reviews fixed2000 result and protection; no automatic extra seed/module/threshold/steps')
        d=endpoint['metrics']['dice']['SCSE-A0'];means=endpoint['absolute_means'];header_record(f'### 单scSE seed2026固定2000步已完成，待Sol科研复盘\n\nGPU1训练2000成功与十次固定Dev评价结束，CPU复核原A0和28概率/逐图分母/NA及配对区间。普通Dice SCSE {100*means["SCSE"]["dice"]:.4f}% vs A0 {100*means["A0"]["dice"]:.4f}%，差{100*d["estimate"]:+.4f}pp，95%区间[{100*d["lower"]:+.4f},{100*d["upper"]:+.4f}]pp；完整保护成本另见报告。自动事实核验与冻结规则判读不冒充Sol已科研审阅，单seed重复Dev不是独立验证。不自动扩组/seed/预算/阈值，Test封存。报告 '+str(OUT/'analysis/report.md'),'management_before_auto_completion')
    except BaseException as e:
        atomic_json(S/'CPU_finalization_failure.json',{'error':repr(e),'extra_GPU_calls':0});update_status(run_state='GPU_COMPLETE_CPU_ANALYSIS_FAILED',controller_pid=None,next='CPU-only report repair, never rerun GPU inference automatically');raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true',required=True);p.parse_args();main()
