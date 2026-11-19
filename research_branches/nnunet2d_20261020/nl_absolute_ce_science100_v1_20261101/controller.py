import os,sys,time,subprocess,signal,secrets,fcntl,shutil
from common import *
from retinal_m0.maples_g1_safety import process_identity

def require_unused():
 if (GROUP/'SESSION.json').exists():raise RuntimeError('scientific launch already consumed; no automatic restart')
def resource_check():
 for i in range(16):
  vals=[v.strip() for v in subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
  procs=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if vals[0]!=GPU or procs or int(vals[1])>=512:raise RuntimeError('GPU0 unavailable; no eviction or switch')
  if int(vals[2])<10:break
  if i==15:raise RuntimeError('GPU0 utilization remains high')
  time.sleep(2)
 if shutil.disk_usage(GROUP).free<50*2**30:raise RuntimeError('disk below50GiB')

def run_child(seed,role):
 key=validate(full=True);stage=GROUP/'runs'/f'seed_{seed}';stage.mkdir(parents=True,exist_ok=True)
 if role in ['train','infer']:resource_check()
 nonce=secrets.token_hex(24)
 atomic(GROUP/'job_state.json',{'status':'RUNNING','active_seed':seed,'role':role,'release_sha256':key,'time':time.time()})
 env=dict(os.environ,RETINA_NL_SEED=str(seed),CUDA_VISIBLE_DEVICES=GPU if role!='analysis' else '',PYTHONHASHSEED=str(seed),NNUNET100_NONCE=nonce,OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',nnUNet_compile='false',nnUNet_n_proc_DA='1',MPLCONFIGDIR=str(stage/'matplotlib'))
 args=[sys.executable,'-B','-u',str(BRANCH/('analyze.py' if role=='analysis' else role+'.py'))]
 if role!='analysis':args+=['--authorized-train' if role=='train' else '--authorized-final-inference','--parent-pid',str(os.getpid()),'--nonce',nonce]
 with (stage/(role+'.log')).open('x') as logf:
  p=subprocess.Popen(args,env=env,cwd=CODE,stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)
  if role!='analysis':atomic(stage/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json'),{'worker':process_identity(p.pid),'controller':process_identity(),'nonce':nonce,'seed':seed,'role':role,'release_sha256':key},True)
  start=time.monotonic();last=start;cursor=None;maxseconds=180000 if role=='train' else 1800 if role=='infer' else 7200
  try:
   while p.poll() is None:
    elapsed=time.monotonic()-start
    f=stage/('budget.json' if role=='train' else 'inference_budget.json')
    data=read(f) if f.exists() else {}
    nowcursor=(data.get('attempts'),data.get('monitor_complete'),data.get('forwards'),sum(v.get('state')=='COMPLETE' for v in data.get('images',{}).values()))
    if nowcursor!=cursor:cursor=nowcursor;last=time.monotonic()
    atomic(GROUP/'heartbeat.json',{'time':time.time(),'seed':seed,'role':role,'pid':p.pid,'elapsed_seconds':elapsed,'counts':data,'no_automatic_restart':True})
    if elapsed>maxseconds:raise RuntimeError(role+' wall budget exceeded')
    if role!='analysis' and time.monotonic()-last>1800:raise RuntimeError('no progress for30minutes')
    if shutil.disk_usage(GROUP).free<50*2**30:raise RuntimeError('disk below50GiB')
    time.sleep(5)
   result={'seed':seed,'role':role,'pid':p.pid,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start}
   atomic(stage/(role+'_process_result.json'),result,True)
   if p.returncode:raise RuntimeError(str(result))
  except BaseException:
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
 return result

def main():
 validate(full=True)
 with (ROOT/'experiments/a14_gpu0.lock').open('a+') as global_lock,(GROUP/'RUN.lock').open('a+') as runlock:
  fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  require_unused();resource_check();atomic(GROUP/'SESSION.json',{'controller':process_identity(),'started_epoch':time.time(),'release_sha256':sha(GROUP/'SCIENCE_RELEASE.json'),'GPU':GPU,'serial_seeds':[2027,2028]},True)
  completed=[];failure=None
  try:
   for seed in [2027,2028]:completed.append(run_child(seed,'train'))
   for seed in [2027,2028]:completed.append(run_child(seed,'infer'))
   completed.append(run_child(2027,'analysis'))
  except BaseException as exc:failure=repr(exc)
  result={'status':'STOPPED_PENDING_SOL_REVIEW' if failure else 'COMPLETE_PENDING_SOL_REVIEW','error':failure,'completed_phases':completed,'automatic_restart':False,'Test_access':False}
  atomic(GROUP/'RESULT.json',result,True);atomic(GROUP/'job_state.json',result)
  # Preserve worker budget on controller death/timeouts, even if child could not finalize.
  for seed in [2027,2028]:
   p=GROUP/'runs'/f'seed_{seed}'/'budget.json'
   if p.exists():
    b=read(p)
    if b.get('pending'):
     atomic(p.with_name('UNRESOLVED_BUDGET_AT_STOP.json'),{'budget':b,'error':failure,'do_not_reset_or_resume':True},True)
  authority_entry('NL后续科学任务已退出，交Sol审查（自动事实记录）',MARKER+'；状态'+result['status']+'；异常'+str(failure)+'。详见原NL V2/science100_v1/RESULT.json和每seed账本。未自动重启、改lambda、扩组或访问Test；完成包括负结果，科研效果另审查。')
  print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
