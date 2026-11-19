"""GPU1 delegation: preserves GPU0 worker and uses native exclusive log as duplicate barrier."""
import os,sys,time,subprocess,signal,secrets,fcntl,shutil
from common import *
from retinal_m0.maples_g1_safety import process_identity
ORIG=ORIGINAL_GROUP

def resource_check():
 vals=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')]
 apps=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if vals[0]!=GPU or apps or int(vals[1])>=512 or int(vals[2])>=10:raise RuntimeError('GPU1 busy; no eviction/switch')
 if shutil.disk_usage(GROUP).free<50*2**30:raise RuntimeError('disk below50GiB')
def complete2027():
 p=ORIG/'runs/seed_2027/training_completion.json';a=ORIG/'runs/seed_2027/PAIRING_COMPLETE.json'
 if not p.exists() or not a.exists():return False
 t=read(p);b=t['counts']
 if t['status']!='TRAINING_COMPLETE' or t['attempts']!=25000 or t['internal_train_monitor_batches']!=5000 or b['fatal'] or b['unknown'] or b['success']+b['skip']!=25000 or read(a)['status']!='PASS':raise RuntimeError('invalid2027completion')
 return True

def expected_retirement(doc):
 phases=doc.get('completed_phases',[])
 return complete2027() and len(phases)==1 and phases[0]['seed']==2027 and phases[0]['role']=='train' and phases[0]['exit_code']==0 and 'FileExistsError' in str(doc.get('error')) and str(ORIG/'runs/seed_2028/train.log') in str(doc['error'])
def check_original():
 f=ORIG/'runs/seed_2027/training_failure.json';p=ORIG/'runs/seed_2027/budget.json'
 if f.exists() or read(p)['status']=='STOPPED':raise RuntimeError('GPU0 original scientific run stopped')
 r=ORIG/'RESULT.json'
 if r.exists():
  doc=read(r)
  if not expected_retirement(doc):raise RuntimeError('unexpected original controller termination '+str(doc))
  return 'EXPECTED_QUEUE_RETIREMENT'
 session=read(ORIG/'SESSION.json')
 if process_identity(session['controller']['pid'])!=session['controller']:raise RuntimeError('original controller identity lost')
 return 'ORIGINAL_RUNNING'
def stop_original_if_active(reason):
 p=ORIG/'runs/seed_2027/budget.json'
 if p.exists() and read(p)['phase']!='complete':
  s=read(ORIG/'runs/seed_2027/TRAIN_SESSION.json');pid=s['worker']['pid']
  if process_identity(pid)==s['worker']:
   atomic(GROUP/'STOP_ORIGINAL_FOR_SHARED_FAILURE.json',{'reason':reason,'worker':s['worker'],'time':time.time()},True)
   os.kill(pid,signal.SIGTERM)

def run_child(seed,role):
 key=validate(full=True);stage=GROUP/'runs'/f'seed_{seed}';stage.mkdir(parents=True,exist_ok=True)
 if role!='analysis':resource_check()
 nonce=secrets.token_hex(24);atomic(GROUP/'job_state.json',{'status':'RUNNING','active_seed':seed,'role':role,'release_sha256':key,'time':time.time()})
 env=dict(os.environ,RETINA_NL_SEED=str(seed),CUDA_VISIBLE_DEVICES=GPU if role!='analysis' else '',PYTHONHASHSEED=str(seed),NNUNET100_NONCE=nonce,OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',nnUNet_compile='false',nnUNet_n_proc_DA='1',MPLCONFIGDIR=str(stage/'matplotlib'))
 args=[sys.executable,'-B','-u',str(BRANCH/('analyze.py' if role=='analysis' else role+'.py'))]
 if role!='analysis':args+=['--authorized-train' if role=='train' else '--authorized-final-inference','--parent-pid',str(os.getpid()),'--nonce',nonce]
 with (stage/(role+'.log')).open('x') as logf:
  # The open(x) above is exactly the old queue's pre-spawn barrier, no fake checkpoint or ledger.
  if role=='train':atomic(GROUP/'DELEGATION_CLAIM.json',{'seed':seed,'GPU':GPU,'canonical_stage':str(stage),'exclusive_log':str(stage/'train.log'),'controller':process_identity(),'key':key,'time':time.time()},True)
  p=subprocess.Popen(args,env=env,cwd=CODE,stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)
  if role!='analysis':atomic(stage/('TRAIN_SESSION.json' if role=='train' else 'INFERENCE_SESSION.json'),{'worker':process_identity(p.pid),'controller':process_identity(),'nonce':nonce,'seed':seed,'role':role,'release_sha256':key},True)
  start=time.monotonic();last=start;cursor=None;cap=180000 if role=='train' else 1800 if role=='infer' else 7200
  try:
   while p.poll() is None:
    original_status=check_original();elapsed=time.monotonic()-start
    f=stage/('budget.json' if role=='train' else 'inference_budget.json');b=read(f) if f.exists() else {}
    cur=(b.get('attempts'),b.get('monitor_complete'),b.get('forwards'),sum(v.get('state')=='COMPLETE' for v in b.get('images',{}).values()))
    if cur!=cursor:cursor=cur;last=time.monotonic()
    atomic(GROUP/'heartbeat.json',{'time':time.time(),'seed':seed,'role':role,'pid':p.pid,'elapsed_seconds':elapsed,'counts':b,'GPU0_status':original_status,'no_automatic_restart':True})
    if elapsed>cap:raise RuntimeError(role+' wall limit')
    if role!='analysis' and time.monotonic()-last>1800:raise RuntimeError('no progress30minutes')
    if shutil.disk_usage(GROUP).free<50*2**30:raise RuntimeError('disk below50GiB')
    time.sleep(5)
   result={'seed':seed,'role':role,'pid':p.pid,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start};atomic(stage/(role+'_process_result.json'),result,True)
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
 with (ROOT/'experiments/a14_gpu1.lock').open('a+') as lock,(GROUP/'RUN.lock').open('a+') as runlock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (GROUP/'SESSION.json').exists() or (ORIG/'runs/seed_2028/train.log').exists() or (ORIG/'runs/seed_2028/budget.json').exists():raise RuntimeError('seed2028 already claimed; no restart')
  check_original();resource_check();atomic(GROUP/'SESSION.json',{'controller':process_identity(),'time':time.time(),'GPU':GPU,'delegated_training_seed':2028,'release_sha256':sha(GROUP/'SCIENCE_RELEASE.json')},True)
  completed=[];failure=None
  try:
   completed.append(run_child(2028,'train'))
   wait_start=time.monotonic()
   while not (ORIG/'RESULT.json').exists():
    check_original()
    if time.monotonic()-wait_start>180000:raise RuntimeError('GPU0 wait exceeded original50h cap')
    atomic(GROUP/'heartbeat.json',{'time':time.time(),'phase':'WAIT_GPU0_COMPLETION','GPU1_released_idle':True});time.sleep(5)
   if not expected_retirement(read(ORIG/'RESULT.json')):raise RuntimeError('original queue not safely retired')
   atomic(GROUP/'ORIGINAL_QUEUE_RETIREMENT.json',{'status':'EXPECTED_DELEGATION_RETIREMENT','original_result':read(ORIG/'RESULT.json'),'scientific_failure':False,'no_duplicate_seed2028_process':True},True)
   for seed in [2027,2028]:completed.append(run_child(seed,'infer'))
   completed.append(run_child(2027,'analysis'))
  except BaseException as e:
   failure=repr(e);stop_original_if_active(failure)
  result={'status':'STOPPED_PENDING_SOL_REVIEW' if failure else 'COMPLETE_PENDING_SOL_REVIEW','error':failure,'completed_phases':completed,'scope':'GPU1 delegated seed2028, existing2027 preserved','automatic_restart':False,'Test_access':False}
  atomic(GROUP/'RESULT.json',result,True);atomic(GROUP/'job_state.json',result)
  for seed in [2027,2028]:
   p=ORIG/'runs'/f'seed_{seed}'/'budget.json'
   if p.exists() and read(p).get('pending'):atomic(GROUP/f'UNRESOLVED_SEED_{seed}.json',{'budget':read(p),'do_not_reset':True},True)
  authority_entry('NL双卡委派任务退出，交Sol审查',MARKER+'。GPU1委派状态'+result['status']+'；错误'+str(failure)+'。证据原NL/science100_v1/parallel_gpu1_v1/RESULT.json。旧串行控制器在2027科学完成后因2028独占日志退出属于预期退役，仅在严格核对后作此分类；任何科学故障仍停止。原两seed预算/评价不变，不自动重跑或访问Test。')
  print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
