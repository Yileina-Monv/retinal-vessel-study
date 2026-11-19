"""CPU-only handoff of the old queue after real GPU0 completion; never restart a worker."""
import sys,os,time,json,signal,subprocess,hashlib,importlib.util,shutil
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'nl_absolute_ce_science100_v1_20261101'
sys.path.insert(0,str(OLD))
from common import *
from retinal_m0.maples_g1_safety import process_identity
ORIG=GROUP;DELEGATED=ORIG/'parallel_gpu1_v1'
def terminal_exit(pid):
 text=Path(f'/proc/{pid}/stat').read_text();parts=text[text.rindex(')')+2:].split()
 return parts[0],int(parts[49])
def validate_completion(stage):
 t=read(stage/'training_completion.json');a=read(stage/'PAIRING_COMPLETE.json');b=t['counts']
 if t['status']!='TRAINING_COMPLETE' or t['attempts']!=25000 or t['internal_train_monitor_batches']!=5000 or b['success']+b['skip']!=25000 or b['fatal'] or b['unknown'] or a['status']!='PASS' or sha(t['checkpoint_final'])!=t['checkpoint_final_sha256']:raise RuntimeError('incomplete GPU0 evidence')
 return t

def prove_duplicate_rejection(stage,module=None):
 spec=importlib.util.spec_from_file_location('retire_old_queue',OLD/'controller.py');m=module
 if m is None:m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 key=m.validate(full=True);m.resource_check()
 with patch.object(m,'validate',lambda **k:key),patch.object(m,'resource_check',lambda:None),patch.object(m.subprocess,'Popen') as spawn:
  try:m.run_child(2028,'train')
  except FileExistsError as e:
   if Path(e.filename).resolve()!=(stage/'train.log').resolve() or spawn.call_count:raise RuntimeError('unexpected duplicate admission evidence')
   return {'type':type(e).__name__,'repr_original':repr(e),'str_with_actual_filename':str(e),'filename':str(e.filename),'duplicate_child_spawn_calls':spawn.call_count}
 raise RuntimeError('native duplicate admission did not reject')

def execute():
 release=read(DELEGATED/'QUEUE_HANDOFF_RELEASE.json')
 if release['guard_sha256']!=sha(HERE/'queue_handoff_guard.py') or release['CPU_sha256']!=sha(DELEGATED/'QUEUE_HANDOFF_CPU.json'):raise RuntimeError('handoff source/CPU binding drift')
 if (DELEGATED/'QUEUE_HANDOFF_SESSION.json').exists():raise RuntimeError('guard already consumed')
 atomic(DELEGATED/'QUEUE_HANDOFF_SESSION.json',{'guard':process_identity(),'time':time.time()},True)
 session=read(ORIG/'runs/seed_2027/TRAIN_SESSION.json');parent=session['controller'];worker=session['worker'];paused=False
 try:
  while True:
   if (DELEGATED/'RESULT.json').exists():return
   if process_identity(parent['pid'])!=parent:raise RuntimeError('old controller lost before handoff')
   b=read(ORIG/'runs/seed_2027/budget.json')
   if b['status']=='STOPPED' or (ORIG/'runs/seed_2027/training_failure.json').exists():return
   if b['attempts']>=24950:break
   atomic(DELEGATED/'queue_handoff_heartbeat.json',{'time':time.time(),'phase':'WAIT_LAST50_ATTEMPTS','GPU0_attempts':b['attempts'],'original_parent_still_running':True})
   time.sleep(5)
  os.kill(parent['pid'],signal.SIGSTOP);paused=True
  for _ in range(100):
   if terminal_exit(parent['pid'])[0] in ['T','t']:break
   time.sleep(.02)
  else:raise RuntimeError('old controller not quiescent')
  atomic(DELEGATED/'OLD_QUEUE_QUIESCED.json',{'time':time.time(),'controller':parent,'worker':worker,'GPU0_budget':b,'only_queue_parent_paused':True,'model_worker_not_signalled':True},True)
  last=time.monotonic();cursor=None
  start=read(ORIG/'job_state.json')['time']
  while True:
   b=read(ORIG/'runs/seed_2027/budget.json')
   if b['status']=='STOPPED':raise RuntimeError('scientific failure during quiescent handoff')
   state,code=terminal_exit(worker['pid'])
   if state=='Z':
    if code!=0:raise RuntimeError('GPU0 worker nonzero kernel exit '+str(code))
    break
   cur=(b['attempts'],b['monitor_complete'])
   if cur!=cursor:cursor=cur;last=time.monotonic()
   if time.monotonic()-last>1800 or time.time()-start>180000 or shutil.disk_usage(ORIG).free<50*2**30:raise RuntimeError('handoff runtime cap')
   if (DELEGATED/'RESULT.json').exists():raise RuntimeError('delegated task already stopped')
   atomic(DELEGATED/'queue_handoff_heartbeat.json',{'time':time.time(),'phase':'GPU0_FINISHING_WITH_ORIGINAL_PARENT_ALIVE','counts':b})
   time.sleep(2)
  complete=validate_completion(ORIG/'runs/seed_2027');validate(full=True)
  # Actual old admission function is executed with a Popen prohibition; captures real OSError filename.
  error=prove_duplicate_rejection(ORIG/'runs/seed_2028')
  phase={'seed':2027,'role':'train','pid':worker['pid'],'exit_code':0,'elapsed_seconds':time.time()-start,'exit_evidence':'/proc worker stateZ and kernel exit_code0, not inferred from checkpoint'}
  atomic(ORIG/'runs/seed_2027/train_process_result.json',phase,True)
  handoff={'status':'DELEGATED_QUEUE_RETIRED','error':error['type']+': '+error['str_with_actual_filename'],'original_exception_repr':error['repr_original'],'completed_phases':[phase],'writer':'authorized CPU queue handoff guard','original_parent_quiesced':True,'worker_exit_verified':True,'duplicate_admission':error,'new_GPU_calls':0,'automatic_restart':False,'Test_access':False,'time':time.time()}
  # A truthful takeover result: the old parent never handled the exception itself.
  atomic(DELEGATED/'QUEUE_HANDOFF_RECEIPT.json',handoff,True)
  atomic(ORIG/'RESULT.json',handoff,True);atomic(ORIG/'job_state.json',handoff)
  if process_identity(parent['pid'])!=parent:raise RuntimeError('parent identity changed before retirement')
  os.kill(parent['pid'],signal.SIGKILL);paused=False
  authority_entry('GPU0科学训练完成，旧串行队列已受控退役',MARKER+'。CPU交接guard仅在最后50attempt阶段暂停队列父进程，GPU0训练worker未暂停/重启；已核对完整25000/5000配对与终点SHA、内核退出码0。实际调用旧队列重复准入在Popen前抛FileExistsError，记录真实filename与str；原repr缺文件名，未修改运行中的训练代码。原RESULT由授权guard代写并明确writer/provenance，不冒称旧控制器自行处理。GPU1继续原seed2028，完成后由其控制器执行已批准的固定终点评价。证据parallel_gpu1_v1/QUEUE_HANDOFF_RECEIPT.json；无额外模型前向、更新或重跑。')
 except BaseException as e:
  atomic(DELEGATED/'QUEUE_HANDOFF_FAILURE.json',{'error':repr(e),'time':time.time(),'no_automatic_retry':True},True)
  if paused and process_identity(parent['pid'])==parent:os.kill(parent['pid'],signal.SIGCONT)
  raise
if __name__=='__main__':execute()
