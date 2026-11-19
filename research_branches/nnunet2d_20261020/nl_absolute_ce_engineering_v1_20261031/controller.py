import os,sys,time,subprocess,signal,shutil,fcntl
from common import *
def main():
 key=validate()
 sys.path.insert(0,str(C))
 from retinal_m0.maples_g1_safety import exclusive_lock
 with exclusive_lock(R/'experiments/a14_gpu0.lock'),exclusive_lock(O/'RUN.lock'):
  if (O/'SESSION.json').exists():raise RuntimeError('One-shot run already consumed')
  values=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  procs=subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if values[0].strip()!=GPU or int(values[1])>=512 or int(values[2])>=10 or procs:raise RuntimeError('GPU0 busy; no killing/switching')
  if shutil.disk_usage(O).free<50*2**30:raise RuntimeError('Disk low')
  ledger=Ledger(O/'budget.json',key);ledger.init();atomic(O/'SESSION.json',{'pid':os.getpid(),'GPU':GPU,'started_epoch':time.time(),'release_sha256':key},True)
  start=time.monotonic();failure=None
  try:
   for role in ORDER:
    validate();ledger.start(role)
    seed='2028' if '2028' in role else '2027'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONHASHSEED=seed,NL_CONTROLLER_PID=str(os.getpid()),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',nnUNet_compile='false',nnUNet_n_proc_DA='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(O/'matplotlib'))
    with (O/(role+'.log')).open('x') as f:
     p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name('worker.py')),role],env=env,cwd=C,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     atomic(O/(role+'_session.json'),{'pid':p.pid,'role':role,'controller':os.getpid()},True)
     while p.poll() is None:
      elapsed=time.monotonic()-start
      atomic(O/'heartbeat.json',{'role':role,'elapsed_seconds':elapsed,'pid':p.pid,'budget':ledger.get()})
      if elapsed>=7200:
       os.killpg(p.pid,signal.SIGTERM)
       try:p.wait(timeout=10)
       except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
       raise RuntimeError('Cumulative 2hour cap')
      time.sleep(2)
    if p.returncode:raise RuntimeError(role+' worker stopped, see failure/log; no automatic retry')
   assert ledger.get()['diag_reserved']==40 and ledger.get()['update_reserved']==128
  except BaseException as e:
   failure=repr(e);ledger.stop(failure)
  result={'status':'STOPPED_PENDING_SOL_REVIEW' if failure else 'COMPLETE_PENDING_SOL_REVIEW','failure':failure,'budget':ledger.get(),'elapsed_seconds':time.monotonic()-start,'science_started':False,'Test_access':False}
  atomic(O/'RESULT.json',result,True);print(result,flush=True)
  # Authority append is factual only, never a scientific release.
  import datetime
  with (C/'.deep_supervision_status_write.lock').open('a+') as lk:
   fcntl.flock(lk,fcntl.LOCK_EX)
   backup=R/'docs/.continuity_backups'/('nl_gpu_engineering_exit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'));backup.mkdir()
   entry='### NL有限GPU工程已退出，待Sol审查（自动事实回写）\n\n状态：'+result['status']+'；累计诊断预留'+str(result['budget']['diag_reserved'])+'/40，更新尝试预留'+str(result['budget']['update_reserved'])+'/128，成功'+str(result['budget']['success'])+'，安全AMP跳步'+str(result['budget']['skips'])+'。失败原因：'+str(failure)+'。详见nl_absolute_ce_cpu_v2_alltargets_20261031/engineering_v1/RESULT.json及各role日志。未自动重试/减lambda/扩大预算，未开始科学训练或Test。当前交回Sol科研审查，正式训练仍未授权。\n'
   for n in ['STATUS.md','项目全局状态与研究决策.md']:
    p=C/n;old=p.read_bytes();(backup/n).write_bytes(old);head,rest=old.decode().split('\n',1);tmp=p.with_name(p.name+'.gpu_exit');tmp.write_text(head+'\n\n'+entry+'\n'+rest);os.replace(tmp,p)
if __name__=='__main__':main()
