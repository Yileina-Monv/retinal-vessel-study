import os,sys,time,secrets,subprocess,signal,traceback,contextlib,fcntl
from support import *
from common import authority_entry

def resource_check(seed):
 gpu=GPUS[seed]
 fields=[x.strip() for x in subprocess.check_output(["nvidia-smi","--id="+gpu,"--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")]
 procs=subprocess.check_output(["nvidia-smi","--id="+gpu,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
 if fields[0]!=gpu or procs or int(fields[1])>=512 or int(fields[2])>=10:raise RuntimeError("GPU occupied; no eviction/switch")

def main():
 key=validate(full=True);approval=read(WORK/"APPROVAL_AMENDMENT.json");children={};started={};last={};cursors={};done={}
 try:
  with contextlib.ExitStack() as stack:
   for path in [ROOT/"experiments/a14_gpu0.lock",ROOT/"experiments/a14_gpu1.lock",GROUP/"RUN.lock",WORK/"RUN.lock"]:
    f=stack.enter_context(path.open("a+"));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if (WORK/"job_state.json").exists():raise RuntimeError("resume run already exists; no automatic restart")
   for seed in [2027,2028]:
    if (stage(seed)/"budget.json").exists() or (stage(seed)/"train.log").exists():raise RuntimeError("resume tail is not unused")
    resource_check(seed)
   atomic(WORK/"job_state.json",{"status":"RUNNING","controller_pid":os.getpid(),"release_sha256":key,"seeds":[2027,2028]},True)
   for seed in [2027,2028]:
    nonce=secrets.token_hex(32)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[seed],PYTHONHASHSEED=str(seed),RETINA_NL_SEED=str(seed),
             OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             nnUNet_compile="false",nnUNet_n_proc_DA="1",NL_RESUME_NONCE=nonce)
    with (stage(seed)/"train.log").open("x") as logf:
     p=subprocess.Popen(["${RETINA_PROJECT_ROOT}/envs/nnunet2d_v2/bin/python","-B",str(HERE/"worker.py"),
                         "--seed",str(seed),"--parent",str(os.getpid()),"--nonce",nonce],cwd=HERE,env=env,
                         stdin=subprocess.DEVNULL,stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)
    children[seed]=p;started[seed]=time.time();last[seed]=started[seed];cursors[seed]=None
    atomic(stage(seed)/"RESUME_SESSION.json",{"seed":seed,"controller_pid":os.getpid(),"worker_pid":p.pid,
         "nonce":nonce,"started":started[seed],"GPU":GPUS[seed],"release_sha256":key},True)
   atomic(WORK/"SESSION.json",{"controller_pid":os.getpid(),"workers":{str(s):p.pid for s,p in children.items()},
         "started":started,"release_sha256":key},True)
   while len(done)<2:
    now=time.time();total=sum(approval["old_train_seconds"].values())
    rows={}
    for seed,p in children.items():
     elapsed=done[seed]["elapsed_seconds"] if seed in done else now-started[seed]
     total+=elapsed
     if seed in done:rows[str(seed)]=done[seed];continue
     if elapsed+approval["old_train_seconds"][str(seed)]>=180000:raise RuntimeError("per-seed cumulative 50h cap")
     progress_path=stage(seed)/"progress.json"
     current=read(progress_path) if progress_path.exists() else {"phase":"INITIALIZING"}
     cursor=(current.get("phase"),current.get("matched"),current.get("charged_attempts"),current.get("logical_completed"))
     if cursor!=cursors[seed]:last[seed]=now;cursors[seed]=cursor
     if now-last[seed]>1800:raise RuntimeError("30 minutes without progress seed "+str(seed))
     code=p.poll()
     if code is not None:
      if code!=0:raise RuntimeError("resume worker failure seed "+str(seed)+" exit "+str(code))
      result=read(stage(seed)/"training_completion.json")
      if result["status"]!="TRAINING_COMPLETE_FIXED_ENDPOINT_EVALUATION_PENDING":raise RuntimeError("terminal completion missing")
      done[seed]={"status":"TRAINING_COMPLETE","elapsed_seconds":elapsed,"completion":str(stage(seed)/"training_completion.json")}
     rows[str(seed)]={"pid":p.pid,"elapsed_seconds":elapsed,"progress":current}
    if not (WORK/"STARTUP_VERIFIED.json").exists():
     ready={}
     for ss in [2027,2028]:
      bp=stage(ss)/"budget.json";rp=stage(ss)/"RESTORE_VERIFIED.json"
      if bp.exists() and rp.exists():
       bb=read(bp);old=read(parent_stage(ss)/"budget.json")
       if bb["success"]>old["success"]:
        ready[str(ss)]={"budget":bb,"restore_evidence":str(rp),"worker_pid":children[ss].pid,
                        "successful_updates_since_resume":bb["success"]-old["success"]}
     if len(ready)==2:
      atomic(WORK/"STARTUP_VERIFIED.json",{"status":"BOTH_SEEDS_ACTUALLY_UPDATING","seeds":ready,
            "time":time.time(),"release_sha256":key,"old_failures_preserved":True},True)
      authority_entry("NL两条原轨迹已实际恢复参数更新（自动事实记录）",MARKER+"。GPU0 seed2027、GPU1 seed2028均已完成真实参数更新；实时证据原修复目录resume_science_v1/STARTUP_VERIFIED.json及heartbeat.json、各原seed/resume_amp_v1_20261102累计账本。原生数据队列从种子重建，全部历史前缀与中断批已匹配；模型/优化器/scaler/调度器/CPU及CUDA RNG和半轮统计恢复一致，随后逐批N0配对。每seed100epochs/25000逻辑输入、25001物理尝试，原中断不清零，ratio>2及原AMP/资源停止规则保留；脱离SSH后台运行，任一异常停止两条，不自动重跑。尚无固定终点Dev效果结论，Test封存。")
    if total>=360000:raise RuntimeError("100h cumulative combined cap")
    if shutil_disk_free()<50*2**30:raise RuntimeError("disk below 50GiB")
    if validate()!=key:raise RuntimeError("release revoked")
    atomic(WORK/"heartbeat.json",{"status":"RUNNING","time":now,"seeds":rows,"cumulative_train_seconds":total})
    time.sleep(5)
   atomic(WORK/"job_state.json",{"status":"TRAINING_COMPLETE_FIXED_ENDPOINT_EVALUATION_PENDING","controller_pid":os.getpid(),"completed":done,"release_sha256":key})
   atomic(WORK/"RESULT.json",{"status":"TRAINING_COMPLETE_FIXED_ENDPOINT_EVALUATION_PENDING","completed":done,"automatic_new_experiments":False,"Test_access":False},True)
   authority_entry("NL两条恢复训练完成，固定终点评价待执行（自动事实记录）",MARKER+"。两条原科学轨迹达到100epochs、各25000逻辑输入/25001累计物理尝试（保留原1次中断）；具体成功更新/AMP跳步、N0配对及终点SHA见各原seed/resume_amp_v1_20261102/training_completion.json与PAIRING_COMPLETE.json。原Loss/lambda与冻结评价不变，尚未产生新Dev终点指标。下一步仅按既有批准固定终点评价与Sol审查，不增加训练组或Test访问。")
 except BaseException as exc:
  if (WORK/"job_state.json").exists():
   state=read(WORK/"job_state.json");state["status"]="STOPPED";state["reason"]=repr(exc);atomic(WORK/"job_state.json",state)
  for seed,p in children.items():
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
  for seed,p in children.items():
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  atomic(WORK/"FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc(),"completed":done,"automatic_restart":False},True)
  authority_entry("NL恢复执行停止，保留累计账本交Sol审查（自动事实记录）",MARKER+"。错误"+repr(exc)+"。证据原修复目录resume_science_v1/FAILURE.json及各seed/resume_amp_v1_20261102。旧中断与本轮累计预算均保留，不自动重启、清零、改Loss/lambda或访问Test。")
  raise

def shutil_disk_free():
 import shutil
 return shutil.disk_usage(WORK).free
if __name__=="__main__":main()
