import os, sys, time, secrets, signal, subprocess, traceback, fcntl
from support import *
def main():
    key=validate()
    with (ROOT/"experiments/a14_gpu1.lock").open("a+") as lock,(WORK/"RUN.lock").open("a+") as local:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (WORK/"budget.json").exists():raise RuntimeError("one-shot GPU budget already exists; no reset")
        values=[v.strip() for v in subprocess.check_output(["nvidia-smi","--id="+GPU,"--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")]
        processes=subprocess.check_output(["nvidia-smi","--id="+GPU,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        if values[0]!=GPU or processes or int(values[1])>=512 or int(values[2])>=10:raise RuntimeError("GPU1 occupied; no eviction")
        atomic(WORK/"budget.json",{"status":"RUNNING","reserved":0,"actual_forwards":0,"complete":0,"optimizer_updates":0,"pending":None,"release_sha256":key},True)
        nonce=secrets.token_hex(32);started=time.time()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                 nnUNet_compile="false",NL_REPAIR_NONCE=nonce)
        with (WORK/"worker.log").open("x") as logf:
            child=subprocess.Popen(["${RETINA_PROJECT_ROOT}/envs/nnunet2d_v2/bin/python","-B",str(HERE/"worker.py"),
                                    "--parent",str(os.getpid()),"--nonce",nonce],cwd=HERE,env=env,stdin=subprocess.DEVNULL,
                                    stdout=logf,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(WORK/"SESSION.json",{"controller_pid":os.getpid(),"worker_pid":child.pid,"nonce":nonce,"started":started,
                                   "GPU":GPU,"release_sha256":key},True)
        try:
            while child.poll() is None:
                if time.time()-started>1800:raise RuntimeError("30 minute GPU deadline")
                if sha(WORK/"GPU_RELEASE.json")!=key:raise RuntimeError("release revoked")
                if __import__("shutil").disk_usage(WORK).free<50*2**30:raise RuntimeError("disk below 50GiB")
                atomic(WORK/"heartbeat.json",{"elapsed":time.time()-started,"budget":read(WORK/"budget.json"),"worker_pid":child.pid})
                time.sleep(5)
            if child.returncode!=0:raise RuntimeError("GPU diagnostic worker failed exit="+str(child.returncode))
            r=read(WORK/"WORKER_RESULT.json");s=read(WORK/"budget.json")
            if r["status"]!="PASS" or s["complete"]!=4 or s["actual_forwards"]!=4 or s["optimizer_updates"] or s["pending"] is not None:
                raise RuntimeError("GPU completion counters mismatch")
            s["status"]="COMPLETE";atomic(WORK/"budget.json",s)
            atomic(WORK/"RESULT.json",{"status":"PASS_PENDING_SOL_REVIEW","elapsed_seconds":time.time()-started,
                                     "forwards":4,"optimizer_updates":0,"training_released":False},True)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
if __name__=="__main__":
    try:main()
    except BaseException as exc:
        if (WORK/"budget.json").exists():
            s=read(WORK/"budget.json");s["status"]="STOPPED";s["reason"]=repr(exc);atomic(WORK/"budget.json",s)
        atomic(WORK/"CONTROLLER_FAILURE.json",{"error":repr(exc),"traceback":traceback.format_exc(),"automatic_retry":False},True)
        raise
