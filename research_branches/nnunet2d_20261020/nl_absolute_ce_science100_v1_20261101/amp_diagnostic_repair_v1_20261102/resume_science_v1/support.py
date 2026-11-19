import os,sys,json,time,fcntl,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from paths import OUT,GROUP,PARENT
from common import ROOT,CODE,DATA_BASE,BASELINE,read,sha,atomic,log
WORK=OUT/"resume_science_v1"
MARKER="NL_RESUME_TWO_SEED_RELEASED_20261102"
GPUS={2027:"${GPU_UUID}",2028:"${GPU_UUID}"}
def stage(seed):return GROUP/"runs"/("seed_"+str(seed))/"resume_amp_v1_20261102"
def parent_stage(seed):return GROUP/"runs"/("seed_"+str(seed))
_cache={"time":0.}
def validate(full=False):
 rpath=WORK/"RESUME_RELEASE.json"
 if not rpath.exists():raise RuntimeError("independent resume release missing")
 r=read(rpath);a=read(WORK/"APPROVAL_AMENDMENT.json")
 if r.get("scope")!="RESUME_ORIGINAL_NL_TWO_SEEDS" or a["new_physical_cap_per_seed"]!=25001 or a["extra_total"]!=2 or a["logical_training_batches_per_seed"]!=25000 or a["epochs"]!=100 or a["lambda"]!=0.0018152656553044105 or a["Test_allowed"]:raise RuntimeError("resume authorization scope drift")
 if r["amendment_sha256"]!=sha(WORK/"APPROVAL_AMENDMENT.json") or r["CPU_review_sha256"]!=sha(WORK/"CPU_REVIEW.json") or read(WORK/"CPU_REVIEW.json")["status"]!="PASS":raise RuntimeError("resume CPU/approval binding failed")
 if MARKER not in (CODE/"STATUS.md").read_text()[:16000]:raise RuntimeError("STATUS resume release missing")
 now=time.monotonic()
 if full or now-_cache["time"]>30:
  for p,h in r["runtime_sources"].items():
   if sha(p)!=h:raise RuntimeError("runtime source drift "+p)
  _cache["time"]=now
 if full:
  for p,h in r["immutable_sources"].items():
   if sha(p)!=h:raise RuntimeError("resume input/state drift "+p)
 return sha(rpath)
def runtime_authorizer(seed,parent,nonce,key):
 s=stage(seed);session=s/"RESUME_SESSION.json"
 for _ in range(100):
  if session.exists():break
  time.sleep(.05)
 identity=read(session);session_sha=sha(session)
 def check():
  if os.getppid()!=parent or os.getpid()!=identity["worker_pid"] or identity["controller_pid"]!=parent or identity["seed"]!=seed or identity["nonce"]!=nonce or os.environ.get("NL_RESUME_NONCE")!=nonce or sha(session)!=session_sha:raise RuntimeError("resume process/session mismatch")
  if read(WORK/"job_state.json")["status"]!="RUNNING" or validate()!=key:raise RuntimeError("resume release revoked")
  a=read(WORK/"APPROVAL_AMENDMENT.json")
  if time.time()-identity["started"]+a["old_train_seconds"][str(seed)]>=180000:raise RuntimeError("cumulative 50h seed budget")
 check();return check
def progress(seed,**kw):atomic(stage(seed)/"progress.json",dict(seed=seed,time=time.time(),**kw))
