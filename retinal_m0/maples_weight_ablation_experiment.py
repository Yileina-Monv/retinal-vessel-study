"""New isolated execution path for the MAPLES weight ablation."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch.utils.data import DataLoader,Dataset
import retinal_m0.maples_core_experiment as parent
from retinal_m0.maples_core import CorePatches
from retinal_m0.maples_weight_ablation import weighted_loss_with_contributions,contribution_records,reweight_base_sample
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save,restore_rng,rng_state
ROOT=Path("${RETINA_PROJECT_ROOT}");CODE=Path(__file__).resolve().parents[1]
PROTO=ROOT/"docs/t0_tw_weight_ablation_20261017/protocol.json";MANIFEST=ROOT/"docs/t0_tw_weight_ablation_20261017/source_manifest.json"
OUT=ROOT/"experiments/t0_tw_weight_ablation_20261017";PROTO_SHA="a2de7e7bb0c3bee647bc9ab8a7e41ffc0d5dc1a0433d29d0a34b0e8986d1f4b7"
SEEDS=(2026,2027,2028);ALPHAS=(1.5,3.0);PAIR=ROOT/"experiments/t0_tw_core_20261016"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def source_binding():
 files=[Path(__file__),CODE/"retinal_m0/maples_weight_ablation.py",CODE/"retinal_m0/maples_weight_ablation_experiment.py",CODE/"retinal_m0/maples_weight_ablation_gate.py",CODE/"tests/test_maples_weight_ablation.py",CODE/"retinal_m0/maples_core.py",CODE/"retinal_m0/maples_core_experiment.py",CODE/"retinal_m0/model.py",CODE/"retinal_m0/state.py",CODE/"retinal_m0/inference.py",CODE/"retinal_m0/maples_evaluate.py",CODE/"retinal_data/maples_native.py",CODE/"retinal_data/maples_fov.py",CODE/"retinal_data/fives.py",CODE/"configs/development_v1.json",CODE/"releases/development_v1/normalization.json"]
 return {str(p.relative_to(CODE)):sha(p) for p in files}
def binding():
 if sha(PROTO)!=PROTO_SHA:raise RuntimeError("Frozen ablation protocol hash mismatch")
 d=json.loads(PROTO.read_text());m=json.loads(MANIFEST.read_text());protected=m["protected_sha256"]
 cert=OUT/"implementation_gate/attempt_01/protected_manifest_check.json"
 if cert.is_file():
  c=json.loads(cert.read_text())
  if c.get("status")!="passed" or c.get("source_manifest_sha256")!=sha(MANIFEST) or c.get("protected_file_count")!=len(protected):raise RuntimeError("Prior protected-manifest check is not reusable")
 else:
  bad=[]
  for path,want in protected.items():
   if not Path(path).is_file() or sha(path)!=want:bad.append(path)
  if bad:raise RuntimeError("Protected manifest mismatch: "+", ".join(bad[:8]))
  cert.parent.mkdir(parents=True,exist_ok=True)
  writej(cert,{"status":"passed","source_manifest_sha256":sha(MANIFEST),"protected_file_count":len(protected),"training_started":False})
 if sha(PAIR/"execution_binding.json")!=d["parent"]["execution_binding_sha256"]:raise RuntimeError("Parent binding mismatch")
 return {"protocol_path":str(PROTO),"protocol_sha256":sha(PROTO),"source_manifest_sha256":sha(MANIFEST),"protected_file_count":len(protected),
 "protected_files_verified":True,"protected_verification_record_sha256":sha(OUT/"implementation_gate/attempt_01/protected_manifest_check.json"),"sources":source_binding(),"parent_binding_sha256":sha(PAIR/"execution_binding.json"),
 "m0_sha256":sha(ROOT/"checkpoints/m0_10000_20261008/last.pt"),"data_manifest_sha256":sha(ROOT/"data/maples_handoff_20261008/data_manifest.csv"),
 "normalization_sha256":sha(CODE/"releases/development_v1/normalization.json"),"torch":str(torch.__version__),"cuda":str(torch.version.cuda),
 "gpu0_uuid":"${GPU_UUID}","seeds":list(SEEDS),"new_alphas":list(ALPHAS),"steps":2000,"draws":8000}
def binding_sha():return sha(OUT/"execution_binding.json")
def require_binding():
 p=OUT/"execution_binding.json"
 if not p.is_file():raise RuntimeError("Missing execution binding")
 current=binding()
 old=json.loads(p.read_text())
 if current!=old:raise RuntimeError("Execution binding/source/protected data changed")
 v=OUT/"implementation_gate/verification.json"
 if not v.is_file() or not json.loads(v.read_text()).get("all_passed") or json.loads(v.read_text()).get("execution_binding_sha256")!=sha(p):raise RuntimeError("Implementation gate is not passed for current binding")
 return sha(p)
def make_config(seed,alpha):
 c=parent.config(seed,"TW",False)
 c.update(kind="MAPLES_T0_TW_weight_ablation_v1",loss="alpha_weighted_BCE_plus_original_unweighted_masked_Dice",
  phase="approved_exploratory_weight_ablation",method="TW_WEIGHT_ABLATION",seed=seed,alpha=float(alpha),
  weight_definition="FOV * (1+(alpha-1)*A), U remains weight one",max_steps=2000,save_every=200,
  protocol_sha256=PROTO_SHA,execution_binding_id="t0_tw_weight_ablation_20261017")
 return c
class AblationDraws(Dataset):
 def __init__(self,c,alpha):self.base=CorePatches(c,"TW");self.alpha=float(alpha)
 def __len__(self):return len(self.base)
 def __getitem__(self,i):return reweight_base_sample(self.base[i],self.alpha)
def expected_pair(seed):
 path=PAIR/f"runs/seed_{seed}/T0/pairing.jsonl"
 rec=[]
 with path.open(encoding="utf-8") as f:
  for line in f:
   r=json.loads(line)
   for j,draw in enumerate(r["draws"]):
    rec.append((int(draw),r["filenames"][j],r["geometry"][j],r["input_sha256"][j]))
 if len(rec)!=8000 or [x[0] for x in rec]!=list(range(8000)):raise RuntimeError(f"Old seed {seed} pairing coverage invalid")
 other=(PAIR/f"runs/seed_{seed}/TW/pairing.jsonl").read_bytes()
 if other!=path.read_bytes():raise RuntimeError(f"Old paired logs differ for seed {seed}")
 return rec
def assert_batch(batch,seed,step,records):
 for slot,draw in enumerate(batch["draw"].tolist()):
  want=records[draw];got=(int(draw),batch["filename"][slot],batch["geometry"][slot].tolist(),batch["input_sha256"][slot])
  if got!=want:raise RuntimeError(f"Pairing mismatch seed={seed} draw={draw}: {got} != {want}")
def initialize(seed,c):
 parent.setup(seed);mc=parent.modelcfg();model=UNet(mc["widths"],mc["group_norm_groups"])
 opt=torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"],betas=tuple(c["betas"]),eps=c["eps"],foreach=False)
 scaler=torch.amp.GradScaler("cuda",init_scale=c["amp_initial_scale"])
 return model,opt,scaler
def old_initial(seed):
 r={}
 for method in ("T0","TW"):
  p=PAIR/f"runs/seed_{seed}/{method}/summary.json";d=json.loads(p.read_text())
  if d.get("status")!="passed" or d.get("completed_steps")!=2000 or d.get("next_global_draw")!=8000:raise RuntimeError(f"Old {seed}/{method} incomplete")
  r[method]=d
 keys=("initial_model","initial_optimizer","initial_scaler","initial_rng")
 if any(r["T0"][k]!=r["TW"][k] for k in keys):raise RuntimeError(f"Old initial states differ for seed {seed}")
 return r["T0"]
def initial_check(seed,c):
 model,opt,scaler=initialize(seed,c)
 # load only original M0 model, then compare initialization state hashes to the old paired run
 raw=torch.load(c["source_m0"],map_location="cpu",weights_only=True);model.load_state_dict(raw["model"],strict=True);del raw
 from retinal_m0.state import state_digest
 statehash=lambda x:hashlib.sha256(state_digest(x).encode()).hexdigest()
 got={"initial_model":statehash(model.state_dict()),"initial_optimizer":statehash(opt.state_dict()),"initial_scaler":statehash(scaler.state_dict()),"initial_rng":statehash(rng_state())}
 want=old_initial(seed)
 if any(got[k]!=want[k] for k in got):raise RuntimeError(f"Fresh initial state differs seed={seed}: {got} vs old summary")
 return got
def worker(config_path,out,resume=False):
 config_path=Path(config_path);out=Path(out);c=json.loads(config_path.read_text());seed=int(c["seed"]);alpha=float(c["alpha"])
 if seed not in SEEDS or alpha not in ALPHAS:raise ValueError("Unapproved seed/alpha")
 bsha=require_binding();runbind={"execution_binding_sha256":bsha,"config_sha256":sha(config_path),"source_m0_sha256":c["source_m0_sha256"],"alpha":alpha,"seed":seed}
 if not out.resolve().is_relative_to((OUT/"runs").resolve()):raise ValueError("Output outside approved ablation root")
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Must expose only GPU0")
 lockpath=ROOT/"experiments/a14_gpu0.lock"
 with lockpath.open("a+") as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);snap=parent.gpu();started=time.perf_counter()
  if resume and not (out/"last.pt").is_file():raise RuntimeError("Resume checkpoint missing")
  if not resume:
   out.mkdir(parents=True,exist_ok=False);writej(out/"config.json",c);writej(out/"run_binding.json",runbind)
  torch.cuda.set_device(0);model,opt,scaler=initialize(seed,c);model=model.cuda().train()
  ds=AblationDraws(c,alpha);records=expected_pair(seed);step=cursor=0;elapsed=0.;init={}
  if resume:
   saved=torch.load(out/"last.pt",map_location="cpu",weights_only=True)
   if saved["config"]!=c or saved["binding"]!=runbind:raise RuntimeError("Checkpoint binding mismatch")
   model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
   step=int(saved["step"]);cursor=int(saved["next_global_draw"]);elapsed=float(saved["elapsed_seconds"])
   init={k:saved[k] for k in ("initial_model","initial_optimizer","initial_scaler","initial_rng")}
   from retinal_m0.maples_t0 import align_log
   align_log(out/"train.jsonl",step);align_log(out/"pairing.jsonl",step)
   tail=__import__("retinal_m0.maples_weight_ablation",fromlist=["align_contribution_log"]).align_contribution_log(out/"bce_contributions.jsonl",step,cursor)
   restore_rng(saved["rng"]);del saved
   writej(out/f"restore_at_{step:06d}.json",{"step":step,"next_global_draw":cursor,"contribution_log_alignment":tail,"passed":True})
  else:
   raw=torch.load(c["source_m0"],map_location="cpu",weights_only=True)
   if raw["step"]!=10000:raise RuntimeError("Wrong original M0 step")
   model.load_state_dict(raw["model"],strict=True);del raw
   from retinal_m0.state import state_digest
   statehash=lambda x:hashlib.sha256(state_digest(x).encode()).hexdigest()
   init={"initial_model":statehash(model.state_dict()),"initial_optimizer":statehash(opt.state_dict()),"initial_scaler":statehash(scaler.state_dict()),"initial_rng":statehash(rng_state())}
   want=old_initial(seed)
   if any(init[k]!=want[k] for k in init):raise RuntimeError("Initial state no longer matches old paired trajectory")
  loader=DataLoader(ds,batch_size=4,sampler=range(cursor,8000),num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(seed))
  summary={"status":"running","seed":seed,"alpha":alpha,"target":2000,"start_step":step,"next_global_draw":cursor,"gpu":snap,"run_binding":runbind,**init}
  try:
   torch.cuda.reset_peak_memory_stats()
   with (out/"train.jsonl").open("a" if resume else "x",encoding="utf-8") as log,(out/"pairing.jsonl").open("a" if resume else "x",encoding="utf-8") as pair,(out/"bce_contributions.jsonl").open("a" if resume else "x",encoding="utf-8") as blog:
    for batch in loader:
     if time.perf_counter()-started>=c["safe_stop_seconds"]:break
     assert_batch(batch,seed,step+1,records)
     x,y,f,w=[batch[k].cuda(non_blocking=True) for k in ("image","mask","fov","weight")]
     if x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or f.shape!=y.shape or w.shape!=y.shape:raise RuntimeError("Batch shape mismatch")
     before=model.head.weight.detach().clone();retry=0
     while True:
      opt.zero_grad(set_to_none=True)
      with torch.autocast("cuda",dtype=torch.float16):z=model(x)
      loss,bce_image,dice_image,parts=weighted_loss_with_contributions(z,y,f,w,alpha)
      if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
      scaler.scale(loss).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.parameters(),c["clip_norm"],error_if_nonfinite=False)
      if torch.isfinite(gn):break
      scaler.update();retry+=1
      if retry>8:raise RuntimeError("AMP overflow limit")
     scaler.step(opt);scaler.update();torch.cuda.synchronize()
     if torch.equal(before,model.head.weight):raise RuntimeError("No successful parameter update")
     step+=1;draws=batch["draw"].tolist();cursor+=4
     rows=contribution_records(parts,list(batch["filename"]),seed,alpha,step,draws,batch["geometry"].tolist(),list(batch["input_sha256"]),retry,bce_image,dice_image)
     bce_batch=float(bce_image.mean().detach());dice_batch=float(dice_image.mean().detach());total=float(loss.detach())
     if not np.isclose(sum(np.mean([r["groups"][g]["C"] for r in rows]) for g in ("A_V1","A_V0","O_V1","O_V0")),bce_batch,rtol=1e-5,atol=1e-7):raise RuntimeError("BCE contribution reconciliation failed")
     if not np.isclose(bce_batch+dice_batch,total,rtol=1e-5,atol=1e-7):raise RuntimeError("Total loss reconciliation failed")
     row={"step":step,"global_draw_start":draws[0],"BCE_batch":bce_batch,"Dice_batch":dice_batch,"actual_total_loss":total,"gradient_norm":float(gn),"amp_scale":scaler.get_scale(),"amp_retries":retry,"draws":draws,"filenames":list(batch["filename"]),"geometry":batch["geometry"].tolist(),"input_sha256":list(batch["input_sha256"]),"parameters_updated":True}
     log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
     pair.write(json.dumps({k:row[k] for k in ("step","draws","filenames","geometry","input_sha256")})+"\n");pair.flush()
     for q in rows:blog.write(json.dumps(q,allow_nan=False)+"\n")
     blog.flush()
     if step%200==0:
      payload={"schema_version":1,"step":step,"next_global_draw":cursor,"epoch":cursor//440,"position_in_epoch":cursor%440,"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"config":c,"binding":runbind,**init,"elapsed_seconds":elapsed+time.perf_counter()-started,"pid":os.getpid()}
      atomic_save(payload,out/"last.pt");atomic_save(payload,out/f"step_{step:06d}.pt")
     if step%25==0:print(f"alpha={alpha} seed={seed} step={step}/2000",flush=True)
   if step and not (out/f"step_{step:06d}.pt").is_file():
    payload={"schema_version":1,"step":step,"next_global_draw":cursor,"epoch":cursor//440,"position_in_epoch":cursor%440,"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"config":c,"binding":runbind,**init,"elapsed_seconds":elapsed+time.perf_counter()-started,"pid":os.getpid()}
    atomic_save(payload,out/"last.pt");atomic_save(payload,out/f"step_{step:06d}.pt")
   final=torch.load(out/"last.pt",map_location="cpu",weights_only=True)
   summary.update(status="passed" if step==2000 else "checkpointed",completed_steps=step,next_global_draw=cursor,model_digest=parent.statehash(final["model"]),optimizer_digest=parent.statehash(final["optimizer"]),scaler_digest=parent.statehash(final["scaler"]),rng_digest=parent.statehash(final["rng"]),checkpoint_sha256=sha(out/"last.pt"),elapsed_seconds=elapsed+time.perf_counter()-started,max_allocated_mib=torch.cuda.max_memory_allocated()/1048576,max_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
   writej(out/"summary.json",summary);return summary
  except Exception:
   summary.update(status="failed",error=traceback.format_exc(),completed_steps=step,next_global_draw=cursor);writej(out/"summary.json",summary);raise
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--train",action="store_true");g.add_argument("--evaluate",action="store_true")
 p.add_argument("--config");p.add_argument("--output");p.add_argument("--resume",action="store_true")
 a=p.parse_args()
 if a.prepare:
  print(json.dumps(binding(),ensure_ascii=False,indent=2));return
 if not a.config or not a.output:raise SystemExit("--config and --output required")
 if a.train:print(json.dumps(worker(a.config,a.output,a.resume),ensure_ascii=False,indent=2))
 else:raise SystemExit("Evaluation is implemented by the bound gate/report runner.")
if __name__=="__main__":main()
