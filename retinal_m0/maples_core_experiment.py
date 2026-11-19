"""Protocol-bound MAPLES T0/TW engineering, training, and evaluation entry."""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, os, random, subprocess, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from retinal_data.maples_native import NativeMaplesDataset
from retinal_data.maples_fov import candidate_fov
from retinal_m0.inference import tiled_probability
from retinal_m0.maples_core import CorePatches, draw_geometry, weighted_masked_loss
from retinal_m0.maples_evaluate import evaluate_arrays, load_mask
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save, restore_rng, rng_state, state_digest

ROOT=Path("${RETINA_PROJECT_ROOT}")
CODE=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/t0_tw_core_20261016"
PROTOCOL=ROOT/"docs/t0_tw_core_protocol_20261016/protocol.json"
BASE=ROOT/"experiments/a15_2_z0_t0_calibration_20261015/train_config.json"
DEV=ROOT/"experiments/a15_2_z0_t0_calibration_20261015/dev_manifest.json"
PROTO_SHA="bc0aefe0d2d8d5ab6d8c950b33993d2ba824e8fd344773d3dfeb07d269317cdd"
SEEDS=(2026,2027,2028); METHODS=("T0","TW")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def statehash(x):return hashlib.sha256(state_digest(x).encode()).hexdigest()
def writej(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def protocol():
 d=json.loads(PROTOCOL.read_text())
 if sha(PROTOCOL)!=PROTO_SHA or d.get("status")!="FROZEN_USER_APPROVED_BEFORE_FORMAL_RUNS":raise RuntimeError("Frozen protocol hash/status mismatch")
 return d
def rows():
 ds=NativeMaplesDataset("${RETINA_PROJECT_ROOT}/data/maples_handoff_20261008",split="train")
 if ds.split_counts!={"train":110,"dev":28,"official_test":60}:raise RuntimeError("Frozen MAPLES counts differ")
 return ds.rows
def config(seed,method,engineering=False):
 if seed not in SEEDS or method not in METHODS:raise ValueError("Unapproved seed/method")
 c=json.loads(BASE.read_text())
 c.update(kind="MAPLES_T0_TW_formal_core_v1",
  loss="weighted_BCE_plus_original_unweighted_masked_Dice" if method=="TW" else "unit_weight_BCE_plus_original_unweighted_masked_Dice",
  phase="engineering_tw_gate" if engineering else "formal_paired_2000_fixed_endpoint",
  method=method,seed=seed,max_steps=50 if engineering else 2000,save_every=25 if engineering else 200,
  official_test=False,train_sample_filenames=[r["filename"] for r in rows()],
  protocol_sha256=PROTO_SHA,execution_binding_id="t0_tw_core_20261016")
 return c
def protected():
 d=protocol();out={}
 for p,want in d["source_evidence"]["protected_files"].items():
  got=sha(p)
  if got!=want:raise RuntimeError("Protected source changed: "+p)
  out[p]=got
 return out
def binding():
 d=protocol();root=ROOT/"data/maples_handoff_20261008";m0=ROOT/"checkpoints/m0_10000_20261008/last.pt"
 if sha(m0)!=d["source_evidence"]["inherited_training_binding"]["source_m0_sha256"]:raise RuntimeError("M0 source changed")
 files=[Path(__file__),CODE/"retinal_m0/maples_core.py",CODE/"retinal_m0/maples_t0.py",CODE/"retinal_m0/model.py",CODE/"retinal_m0/state.py",CODE/"retinal_m0/inference.py",CODE/"retinal_m0/maples_evaluate.py",CODE/"retinal_m0/objectives.py",CODE/"retinal_data/maples_native.py",CODE/"retinal_data/maples_fov.py",CODE/"retinal_data/fives.py",CODE/"retinal_m1/maps.py",CODE/"retinal_m0/maples_core_gate.py",CODE/"configs/development_v1.json",CODE/"tests/test_maples_core.py"]
 return {"protocol_path":str(PROTOCOL),"protocol_sha256":sha(PROTOCOL),
  "code_root":str(CODE),"sources":{str(p.relative_to(CODE)):sha(p) for p in files},
  "protected_source_hashes":protected(),
  "data":{"manifest_sha256":sha(root/"data_manifest.csv"),"split_sha256":sha(root/"split_record.json"),
  "normalization_sha256":sha(CODE/"releases/development_v1/normalization.json"),
  "original_m0_sha256":sha(m0)},
  "environment":{"python":sys.version,"torch":str(torch.__version__),"cuda":str(torch.version.cuda),"numpy":np.__version__,
  "gpu0_uuid":"${GPU_UUID}"},
  "scope":{"seeds":list(SEEDS),"methods":list(METHODS),"updates_each":2000,"draws_each":8000,"dev_images":28,"official_test_accessed":False}}
def bind_sha():
 p=OUT/"execution_binding.json";return sha(p) if p.is_file() else None
def require_binding(gate=False):
 p=OUT/"execution_binding.json"
 if not p.is_file() or json.loads(p.read_text())!=binding():raise RuntimeError("Current source/data differs from bound implementation")
 if not gate:
  v=OUT/"engineering/verification.json"
  if not v.is_file() or not json.loads(v.read_text()).get("all_passed") or json.loads(v.read_text()).get("execution_binding_sha256")!=sha(p):raise RuntimeError("Engineering gate has not passed for this binding")
 return sha(p)
def gpu():
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0")
 raw=subprocess.check_output(["nvidia-smi","--id=0","--query-gpu=index,uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=10).strip().split(",")
 u=raw[1].strip()
 if u!="${GPU_UUID}":raise RuntimeError("GPU0 UUID differs")
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True,timeout=10)
 if any(x.split(",")[0].strip()==u for x in apps.splitlines() if x.strip()) or int(raw[2])>=512 or int(raw[3])>=10:raise RuntimeError("GPU0 busy; no process was stopped")
 return {"index":0,"uuid":u,"memory_mib":int(raw[2]),"util_percent":int(raw[3])}
def setup(seed):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def modelcfg():
 c=json.loads((CODE/"configs/development_v1.json").read_text())
 if c["widths"]!=[32,64,128,256,512] or c["group_norm_groups"]!=8:raise RuntimeError("M0 architecture differs")
 return c
def worker(cfgpath,out,stop=None,resume=False,gate=False):
 cfgpath=Path(cfgpath);out=Path(out);c=json.loads(cfgpath.read_text())
 if c["method"] not in METHODS or c["seed"] not in SEEDS:raise ValueError("Unapproved run settings")
 if gate and (c["phase"]!="engineering_tw_gate" or c["method"]!="TW" or c["max_steps"]!=50):raise ValueError("Engineering worker scope mismatch")
 bsha=require_binding(gate=gate);runbind={"execution_binding_sha256":bsha,"config_sha256":sha(cfgpath),"source_m0_sha256":c["source_m0_sha256"]}
 target=int(stop if stop is not None else c["max_steps"])
 if target<1 or target>c["max_steps"] or (resume and target<=0):raise ValueError("Invalid target")
 if not out.resolve().is_relative_to(OUT.resolve()):raise ValueError("Run output outside authorized root")
 lockpath=ROOT/"experiments/a14_gpu0.lock";out.parent.mkdir(parents=True,exist_ok=True)
 with lockpath.open("a+") as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);snap=gpu();started=time.perf_counter()
  if resume and not (out/"last.pt").is_file():raise ValueError("Missing checkpoint")
  if not resume:
   out.mkdir(parents=True,exist_ok=False);writej(out/"config.json",c);writej(out/"run_binding.json",runbind)
  torch.cuda.set_device(0);setup(c["seed"]);ds=CorePatches(c,c["method"]);mc=modelcfg()
  model=UNet(mc["widths"],mc["group_norm_groups"]).cuda().train()
  opt=torch.optim.AdamW(model.parameters(),lr=c["learning_rate"],weight_decay=c["weight_decay"],betas=tuple(c["betas"]),eps=c["eps"],foreach=False)
  scaler=torch.amp.GradScaler("cuda",init_scale=c["amp_initial_scale"])
  step=cursor=0;elapsed0=0.;init={}
  if resume:
   saved=torch.load(out/"last.pt",map_location="cpu",weights_only=True)
   if saved["config"]!=c or saved["binding"]!=runbind:raise RuntimeError("Checkpoint binding mismatch")
   model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
   step,cursor=int(saved["step"]),int(saved["next_global_draw"]);elapsed0=float(saved["elapsed_seconds"])
   init={k:saved[k] for k in ("initial_model","initial_optimizer","initial_scaler","initial_rng")}
   from retinal_m0.maples_t0 import align_log
   align_log(out/"train.jsonl",step);align_log(out/"pairing.jsonl",step);restore_rng(saved["rng"])
   checks={"model":statehash(model.state_dict()),"optimizer":statehash(opt.state_dict()),"scaler":statehash(scaler.state_dict()),"rng":statehash(rng_state())}
   expected={k:statehash(saved[k]) for k in ("model","optimizer","scaler","rng")}
   if checks!=expected:raise RuntimeError("Restore state digest differs")
   writej(out/f"restore_at_{step:06d}.json",{"step":step,"next_global_draw":cursor,"pid":os.getpid(),"previous_pid":saved["pid"],"passed":True,"digests":checks})
   del saved
  else:
   saved=torch.load(c["source_m0"],map_location="cpu",weights_only=True)
   if saved["step"]!=10000:raise RuntimeError("Wrong original M0 checkpoint")
   model.load_state_dict(saved["model"],strict=True);del saved
   init={"initial_model":statehash(model.state_dict()),"initial_optimizer":statehash(opt.state_dict()),"initial_scaler":statehash(scaler.state_dict()),"initial_rng":statehash(rng_state())}
  loader=DataLoader(ds,batch_size=4,sampler=range(cursor,target*4),num_workers=0,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(c["seed"]))
  summary={"status":"running","method":c["method"],"seed":c["seed"],"target":target,"start_step":step,"gpu":snap,"run_binding":runbind,**init}
  try:
   torch.cuda.reset_peak_memory_stats()
   with (out/"train.jsonl").open("a" if resume else "x") as log,(out/"pairing.jsonl").open("a" if resume else "x") as pair:
    for batch in loader:
     if time.perf_counter()-started>=c["safe_stop_seconds"]:break
     x,y,f,w=[batch[k].cuda(non_blocking=True) for k in ("image","mask","fov","weight")]
     if x.shape!=(4,3,512,512) or y.shape!=(4,1,512,512) or f.shape!=y.shape or w.shape!=y.shape:raise ValueError("Batch shape differs")
     before=model.head.weight.detach().clone();retry=0
     while True:
      opt.zero_grad(set_to_none=True)
      with torch.autocast("cuda",dtype=torch.float16):z=model(x)
      loss=weighted_masked_loss(z,y,f,w,epsilon=1e-6)
      if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
      scaler.scale(loss).backward();scaler.unscale_(opt)
      gn=torch.nn.utils.clip_grad_norm_(model.parameters(),c["clip_norm"],error_if_nonfinite=False)
      if torch.isfinite(gn):break
      scaler.update();retry+=1
      if retry>8:raise RuntimeError("AMP overflow limit")
     scaler.step(opt);scaler.update();torch.cuda.synchronize()
     if torch.equal(before,model.head.weight):raise RuntimeError("No successful parameter update")
     step+=1;cursor+=4
     row={"step":step,"loss":float(loss.detach()),"gradient_norm":float(gn),"amp_scale":scaler.get_scale(),"amp_retries":retry,
      "draws":batch["draw"].tolist(),"filenames":list(batch["filename"]),"geometry":batch["geometry"].tolist(),
      "input_sha256":list(batch["input_sha256"]),
      "weight_counts":[{"one":int((w0==1).sum()),"two":int((w0==2).sum()),"outside":int((w0==0).sum())} for w0 in batch["weight"]],
      "parameters_updated":True,"elapsed_seconds":elapsed0+time.perf_counter()-started}
     log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
     pair.write(json.dumps({k:row[k] for k in ("step","draws","filenames","geometry","input_sha256")})+"\n");pair.flush()
     if step%c["save_every"]==0 or step==target:
      payload={"schema_version":1,"step":step,"next_global_draw":cursor,"epoch":cursor//440,"position_in_epoch":cursor%440,
       "model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"config":c,"binding":runbind,
       **init,"elapsed_seconds":elapsed0+time.perf_counter()-started,"pid":os.getpid()}
      atomic_save(payload,out/"last.pt");atomic_save(payload,out/f"step_{step:06d}.pt")
     if step%25==0:print(f"{c['method']} seed={c['seed']} step={step}/{target} loss={row['loss']:.6f}",flush=True)
   if step and not (out/f"step_{step:06d}.pt").is_file():
    payload={"schema_version":1,"step":step,"next_global_draw":cursor,"epoch":cursor//440,"position_in_epoch":cursor%440,
     "model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"config":c,"binding":runbind,
     **init,"elapsed_seconds":elapsed0+time.perf_counter()-started,"pid":os.getpid()}
    atomic_save(payload,out/"last.pt");atomic_save(payload,out/f"step_{step:06d}.pt")
   final=torch.load(out/"last.pt",map_location="cpu",weights_only=True)
   if final["step"]!=step or final["next_global_draw"]!=cursor:raise RuntimeError("Saved checkpoint cursor mismatch")
   summary.update(status="passed" if step==c["max_steps"] else "checkpointed",completed_steps=step,next_global_draw=cursor,
    model_digest=statehash(final["model"]),optimizer_digest=statehash(final["optimizer"]),scaler_digest=statehash(final["scaler"]),rng_digest=statehash(final["rng"]),
    checkpoint_sha256=sha(out/"last.pt"),elapsed_seconds=elapsed0+time.perf_counter()-started,
    max_allocated_mib=torch.cuda.max_memory_allocated()/1048576,max_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
   writej(out/"summary.json",summary);return summary
  except Exception:
   summary.update(status="failed",error=traceback.format_exc(),completed_steps=step,next_global_draw=cursor);writej(out/"summary.json",summary);raise
def eval_prob(prob,ref,threshold=.5):
 bpath=Path(ref["region_bundle"])
 if sha(bpath)!=ref["region_bundle_sha256"]:raise RuntimeError("Frozen regional bundle changed")
 with np.load(bpath,allow_pickle=False) as b:
  return evaluate_arrays(prob,load_mask(ref["vessel_path"]),load_mask(ref["fov_path"]),threshold=threshold,u=b["U"],
   regions={k:b[k] for k in ("internal","near","far")},thin=b["thin"],matching_support="omega")
def evaluate(cfgpath,out,step):
 cfg=json.loads(Path(cfgpath).read_text());bsha=require_binding()
 runbind={"execution_binding_sha256":bsha,"config_sha256":sha(cfgpath),"source_m0_sha256":cfg["source_m0_sha256"]}
 ckpt=Path(out)/f"step_{step:06d}.pt";csha=cfg["source_m0_sha256"] if step==0 else sha(ckpt);edir=Path(out)/"evaluation"/f"step_{step:06d}";mp=edir/"metrics.json"
 if mp.is_file():
  d=json.loads(mp.read_text())
  if d.get("checkpoint_sha256")==csha and d.get("run_binding")==runbind:return d
  raise RuntimeError("Stale evaluation output binding")
 with (ROOT/"experiments/a14_gpu0.lock").open("a+") as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);gs=gpu()
  manifest=json.loads(DEV.read_text());ds=NativeMaplesDataset("${RETINA_PROJECT_ROOT}/data/maples_handoff_20261008",split="dev")
  refs={r["filename"]:r for r in manifest["images"]}
  if len(refs)!=28 or {r["filename"] for r in ds.rows}!=set(refs):raise RuntimeError("Dev28 membership differs")
  if step==0:
   zdir=ROOT/"experiments/a15_2_z0_t0_calibration_20261015/z0/probabilities"; rec=[]
   for n in sorted(refs):rec.append(dict(filename=n,**eval_prob(np.load(zdir/f"{n}.npy",allow_pickle=False),refs[n])))
   source="bound shared Z0@0.5 predictions"
  else:
   saved=torch.load(ckpt,map_location="cpu",weights_only=True)
   if saved["step"]!=step or saved["binding"]!=runbind:raise RuntimeError("Checkpoint binding mismatch")
   mc=json.loads((CODE/"configs/development_v1.json").read_text());model=UNet(mc["widths"],mc["group_norm_groups"]).cuda()
   model.load_state_dict(saved["model"],strict=True);model.eval()
   norm=json.loads((CODE/"releases/development_v1/normalization.json").read_text());mu=torch.tensor(norm["mean"])[:,None,None];sd=torch.tensor(norm["std"])[:,None,None]
   pdir=edir/"probabilities";rec=[]
   for i,row in enumerate(ds.rows,1):
    item=ds[i-1];raw=item["image"].permute(1,2,0).numpy();fov=candidate_fov(np.rint(raw*255).astype(np.uint8))
    image=((item["image"]-mu)/sd)*torch.from_numpy(fov[None]).float()
    p,tiles=tiled_probability(model,image,patch=512,stride=256,batch_size=4,device="cuda",amp=False)
    a=p.numpy().astype(np.float32);r=eval_prob(a,refs[row["filename"]],.5);r.update(filename=row["filename"],tiled_count=tiles)
    rec.append(r)
    if step==2000:
     pdir.mkdir(parents=True,exist_ok=True);np.save(pdir/f"{row['filename']}.npy",a,allow_pickle=False)
    if i%7==0:print(f"{cfg['method']} seed={cfg['seed']} eval={step} images={i}/28",flush=True)
   del model,saved;source=str(ckpt)
 aggregate={}
 for m in ("dice","precision","recall","fp_density","near_pixel_recall","near_thin_recall_d1","far_recall"):
  v=[r[m]["value"] for r in rec if r[m]["value"] is not None]
  aggregate[m]={"mean":float(np.mean(v)) if v else None,"n":len(v),"NA":len(rec)-len(v)}
 result={"status":"passed","method":cfg["method"],"seed":cfg["seed"],"step":step,"threshold":.5,"checkpoint_sha256":csha,
  "run_binding":runbind,"source":source,"gpu":gs,"dev_images":len(rec),"aggregate":aggregate,"images":rec,"endpoint_probabilities_saved":step==2000}
 writej(mp,result);edir.mkdir(parents=True,exist_ok=True)
 fields=["filename"]+[m+s for m in aggregate for s in ("_value","_numerator","_denominator","_NA_reason")]
 with (edir/"per_image.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rec:
   q={"filename":r["filename"]}
   for m in aggregate:
    for suf,k in (("_value","value"),("_numerator","numerator"),("_denominator","denominator"),("_NA_reason","NA_reason")):q[m+suf]=r[m][k]
   w.writerow(q)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--worker",action="store_true");p.add_argument("--gate-worker",action="store_true")
 p.add_argument("--evaluate-step",type=int);p.add_argument("--config",required=True);p.add_argument("--output",required=True)
 p.add_argument("--stop-at",type=int);p.add_argument("--resume",action="store_true");a=p.parse_args()
 if a.evaluate_step is not None:r=evaluate(a.config,a.output,a.evaluate_step)
 else:r=worker(a.config,a.output,a.stop_at,a.resume,a.gate_worker)
 print(json.dumps(r,ensure_ascii=False,indent=2,default=str))
if __name__=="__main__":main()

