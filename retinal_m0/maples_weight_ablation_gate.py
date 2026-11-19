"""CPU-only implementation and pairing gate for the weight ablation."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import retinal_m0.maples_weight_ablation_experiment as e
from retinal_m0.maples_weight_ablation_experiment import make_config,expected_pair,initial_check
from retinal_m0.maples_weight_ablation import reweight_base_sample
from retinal_m0.maples_core import CorePatches,draw_geometry
ROOT=e.ROOT;CODE=e.CODE;OUT=e.OUT
def run_gate():
 d=json.loads(e.PROTO.read_text());m=json.loads(e.MANIFEST.read_text());protected=m["protected_sha256"]
 if e.sha(e.PROTO)!=e.PROTO_SHA:raise RuntimeError("Frozen protocol SHA mismatch")
 cert=OUT/"implementation_gate/attempt_01/protected_manifest_check.json"
 c=json.loads(cert.read_text())
 if c.get("status")!="passed" or c.get("source_manifest_sha256")!=e.sha(e.MANIFEST) or c.get("protected_file_count")!=len(protected):raise RuntimeError("Missing full protected-manifest verification")
 reused=[]
 for r in d["weights"]["reuse"]:
  run=Path(r["run"])
  for rel in ("config.json","pairing.jsonl","summary.json"):
   if not (run/rel).is_file():raise RuntimeError(f"Missing reused artifact {run/rel}")
  s=json.loads((run/"summary.json").read_text())
  if s.get("status")!="passed" or s.get("completed_steps")!=2000 or s.get("next_global_draw")!=8000:raise RuntimeError(f"Reuse run incomplete: {run}")
  cp=Path(r["checkpoint"]);want=protected.get(str(cp))
  if not cp.is_file() or (want and e.sha(cp)!=want):raise RuntimeError(f"Reused checkpoint mismatch: {cp}")
  reused.append({"alpha":r["weight"],"seed":r["seed"],"method":r["method"],"checkpoint":str(cp),"checkpoint_sha256":e.sha(cp),"summary_sha256":e.sha(run/"summary.json")})
 if (OUT/"runs").exists():raise RuntimeError("New ablation runs directory exists; refusing overwrite")
 OUT.mkdir(parents=True,exist_ok=True)
 bind=e.binding();bp=OUT/"execution_binding.json";e.writej(bp,bind);bsha=e.sha(bp)
 proc=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_maples_weight_ablation.py","-v"],cwd=CODE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 eng=OUT/"implementation_gate";eng.mkdir(parents=True,exist_ok=True);(eng/"targeted_tests.log").write_text(proc.stdout,encoding="utf-8")
 if proc.returncode:raise RuntimeError("Targeted CPU tests failed")
 Native=__import__("retinal_data.maples_native",fromlist=["NativeMaplesDataset"]).NativeMaplesDataset
 train=Native("${RETINA_PROJECT_ROOT}/data/maples_handoff_20261008",split="train")
 if len(train)!=110 or train.split_counts!={"train":110,"dev":28,"official_test":60}:raise RuntimeError("MAPLES split counts differ")
 seed_results={}
 for seed in e.SEEDS:
  old=expected_pair(seed); rows=train.rows; trace=hashlib.sha256()
  for draw,(draw0,name,geom,ihash) in enumerate(old):
   r=rows[draw%len(rows)]
   g=list(draw_geometry(int(seed),int(draw),int(r["height"]),int(r["width"])))
   if draw0!=draw or name!=r["filename"] or geom!=g:raise RuntimeError(f"Full geometry/order mismatch seed={seed} draw={draw}")
   trace.update(json.dumps([draw,name,g,ihash],separators=(",",":")).encode())
  cfg=make_config(seed,1.5);base=CorePatches(cfg,"TW");selected=(0,1,2,3,4,5,6,7,7199,7200,7999)
  e.parent.setup(seed);before=e.parent.statehash(e.parent.rng_state());evidence=[]
  for draw in selected:
   raw=base[draw];a=reweight_base_sample(raw,1.5);b=reweight_base_sample(raw,3.0);want=old[draw]
   got=(draw,a["filename"],a["geometry"].tolist(),a["input_sha256"])
   if got!=want:raise RuntimeError(f"Actual input hash mismatch seed={seed} draw={draw}")
   if a["input_sha256"]!=b["input_sha256"] or not torch.equal(a["geometry"],b["geometry"]) or not torch.equal(a["image"],b["image"]) or not torch.equal(a["mask"],b["mask"]) or not torch.equal(a["fov"],b["fov"]):raise RuntimeError(f"New alpha input stream differs seed={seed} draw={draw}")
   evidence.append({"draw":draw,"filename":a["filename"],"geometry":a["geometry"].tolist(),"input_sha256":a["input_sha256"],"same_input_alpha_1p5_3":True})
  after=e.parent.statehash(e.parent.rng_state())
  if before!=after:raise RuntimeError(f"Sampling/weight map consumed global RNG seed={seed}")
  init=initial_check(seed,cfg)
  seed_results[str(seed)]={"draws":8000,"full_order_geometry_sha256":trace.hexdigest(),"all_draw_geometry_matches_old_pairing":True,"actual_real_input_checks":evidence,"new_alpha_inputs_identical":True,"global_rng_unchanged":True,"initial_state_matches_old_pair":init}
 check={"targeted_tests":"passed","protected_source_manifest":{"count":len(protected),"all_hashes_match":True,"verification_record":str(cert),"verification_record_sha256":e.sha(cert)},
 "reused_formal_runs":reused,"three_seed_full_8000_draw_pairing":seed_results,"selected_real_input_positions":[0,1,2,3,4,5,6,7,7199,7200,7999],
 "contribution_recovery_synthetic_test":"passed","official_test_read":False,"optimizer_updates":0}
 verification={"status":"passed","all_passed":True,"protocol_sha256":e.PROTO_SHA,"execution_binding_sha256":bsha,"checks":check}
 e.writej(OUT/"implementation_gate/verification.json",verification)
 print(json.dumps(verification,ensure_ascii=False,indent=2))
 return verification
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--run-gate",action="store_true",required=True);p.parse_args();run_gate()
if __name__=="__main__":main()
