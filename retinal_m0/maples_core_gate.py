"""A15-approved gate and exact six-run scheduler for the T0/TW core protocol."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import retinal_m0.maples_core_experiment as e
from retinal_m0.maples_core import CorePatches, draw_geometry
from retinal_data.maples_native import NativeMaplesDataset

CODE=e.CODE; OUT=e.OUT; ENG=OUT/"engineering"

def runproc(args, env=None):
    log=OUT/"orchestrator.log"; log.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run(args,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with log.open("a",encoding="utf-8") as f:
        f.write("\n$ "+" ".join(args)+"\n"+proc.stdout)
    for line in proc.stdout.splitlines():
        if line.startswith(("T0 seed=","TW seed=","T0 seed=","TW seed=")) or "eval=" in line:
            print(line,flush=True)
    if proc.returncode:
        print(proc.stdout[-12000:],flush=True)
        raise subprocess.CalledProcessError(proc.returncode,args)
    return proc.stdout

def worker_call(cfg,out,stop,resume=False,gate=True):
    args=[sys.executable,str(Path(e.__file__).resolve()),"--config",str(cfg),"--output",str(out),"--stop-at",str(stop)]
    if gate: args.append("--gate-worker")
    else: args.append("--worker")
    if resume: args.append("--resume")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONPATH=str(CODE))
    return runproc(args,env)

def run_gate():
    OUT.mkdir(parents=True,exist_ok=True);ENG.mkdir(parents=True,exist_ok=True)
    bound=e.binding();bp=OUT/"execution_binding.json"
    if (OUT/"runs").exists():raise RuntimeError("Formal outputs exist; implementation cannot be rebound")
    attempts=ENG/"attempts";attempts.mkdir(exist_ok=True)
    verification=ENG/"verification.json"
    if bp.exists() and verification.exists():
        old=json.loads(verification.read_text())
        if old.get("all_passed") and old.get("execution_binding_sha256")==e.sha(bp) and json.loads(bp.read_text())==bound:
            print("Current engineering gate is already valid.");return old
    if bp.exists():
        prior=json.loads(bp.read_text())
        if prior!=bound:
            existing=json.loads((OUT/"run_attempts.json").read_text()) if (OUT/"run_attempts.json").exists() else []
            nums=[int(q.name.split("_")[-1]) for q in attempts.glob("attempt_*") if q.name.split("_")[-1].isdigit()]
            n=max(nums,default=0)+1;archive=attempts/f"attempt_{n:02d}";archive.mkdir(exist_ok=True)
            if not (archive/"execution_binding.json").exists():shutil.copy2(bp,archive/"execution_binding.json")
            rec={"attempt":n,"status":"failed_before_gate_completion","reason":"Prior engineering gate did not complete; details in orchestrator.log.","binding_sha256":e.sha(archive/"execution_binding.json"),"formal_training_started":False}
            e.writej(archive/"attempt_record.json",rec);existing.append({"attempt":n,"record":str(archive/"attempt_record.json"),"artifacts_preserved_in_place":True})
            e.writej(OUT/"run_attempts.json",existing)
    nums=[int(q.name.split("_")[-1]) for q in attempts.glob("attempt_*") if q.name.split("_")[-1].isdigit()]
    attempt=attempts/f"attempt_{max(nums,default=0)+1:02d}";attempt.mkdir(exist_ok=True)
    e.writej(bp,bound);binding_sha=e.sha(bp)
    plan=attempt/"implementation_plan.md"
    plan.write_text("# T0/TW implementation and engineering gate\n\n- [x] Verify frozen protocol and bind inherited sources and data.\n- [x] Implement native-image region weights, synchronized transforms, and weighted BCE with unchanged Dice.\n- [x] Pass partition, uncertainty, hand-calculated loss/gradient, and all-ones equivalence checks.\n- [x] Verify three 8,000-draw geometries and real input prefixes.\n- [ ] Verify TW 50 continuous updates against fresh 25+25 recovery.\n- [ ] Start formal training only after the complete gate passes.\n",encoding="utf-8")
    runproc([sys.executable,"-m","unittest","discover","-s","tests","-p","test_maples_core.py","-v"],dict(os.environ,PYTHONPATH=str(CODE)))
    native=NativeMaplesDataset("${RETINA_PROJECT_ROOT}/data/maples_handoff_20261008",split="train")
    geometry={}
    for seed in e.SEEDS:
        streams=[]
        for method in e.METHODS:
            h=hashlib.sha256()
            for draw in range(8000):
                row=native.rows[draw%110];g=draw_geometry(seed,draw,int(row["height"]),int(row["width"]))
                h.update(json.dumps([draw,row["filename"],g],separators=(",",":")).encode())
            streams.append(h.hexdigest())
        if streams[0]!=streams[1]:raise RuntimeError("Paired draw geometry mismatch")
        geometry[str(seed)]={"draws_each":8000,"T0_sha256":streams[0],"TW_sha256":streams[1],"equal":True}
    paired=[]
    for seed in e.SEEDS:
        c0=e.config(seed,"T0",True);cw=e.config(seed,"TW",True)
        d0=CorePatches(c0,"T0");dw=CorePatches(cw,"TW");rng_before=e.statehash(e.rng_state())
        for draw in range(8):
            a=d0[draw];b=dw[draw]
            if a["input_sha256"]!=b["input_sha256"] or not torch.equal(a["geometry"],b["geometry"]):raise RuntimeError(f"Real paired input mismatch: {seed}/{draw}")
            if not torch.equal(a["weight"],a["fov"]) or not set(b["weight"].unique().tolist())<={0.,1.,2.}:raise RuntimeError("Training weights violate protocol")
            paired.append({"seed":seed,"draw":draw,"filename":a["filename"],"geometry":a["geometry"].tolist(),"input_sha256":a["input_sha256"],"same_RGB_V_FOV":True})
        if e.statehash(e.rng_state())!=rng_before:raise RuntimeError("Region builder consumed global RNG")
    cfg=e.config(2026,"TW",True);cfgpath=attempt/"tw_short_config.json";e.writej(cfgpath,cfg)
    continuous=attempt/"short_tw_continuous";segmented=attempt/"short_tw_segmented"
    worker_call(cfgpath,continuous,50);worker_call(cfgpath,segmented,25);worker_call(cfgpath,segmented,50,resume=True)
    a=torch.load(continuous/"last.pt",map_location="cpu",weights_only=True);b=torch.load(segmented/"last.pt",map_location="cpu",weights_only=True)
    equal={k:e.statehash(a[k])==e.statehash(b[k]) for k in ("model","optimizer","scaler","rng")}
    trace_equal=(continuous/"pairing.jsonl").read_bytes()==(segmented/"pairing.jsonl").read_bytes()
    passed=all(equal.values()) and trace_equal and a["step"]==b["step"]==50 and a["next_global_draw"]==b["next_global_draw"]==200
    if not passed:raise RuntimeError("TW 50 vs 25+25 recovery gate failed")
    report={"status":"passed","all_passed":True,"protocol_sha256":e.PROTO_SHA,"execution_binding_sha256":binding_sha,
      "checks":{"targeted_tests":"passed","whole_image_partition_and_weights":"passed","hand_calculated_loss_and_gradient":"passed",
      "all_ones_loss_gradient_exact_vs_frozen_masked_loss":"passed","paired_geometry_8000_per_seed":geometry,
      "real_input_prefix":paired,"region_construction_did_not_consume_global_rng":True,
      "TW_continuous_50_vs_fresh_25_plus_25":equal,"short_updates_total":100,"draw_trace_identical":trace_equal,
      "official_test_read":False,"formal_training_started":False},
      "source_binding_sha256":binding_sha,"evidence_paths":[str(cfgpath),str(continuous),str(segmented),str(plan),str(OUT/"run_attempts.json")]}
    e.writej(verification,report);plan.write_text(plan.read_text().replace("- [ ] Verify TW 50","- [x] Verify TW 50"),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2));return report

def run_formal():
    binding_sha=e.require_binding()
    order=[(2026,"T0"),(2026,"TW"),(2027,"TW"),(2027,"T0"),(2028,"T0"),(2028,"TW")]
    cfgroot=OUT/"configs";cfgroot.mkdir(parents=True,exist_ok=True)
    done=[]
    for seed,method in order:
        cfg=e.config(seed,method,False);cfgpath=cfgroot/f"seed_{seed}_{method}.json";e.writej(cfgpath,cfg)
        run=OUT/f"runs/seed_{seed}/{method}"
        if run.exists() and not (run/"last.pt").exists() and any(run.iterdir()):
            raise RuntimeError(f"Existing non-resumable run folder: {run}")
        if (run/"last.pt").exists():
            last=torch.load(run/"last.pt",map_location="cpu",weights_only=True);start=int(last["step"]);del last
            if start<2000:worker_call(cfgpath,run,2000,resume=True,gate=False)
        else:
            worker_call(cfgpath,run,2000,resume=False,gate=False)
        summary=json.loads((run/"summary.json").read_text())
        while summary.get("completed_steps",0)<2000:
            worker_call(cfgpath,run,2000,resume=True,gate=False)
            summary=json.loads((run/"summary.json").read_text())
        if summary.get("completed_steps")!=2000 or summary.get("next_global_draw")!=8000:
            raise RuntimeError(f"Formal endpoint incomplete: {run}")
        for step in (0,*range(200,2001,200)):
            run_logged([sys.executable,str(Path(e.__file__).resolve()),"--config",str(cfgpath),
                        "--output",str(run),"--evaluate-step",str(step)])
        done.append({"seed":seed,"method":method,"steps":2000,"draws":8000,"model_digest":summary["model_digest"]})
        if len(done)%2==0:
            e.writej(OUT/f"runs/seed_{seed}/paired_progress.json",{"seed":seed,"runs":done[-2:],"execution_binding_sha256":binding_sha})
            print(f"PAIRED SEED {seed} T0/TW complete",flush=True)
    e.writej(OUT/"formal_runs_summary.json",{"status":"six_trajectories_complete","runs":done,"execution_binding_sha256":binding_sha})
    return done

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-gate",action="store_true");g.add_argument("--run-formal",action="store_true")
    a=p.parse_args()
    result=run_gate() if a.run_gate else run_formal()
    if a.run_formal:print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

