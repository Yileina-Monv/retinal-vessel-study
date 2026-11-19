import sys, os, json, time, math, fcntl, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from paths import OUT, GROUP
from common import ROOT, CODE, DATA_BASE, BASELINE, read, sha, atomic, log
WORK=OUT/"gpu_verification_v1"
GPU="${GPU_UUID}"
MARKER="NL_AMP_REPAIR_GPU_ZERO4_RELEASED_20261102"
def validate():
    p=WORK/"GPU_RELEASE.json"
    if not p.exists():raise RuntimeError("independent zero-update GPU release absent")
    r=read(p)
    if r.get("scope")!="FOUR_FIXED_FORWARDS_ZERO_UPDATES" or r.get("GPU")!=GPU or r.get("max_forwards")!=4 or r.get("max_updates")!=0 or r.get("seconds_cap")!=1800:
        raise RuntimeError("release scope drift")
    if MARKER not in (CODE/"STATUS.md").read_text()[:16000]:
        raise RuntimeError("STATUS lacks current GPU release")
    for path,h in r["bound_files"].items():
        if sha(path)!=h:raise RuntimeError("release source/evidence drift "+path)
    if read(OUT/"CPU_PREPARATION_RESULT.json")["status"]!="PASS_CPU_PREPARATION":
        raise RuntimeError("CPU preparation not passed")
    return sha(p)
def case_list():
    cases=[]
    for seed in [2027,2028]:
        result=read(OUT/("replay_seed_"+str(seed))/"RESULT.json")
        items=result["selected_diagnostics"]
        for case in ["diagnostic_interrupted.pt","diagnostic_nonempty.pt"]:
            found=[x for x in items if Path(x["path"]).name==case]
            if len(found)!=1:raise RuntimeError("fixed diagnostic missing")
            cases.append({"seed":seed,"name":case,"batch":found[0],
                          "checkpoint":str(GROUP/"runs"/("seed_"+str(seed))/"state_interrupted.pt")})
    return cases
def reserve_forward(ledger,case_index):
    if ledger["status"]!="RUNNING" or ledger["optimizer_updates"]!=0 or ledger["reserved"]!=case_index or ledger["reserved"]>=4 or ledger["pending"] is not None:
        raise RuntimeError("GPU forward admission denied")
    ledger["reserved"]+=1
    ledger["pending"]={"case_index":case_index,"forward_called":False}
    return ledger
def mark_forward(ledger,case_index):
    p=ledger["pending"]
    if ledger["status"]!="RUNNING" or p is None or p["case_index"]!=case_index or p["forward_called"] or ledger["optimizer_updates"]:
        raise RuntimeError("extra or unbound GPU forward")
    p["forward_called"]=True;ledger["actual_forwards"]+=1
    return ledger
def complete_case(ledger,case_index):
    if ledger["pending"]!={"case_index":case_index,"forward_called":True} or ledger["optimizer_updates"]:
        raise RuntimeError("incomplete diagnostic case")
    ledger["complete"]+=1;ledger["pending"]=None
    return ledger
def ratio_guard(base_l2,aux_l2,coefficient):
    ratio=coefficient*aux_l2/base_l2 if base_l2 else (0. if aux_l2==0 else math.inf)
    if not math.isfinite(ratio) or ratio>2:raise RuntimeError("original ratio>2 guard")
    return ratio
