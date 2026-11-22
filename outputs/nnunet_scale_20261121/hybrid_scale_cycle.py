"""Keep original context at 1x, enhance the zoomed view, assess matched-width remote recall."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT,split
from compare_addback_thresholds import make_model,load
from evaluate import probabilities
from matched_fp import GRID
from two_scale import score
from contrast_cycle import enhance
import paired

OUT=ROOT/"hybrid_scale_cycle"

def run(cases,hard,fold,seed):
    names=split(fold)
    folder=ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction"
    cal=load(folder/"calibration.json")
    original_enh=load(ROOT/"contrast_cycle"/f"fold{fold}_{seed}.json")
    pred=make_model(fold,seed)
    def views(d):
        native=probabilities(pred,d["rgb"],native_only=True)["native"]
        enh=probabilities(pred,d["enhanced"])
        return (native+enh["large"])*.5,enh["dual"]
    curves=[]
    for name in names["train"]:
        d=cases[name];area=d["n"].sum()
        if area:
            hybrid,_=views(d)
            vals=np.sort(hybrid[d["n"]&~d["v"]])
            curves.append((len(vals)-np.searchsorted(vals,GRID))/area)
    curve=np.mean(curves,axis=0)
    ts={key:float(GRID[np.flatnonzero(curve<=cal[target])[0]]) for key,target in
        (("m0matched","m0_target"),("matched","target"))}
    rows=[];spatial=[]
    for name in names["dev"]:
        d=cases[name]; h,e=views(d);near=d["s"]&d["n"];far=d["s"]&d["f"]&~(d["n"]|d["h"])
        for key,t in ts.items():
            with np.load(folder/(name+".npz")) as z:
                base=np.unpackbits(z["dual_"+key]).reshape(near.shape).astype(bool)
            masks=[base,(h>=t)&d["f"],(e>=original_enh["thresholds"][key])&d["f"]]
            hits=[ndi.binary_dilation(m,structure=paired.cross) for m in masks]
            stats=[score(d,m,.5) for m in masks]
            for s,b in zip(stats,hits):
                s["far_thin_d1"]=float(b[far].mean()) if near.any() and far.any() else None
            old=next(r for r in original_enh["rows"] if r["image"]==name and r["point"]==key)
            assert old["candidate"]["thin_d1"]==stats[2]["thin_d1"]
            rows.append(dict(image=name,point=key,base=stats[0],candidate=stats[1],gains=int((near&hits[1]&~hits[0]).sum()),
                losses=int((near&hits[0]&~hits[1]).sum()),hard_base=int((hard[name]&hits[0]).sum()),hard_cand=int((hard[name]&hits[1]).sum())))
            spatial.append(dict(image=name,point=key,base=stats[0],candidate=stats[2],
                                **{k:old[k] for k in ("gains","losses","hard_base","hard_cand")}))
    for dest,rec in ((OUT,rows),(OUT/"enhanced_replay",spatial)):
        dest.mkdir(exist_ok=True,parents=True)
        tmp=dest/f"fold{fold}_{seed}.tmp"
        tmp.write_text(json.dumps(dict(fold=fold,seed=seed,thresholds=ts,rows=rec)))
        tmp.replace(tmp.with_suffix(".json"))
    print("Hybrid",fold,seed,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    cases=paired.dataset()[0]
    hard={}
    for n,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[n]=np.zeros(d["n"].shape,bool)
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for fold in range(3):
        for seed in (2027,2028,2029):
            if not (OUT/"enhanced_replay"/f"fold{fold}_{seed}.json").exists():
                run(cases,hard,fold,seed)
