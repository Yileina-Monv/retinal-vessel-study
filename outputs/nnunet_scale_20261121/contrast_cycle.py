"""Pre-specified frozen-weight cycle: mild RGB local contrast versus archived input."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT, split
from compare_addback_thresholds import make_model, load
from evaluate import probabilities
from two_scale import score
from matched_fp import GRID
import paired

OUT=ROOT/"contrast_cycle"

def enhance(rgb):
    # Add a bounded green-channel high-pass signal equally to RGB.
    valid=rgb.max(2)>10
    g=rgb[:,:,1].astype(np.float32)
    denom=ndi.gaussian_filter(valid.astype(np.float32),5)
    local=ndi.gaussian_filter(g*valid,5)/np.maximum(denom,.001)
    shift=np.clip(.5*(g-local),-8,8)*valid
    return np.clip(np.rint(rgb.astype(np.float32)+shift[:,:,None]),0,255).astype(np.uint8)

def run(cases,hard,fold,seed):
    layout=split(fold)
    folder=ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction"
    cal=load(folder/"calibration.json")
    model=make_model(fold,seed)
    curves=[]
    for name in layout["train"]:
        d=cases[name]
        if not d["n"].any(): continue
        prob=probabilities(model,d["enhanced"])["dual"]
        bg=np.sort(prob[d["n"] & ~d["v"]])
        curves.append((bg.size-np.searchsorted(bg,GRID))/d["n"].sum())
    curve=np.mean(curves,axis=0)
    thresholds={}
    for key,target in (("m0matched",cal["m0_target"]),("matched",cal["target"])):
        thresholds[key]=float(GRID[np.flatnonzero(curve<=target)[0]])
    rows=[]
    for name in layout["dev"]:
        d=cases[name]
        prob=probabilities(model,d["enhanced"])["dual"]
        thin=d["s"] & d["n"]
        with np.load(folder/(name+".npz")) as archive:
            for key,threshold in thresholds.items():
                base=np.unpackbits(archive["dual_"+key]).reshape(thin.shape).astype(bool)
                cand=(prob>=threshold)&d["f"]
                b,c=[ndi.binary_dilation(z,structure=paired.cross) for z in (base,cand)]
                rows.append(dict(image=name,point=key,base=score(d,base,.5),candidate=score(d,cand,.5),
                    gains=int((thin&c&~b).sum()),losses=int((thin&b&~c).sum()),
                    hard_base=int((hard[name]&b).sum()),hard_cand=int((hard[name]&c).sum())))
    file=OUT/f"fold{fold}_{seed}.json"
    tmp=file.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(fold=fold,seed=seed,thresholds=thresholds,rows=rows),allow_nan=False))
    tmp.replace(file)
    print(fold,seed,thresholds,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cases=paired.dataset()[0]
    hard={}
    for name,d in cases.items():
        d["enhanced"]=enhance(d["rgb"])
        hard[name]=np.zeros(d["n"].shape,bool)
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    assert sum(h.sum() for h in hard.values())==897
    for fold in range(3):
        for seed in (2027,2028,2029):
            if not (OUT/f"fold{fold}_{seed}.json").exists():
                run(cases,hard,fold,seed)
