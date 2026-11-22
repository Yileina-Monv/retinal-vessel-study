"""Direct transfer of the fixed RGB contrast transform to existing M0 U-Net weights."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT,OLD,split
from contrast_cycle import enhance
from matched_fp import GRID
from two_scale import predict,score
from compare_addback_thresholds import load
import paired

OUT=ROOT/"m0_contrast_transfer"

def run(cases,hard,fold,seed):
    stage=OLD if not fold else OLD/f"fold_{fold}"
    src=stage/"rgb_only"/f"local_{seed}"/"weights.pt"
    model=paired.UNet().cuda().eval()
    model.load_state_dict(torch.load(src,map_location="cpu",weights_only=True))
    low=load(stage/"two_scale"/f"local_{seed}"/"threshold.json")
    high=load(ROOT/"common_high"/f"fold{fold}_{seed}.json")
    original={"m0matched":low,"matched":high}
    curves=[]
    for n in split(fold)["train"]:
        d=cases[n];area=d["n"].sum()
        if area:
            pr=predict(model,{"rgb":d["enhanced"]})[1]
            vals=np.sort(pr[d["n"]&~d["v"]]);curves.append((vals.size-np.searchsorted(vals,GRID))/area)
    curve=np.mean(curves,axis=0)
    thresholds={k:float(GRID[np.flatnonzero(curve<=v["target"])[0]]) for k,v in original.items()}
    rows=[]
    for n in split(fold)["dev"]:
        d=cases[n];thin=d["s"]&d["n"];far=d["s"]&d["f"]&~(d["h"]|d["n"])
        orig=predict(model,d)[1];updated=predict(model,{"rgb":d["enhanced"]})[1]
        for key,t in thresholds.items():
            a=(orig>=original[key]["threshold"])&d["f"];b=(updated>=t)&d["f"]
            h,j=[ndi.binary_dilation(m,structure=paired.cross) for m in (a,b)]
            stats=[score(d,m,.5) for m in (a,b)]
            for s,m in zip(stats,(h,j)): s["far_thin_d1"]=float(m[far].mean()) if near_valid(thin,far) else None
            rows.append(dict(image=n,point=key,base=stats[0],candidate=stats[1],gains=int((thin&j&~h).sum()),
                losses=int((thin&h&~j).sum()),hard_base=int((hard[n]&h).sum()),hard_cand=int((hard[n]&j).sum())))
    tmp=OUT/f"fold{fold}_{seed}.tmp"
    tmp.write_text(json.dumps(dict(fold=fold,seed=seed,rows=rows,thresholds=thresholds)))
    tmp.replace(tmp.with_suffix(".json"))
    print("M0 transfer",fold,seed,flush=True)

def near_valid(thin,far):
    return thin.any() and far.any()

if __name__=="__main__":
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cases=paired.dataset()[0];hard={}
    for n,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[n]=np.zeros(d["n"].shape,bool)
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    OUT.mkdir(exist_ok=True)
    for fold in range(3):
        for seed in (2027,2028,2029):
            if not (OUT/f"fold{fold}_{seed}.json").exists(): run(cases,hard,fold,seed)
