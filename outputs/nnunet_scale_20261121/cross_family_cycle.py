"""Combine two complementary enhanced 1x+2x family predictions at fixed equal weights."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT,OLD,split
from contrast_cycle import enhance
from compare_addback_thresholds import make_model,load
from evaluate import probabilities
from matched_fp import GRID
from two_scale import predict,score
import paired

OUT=ROOT/"cross_family_cycle"

def run(cases,hard,fold,seed):
    stage=OLD if not fold else OLD/f"fold_{fold}"
    m=paired.UNet().cuda().eval()
    m.load_state_dict(torch.load(stage/"rgb_only"/f"local_{seed}"/"weights.pt",weights_only=True,map_location="cpu"))
    n=make_model(fold,seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    def forward(d):
        return predict(m,{"rgb":d["enhanced"]})[1],probabilities(n,d["enhanced"])["dual"]
    cal=load(ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction/calibration.json")
    refs=[load(ROOT/sub/f"fold{fold}_{seed}.json") for sub in ["m0_contrast_transfer","contrast_cycle"]]
    curves=[]
    for name in split(fold)["train"]:
        d=cases[name];area=d["n"].sum()
        if area:
            a,b=forward(d)
            vals=np.sort(((a+b)*.5)[d["n"]&~d["v"]])
            curves.append((len(vals)-np.searchsorted(vals,GRID))/area)
    curve=np.mean(curves,axis=0)
    ts={k:float(GRID[np.flatnonzero(curve<=cal[target])[0]]) for k,target in (("m0matched","m0_target"),("matched","target"))}
    rows=[[],[]]
    for name in split(fold)["dev"]:
        d=cases[name];thin=d["s"]&d["n"];far=d["s"]&d["f"]&~(d["h"]|d["n"])
        a,b=forward(d)
        for key,t in ts.items():
            combo=((a+b)*.5>=t)&d["f"]
            ch=ndi.binary_dilation(combo,structure=paired.cross); candidate=score(d,combo,.5)
            candidate["far_thin_d1"]=float(ch[far].mean()) if far.any() and thin.any() else None
            for i,pr in enumerate((a,b)):
                mask=(pr>=refs[i]["thresholds"][key])&d["f"]
                base=score(d,mask,.5)
                expected=next(r["candidate"] for r in refs[i]["rows"] if r["image"]==name and r["point"]==key)
                assert all(expected[k]==v for k,v in base.items())
                bh=ndi.binary_dilation(mask,structure=paired.cross)
                base["far_thin_d1"]=float(bh[far].mean()) if far.any() and thin.any() else None
                rows[i].append(dict(image=name,point=key,base=base,candidate=candidate,
                    hard_base=int((hard[name]&bh).sum()),hard_cand=int((hard[name]&ch).sum()),
                    gains=int((thin&ch&~bh).sum()),losses=int((thin&bh&~ch).sum())))
    for i,family in enumerate(("m0","nnunet")):
        dest=OUT/family
        dest.mkdir(exist_ok=True,parents=True)
        tmp=dest/f"fold{fold}_{seed}.tmp"
        tmp.write_text(json.dumps(dict(fold=fold,seed=seed,thresholds=ts,rows=rows[i])))
        tmp.replace(tmp.with_suffix(".json"))
    print("paired families",fold,seed,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    cases=paired.dataset()[0];hard={}
    for n,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[n]=np.zeros(d["n"].shape,bool)
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for fold in range(3):
        for seed in (2027,2028,2029):
            if not (OUT/"nnunet"/f"fold{fold}_{seed}.json").exists():
                run(cases,hard,fold,seed)
