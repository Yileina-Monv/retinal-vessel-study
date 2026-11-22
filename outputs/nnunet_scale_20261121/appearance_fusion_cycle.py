"""Frozen ablations of appearance, family weight and two-scale M0 support."""
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
from report_paired_cycle import summarize
import paired

OUT=ROOT/"appearance_fusion_cycle"
METHODS=("raw","eight","m0_two_thirds","m0_supported")

def variants(rgb,enhanced,m,n):
    original=(predict(m,{"rgb":rgb})[1]+probabilities(n,rgb)["dual"])*.5
    native,mp=predict(m,{"rgb":enhanced})
    nn=probabilities(n,enhanced)["dual"]
    base=(mp+nn)*.5
    zoom=np.clip(2*mp-native,0,1)
    return base,mp,dict(raw=original,eight=(original+base)*.5,m0_two_thirds=(2*mp+nn)/3,
        m0_supported=base+.5*np.maximum(np.minimum(native,zoom)-nn,0))

def run(cases,hard,fold,seed,methods=METHODS,output=OUT,produce=variants,mask_archive=None):
    stage=OLD if not fold else OLD/f"fold_{fold}"
    m=paired.UNet().cuda().eval()
    m.load_state_dict(torch.load(stage/"rgb_only"/f"local_{seed}"/"weights.pt",weights_only=True,map_location="cpu"))
    n=make_model(fold,seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    prior=load(ROOT/"cross_family_cycle/m0"/f"fold{fold}_{seed}.json")
    mref=load(ROOT/"m0_contrast_transfer"/f"fold{fold}_{seed}.json")["thresholds"]
    cal=load(ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction/calibration.json")
    curves={k:[] for k in methods}
    for name in split(fold)["train"]:
        d=cases[name]
        if not d["n"].any(): continue
        _,_,maps=produce(d["rgb"],d["enhanced"],m,n)
        bg=d["n"]&~d["v"]
        for k,pr in maps.items():
            vals=np.sort(pr[bg]);curves[k].append((len(vals)-np.searchsorted(vals,GRID))/d["n"].sum())
    rows={k:[] for k in methods}
    ts={k:{point:float(GRID[np.flatnonzero(np.mean(c,axis=0)<=cal[field])[0]]) for point,field in
        (("m0matched","m0_target"),("matched","target"))} for k,c in curves.items()}
    for name in split(fold)["dev"]:
        d=cases[name];thin=d["n"]&d["s"];far=d["f"]&d["s"]&~(d["n"]|d["h"])
        p,mp,maps=produce(d["rgb"],d["enhanced"],m,n)
        packed={}
        for point,th in prior["thresholds"].items():
            base=(p>=th)&d["f"];bh=ndi.binary_dilation(base,structure=paired.cross)
            if mask_archive is not None: packed[point+"_base"]=np.packbits(base)
            mh=ndi.binary_dilation((mp>=mref[point])&d["f"],structure=paired.cross)
            stats=score(d,base,.5)
            record=next(r for r in prior["rows"] if r["point"]==point and r["image"]==name)
            assert all(v==record["candidate"][j] for j,v in stats.items())
            stats["far_thin_d1"]=float(bh[far].mean()) if thin.any() and far.any() else None
            stats["thin_d0"]=float(base[thin].mean()) if thin.any() else None
            for k,pr in maps.items():
                mask=(pr>=ts[k][point])&d["f"];hh=ndi.binary_dilation(mask,structure=paired.cross)
                if mask_archive is not None: packed[point+"_"+k]=np.packbits(mask)
                sc=score(d,mask,.5)
                sc["far_thin_d1"]=float(hh[far].mean()) if thin.any() and far.any() else None
                sc["thin_d0"]=float(mask[thin].mean()) if thin.any() else None
                h=hard[name]; lost_m=h&mh&~bh
                rows[k].append(dict(image=name,point=point,base=stats,candidate=sc,
                    gains=int((thin&~bh&hh).sum()),losses=int((thin&bh&~hh).sum()),
                    hard_base=int((h&bh).sum()),hard_cand=int((h&hh).sum()),
                    hard_gained=int((h&hh&~bh).sum()),hard_lost=int((h&~hh&bh).sum()),
                    m0_only=int(lost_m.sum()),m0_rescued=int((lost_m&hh).sum())))
        if mask_archive is not None:
            folder=mask_archive/f"fold{fold}_{seed}"
            folder.mkdir(exist_ok=True,parents=True)
            tmp=folder/(name+".tmp")
            with tmp.open("wb") as f:np.savez_compressed(f,**packed)
            tmp.replace(tmp.with_suffix(".npz"))
    for k in methods:
        dest=output/k;dest.mkdir(exist_ok=True,parents=True)
        tmp=dest/f"fold{fold}_{seed}.tmp"
        tmp.write_text(json.dumps(dict(fold=fold,seed=seed,thresholds=ts[k],rows=rows[k])))
        tmp.replace(tmp.with_suffix(".json"))
    print("complete",fold,seed,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    cases=paired.dataset()[0];hard={}
    for name,d in cases.items():
        d["enhanced"]=enhance(d["rgb"]);hard[name]=np.zeros(d["n"].shape,bool)
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for fold in range(3):
        for seed in (2027,2028,2029):
            if not all((OUT/k/f"fold{fold}_{seed}.json").exists() for k in METHODS):
                run(cases,hard,fold,seed)
    for k in METHODS: summarize(OUT/k)
