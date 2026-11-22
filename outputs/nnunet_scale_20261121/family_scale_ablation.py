"""Factorial diagnostic: equal family weights, change scales available to each family."""
import numpy as np
import torch
from appearance_fusion_cycle import ROOT,run,predict,probabilities,enhance,paired,load,summarize

OUT=ROOT/"family_scale_ablation"
PAIRS=[(m,n) for m in ("native","large","dual") for n in ("native","large","dual")
       if (m,n)!=("dual","dual")]
METHODS=[m+"_"+n for m,n in PAIRS]

def variants(rgb,enhanced,m,n):
    a,d=predict(m,{"rgb":enhanced})
    mv={"native":a,"large":np.clip(2*d-a,0,1),"dual":d}
    nv=probabilities(n,enhanced)
    return (d+nv["dual"])*.5,d,{i+"_"+j:(mv[i]+nv[j])*.5 for i,j in PAIRS}

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
                run(cases,hard,fold,seed,METHODS,OUT,variants)
    for k in METHODS: summarize(OUT/k)
