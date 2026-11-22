"""Compare equal log odds with family evidence scaled by two training-only FP anchors."""
import numpy as np
import torch
from scipy.special import expit,logit
from appearance_fusion_cycle import ROOT,run,predict,probabilities,enhance,paired,load,summarize

OUT=ROOT/"evidence_scale_cycle"
METHODS=("odds","aligned_odds")

def odds(x):
    return logit(np.clip(x,1e-5,1-1e-5))

def mapper(fold,seed):
    cuts=[load(ROOT/k/f"fold{fold}_{seed}.json")["thresholds"] for k in ("m0_contrast_transfer","contrast_cycle")]
    anchors=[(float(odds(c["m0matched"])),float(odds(c["matched"]))) for c in cuts]
    assert all(lo>hi for lo,hi in anchors)
    def variants(rgb,enhanced,m,n):
        mp=predict(m,{"rgb":enhanced})[1]
        nn=probabilities(n,enhanced)["dual"]
        a,b=odds(mp),odds(nn)
        (l1,h1),(l2,h2)=anchors
        return (mp+nn)*.5,mp,dict(odds=expit((a+b)*.5),aligned_odds=expit(.5*((a-l1)/(l1-h1)+(b-l2)/(l2-h2))))
    return variants

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
                run(cases,hard,fold,seed,METHODS,OUT,mapper(fold,seed))
    for k in METHODS:summarize(OUT/k)
