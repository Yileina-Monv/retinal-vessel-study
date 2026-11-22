"""Test agreement across independently fitted weights at a shared training FP target."""
import itertools
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

SEEDS=(2027,2028,2029)
GROUPS=[(s,) for s in SEEDS]+list(itertools.combinations(SEEDS,2))+[SEEDS]
OUT=ROOT/"seed_agreement_cycle"

def run(cases,hard,fold):
    layout=split(fold)
    models={s:make_model(fold,s) for s in SEEDS}
    folders={s:ROOT/"eval_teacher"/f"fold{fold}"/str(s)/"correction" for s in SEEDS}
    cals={s:load(f/"calibration.json") for s,f in folders.items()}
    curves={g:[] for g in GROUPS}
    def probs(d):
        return {s:probabilities(m,d["rgb"])["dual"] for s,m in models.items()}
    for name in layout["train"]:
        d=cases[name]; area=d["n"].sum()
        if not area: continue
        maps=probs(d); bg=d["n"]&~d["v"]
        selected={s:pr[bg] for s,pr in maps.items()}
        for g in GROUPS:
            dist=np.sort(np.mean([selected[s] for s in g],axis=0))
            curves[g].append((len(dist)-np.searchsorted(dist,GRID))/area)
    cuts={}
    for key,target in (("m0matched","m0_target"),("matched","target")):
        budget=np.mean([c[target] for c in cals.values()])
        cuts[key]={g:float(GRID[np.flatnonzero(np.mean(curves[g],axis=0)<=budget)[0]]) for g in GROUPS}
    entries=[]
    for name in layout["dev"]:
        d=cases[name]; maps=probs(d); thin=d["s"]&d["n"]
        for s in SEEDS:
            with np.load(folders[s]/(name+".npz")) as z:
                assert np.array_equal(np.packbits((maps[s]>=cals[s]["m0_thresholds"]["dual"])&d["f"]),z["dual_m0matched"])
        for key,ts in cuts.items():
            hits={}; scores={}
            for g,t in ts.items():
                pr=np.mean([maps[s] for s in g],axis=0)
                mask=(pr>=t)&d["f"]
                hits[g]=ndi.binary_dilation(mask,structure=paired.cross)
                scores[g]=score(d,mask,.5)
            for g in GROUPS:
                if len(g)==1: continue
                for s in g:
                    a=hits[(s,)]; b=hits[g]
                    entries.append(dict(image=name,point=key,group=list(g),reference=s,base=scores[(s,)],candidate=scores[g],
                        gains=int((thin&b&~a).sum()),losses=int((thin&a&~b).sum()),hard_base=int((hard[name]&a).sum()),hard_cand=int((hard[name]&b).sum())))
    (OUT/f"groups_fold{fold}.json").write_text(json.dumps(dict(fold=fold,thresholds={k:{"_".join(map(str,g)):t for g,t in v.items()} for k,v in cuts.items()},rows=entries)))
    for s in SEEDS:
        rows=[r for r in entries if r["group"]==list(SEEDS) and r["reference"]==s]
        (OUT/f"fold{fold}_{s}.json").write_text(json.dumps(dict(seed=s,fold=fold,rows=rows)))
    print("Completed ensemble fold",fold,flush=True)

if __name__=="__main__":
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cases=paired.dataset()[0]; hard={k:np.zeros(d["n"].shape,bool) for k,d in cases.items()}
    for r in load(ROOT/"far_fragments/segments.json"):
        hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for fold in range(3):
        if not all((OUT/f"fold{fold}_{s}.json").exists() for s in SEEDS):
            run(cases,hard,fold)
