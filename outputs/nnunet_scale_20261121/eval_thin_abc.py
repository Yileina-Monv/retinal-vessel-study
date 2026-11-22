"""Train-only threshold calibration and held-out point transitions for A/B/C."""
import json
import numpy as np
import torch
from scipy import ndimage as ndi
from prepare import ROOT,split
from compare_addback_thresholds import make_model,load
from mine_thin_nested import OUT
from evaluate import probabilities
from matched_fp import GRID
from two_scale import score
import paired

def evaluate(cases,fold,seed,arms=("A","B","C"),comparisons=(("A","B"),("B","C"),("A","C"))):
    layout=split(fold); dest=OUT/f"fold{fold}"/str(seed)
    cal=load(ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction/calibration.json")
    prior=load(ROOT/"far_fragments/segments.json")
    hard={n:np.zeros(cases[n]["n"].shape,bool) for n in layout["dev"]}
    for r in prior:
        if r["image"] in hard: hard[r["image"]][tuple(np.array(r["yx"]).T)]=True
    for m in arms:
        arm=dest/m
        if (arm/"dev.json").exists(): continue
        model=make_model(fold,seed)
        model.network.load_state_dict(torch.load(arm/"weights.pt",weights_only=True,map_location="cpu"))
        curves={"native":[],"dual":[]}
        for name in layout["train"]:
            d=cases[name]
            if not d["n"].any(): continue
            p=probabilities(model,d["rgb"])
            for v in curves:
                dist=np.sort(p[v][d["n"]&~d["v"]])
                curves[v].append((len(dist)-np.searchsorted(dist,GRID))/d["n"].sum())
        thresholds={k:{v:float(GRID[np.flatnonzero(np.mean(cs,axis=0)<=cal[target])[0]]) for v,cs in curves.items()}
                    for k,target in (("m0matched","m0_target"),("matched","target"))}
        rows=[]
        for name in layout["dev"]:
            d=cases[name]; prob=probabilities(model,d["rgb"]); thin=d["s"]&d["n"]
            thinfar=d["s"]&d["f"]&~(d["h"]|d["n"])
            for point,ct in thresholds.items():
                for view,t in ct.items():
                    mask=(prob[view]>=t)&d["f"]
                    hit=ndi.binary_dilation(mask,structure=paired.cross)
                    stats=score(d,mask,.5)
                    stats["thin_d0"]=float(mask[thin].mean()) if thin.any() else None
                    stats["far_thin_d1"]=float(hit[thinfar].mean()) if thin.any() and thinfar.any() else None
                    rows.append(dict(image=name,point=point,view=view,metrics=stats,
                        hit=np.packbits(hit[thin]).tobytes().hex(),hard=int((hit&hard[name]).sum())))
        tmp=arm/"dev.tmp"
        tmp.write_text(json.dumps(dict(thresholds=thresholds,rows=rows),allow_nan=False))
        tmp.replace(arm/"dev.json")
    def index(m):
        return {(r["image"],r["view"],r["point"]):r for r in load(dest/m/"dev.json")["rows"]}
    archive={m:index(m) for pair in comparisons for m in pair}
    for a,b in comparisons:
        for view in ("native","dual"):
            data=[]
            for key,r in archive[a].items():
                name,v,point=key
                if v!=view: continue
                q=archive[b][key]
                h,j=[np.unpackbits(np.frombuffer(bytes.fromhex(z["hit"]),dtype=np.uint8)).astype(bool) for z in (r,q)]
                data.append(dict(image=name,point=point,base=r["metrics"],candidate=q["metrics"],
                    gains=int((~h&j).sum()),losses=int((h&~j).sum()),hard_base=r["hard"],hard_cand=q["hard"]))
            sub=OUT/(b+"_vs_"+a+"_"+view);sub.mkdir(exist_ok=True)
            tmp=sub/f"fold{fold}_{seed}.tmp"
            tmp.write_text(json.dumps(dict(fold=fold,seed=seed,rows=data),allow_nan=False))
            tmp.replace(tmp.with_suffix(".json"))
    print("evaluated",fold,seed,flush=True)
