"""Measure cross-process inference variation on all first- and second-stage holdouts."""
import csv
import json
from collections import defaultdict
from prepare import ROOT,split
import numpy as np
import torch
from analyze import METRICS,SEEDS
import paired
from evaluate import make_predictor,probabilities
from two_scale import score


def main():
    torch.set_num_threads(2)
    data,_,_=paired.dataset()
    out=ROOT/"inference_replay"
    out.mkdir(exist_ok=True)
    for fold in range(3):
        for seed in SEEDS:
            for method in ("standard","mix","continue","branch","branch_aux"):
                result=out/f"f{fold}_{seed}_{method}.json"
                if result.is_file(): continue
                parent="eval_paired" if method in ("standard","mix") else "eval_focus"
                recorded=ROOT/parent/f"fold{fold}"/str(seed)/method
                saved={(r["image"],r["view"],r["point"]):r for r in json.loads((recorded/"dev.json").read_text())}
                cal=json.loads((recorded/"calibration.json").read_text())
                predictor=make_predictor(fold,seed,method)
                rows=[]
                for name in split(fold)["dev"]:
                    d=data[name]
                    with np.load(recorded/(name+".npz")) as bits:
                        for view,pr in probabilities(predictor,d["rgb"]).items():
                            for point,thr in (("fixed",.5),("matched",cal["thresholds"][view]),
                                              ("m0matched",cal["m0_thresholds"][view])):
                                mask=(pr>=thr)&d["f"]
                                old=np.unpackbits(bits[view+"_"+point]).reshape(mask.shape).astype(bool)
                                diff=old!=mask
                                now=score(d,pr,thr)
                                prior=saved[name,view,point]
                                rows.append(dict(image=name,fold=fold,seed=seed,method=method,view=view,point=point,
                                                 changed=int(diff.sum()),fov=int(d["f"].sum()),
                                                 max_margin=float(np.max(np.abs(pr[diff]-thr),initial=0)),
                                                 **{k:None if now[k] is None else 100*(now[k]-prior[k]) for k in METRICS}))
                tmp=result.with_suffix(".tmp");tmp.write_text(json.dumps(rows));tmp.replace(result)
            print("Frozen replay",fold,seed,flush=True)
    records=[r for path in sorted(out.glob("f*_*.json")) for r in json.loads(path.read_text())]
    assert len(records)==45*18*9
    groups=defaultdict(list)
    for r in records: groups[r["method"],r["view"],r["point"]].append(r)
    summaries=[]
    for (m,v,p),rows in groups.items():
        def avg(key):
            values=[x[key] for x in rows if x[key] is not None]
            return float(np.mean(values))
        summaries.append(dict(method=m,view=v,point=p,max_pixels=max(x["changed"] for x in rows),
                              changed_fraction=sum(x["changed"] for x in rows)/sum(x["fov"] for x in rows),
                              max_margin=max(x["max_margin"] for x in rows),**{k:avg(k) for k in METRICS}))
    (out/"summary.json").write_text(json.dumps(summaries,indent=2))
    print("Re-evaluated",len(records),"masks on fixed weights, thresholds, and held-out images.")


if __name__=="__main__": main()
