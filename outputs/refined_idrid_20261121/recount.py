"""Independently recount saved binary predictions and tabulate paired gain/loss."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import paired as p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("run",type=Path)
    ap.add_argument("candidate")
    ap.add_argument("--fold",type=int,choices=[0,1,2],default=0)
    args=ap.parse_args()
    data,_,dev=p.dataset(args.fold)
    rows=[]
    for seed in (2027,2028,2029):
        for image in dev:
            case=data[image]; ids=case["i"]; sel=ids>0
            keys,den=np.unique(ids[sel],return_counts=True)
            def read(method):
                folder=args.run/f"{method}_{seed}"
                recorded=next(r for r in json.loads((folder/"dev.json").read_text()) if r["image"]==image)
                with np.load(folder/(image+".npz")) as z:
                    pred=np.unpackbits(z["prediction"]).reshape(ids.shape).astype(bool)
                    near_fp=int((pred & case["n"] & ~case["v"]).sum())
                    # d1 independently uses a distance transform against predicted foreground.
                    hit=ndi.distance_transform_edt(~pred)<=1 if pred.any() else pred
                    hits=np.bincount(ids[sel & hit],minlength=int(ids.max())+1)[keys]
                    assert np.array_equal(z["hits"],hits) and np.array_equal(z["denominators"],den)
                    assert recorded["near_fp"]==near_fp
                    assert recorded["thin_hits"]==int((hit & case["s"] & case["n"]).sum())
                    score=float((hits/den).mean()) if len(keys) else None
                    assert score==recorded["path_d1"]
                    return hits,hit,near_fp
            a,x,fa=read("base");b,y,fb=read(args.candidate)
            thin=case["s"]&case["n"]
            rows.append(dict(seed=seed,image=image,paths=len(keys),base_zero=int((a==0).sum()),
                recovered_from_zero=int(((a==0)&(b>0)).sum()),lost_to_zero=int(((a>0)&(b==0)).sum()),
                paths_improved=int((b>a).sum()),paths_declined=int((a>b).sum()),
                thin_gained=int((~x&y&thin).sum()),thin_lost=int((x&~y&thin).sum()),delta_near_fp=fb-fa))
    path=args.run/(args.candidate+"_case_transitions.csv")
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for seed in (2027,2028,2029):
        s=[r for r in rows if r["seed"]==seed]
        print(seed,{k:sum(r[k] for r in s) for k in list(s[0])[3:]})


if __name__=="__main__": main()
