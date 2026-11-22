"""Aggregate the fixed endpoints by image. Repeated seeds share the same Dev images."""
import csv
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SEEDS=(2027,2028,2029)
METRICS=("path_d1","thin_d1","near_fp_density","dice","precision","fp_fov","far_recall")


def summarize(folder=HERE/"rgb_only", baseline="base", candidate="local", prefix="paired", images=None):
    if images is None:
        images=json.loads((HERE/"split.json").read_text())["dev"]
    data={}
    for seed in SEEDS:
        for method in (baseline,candidate):
            rec=json.loads((folder/f"{method}_{seed}"/"dev.json").read_text())
            assert [x["image"] for x in rec]==images
            data[seed,method]=rec
    rows=[]
    by_image=[]
    for metric in METRICS:
        cube=np.asarray([[[np.nan if x[metric] is None else x[metric] for x in data[seed,m]]
                          for m in (baseline,candidate)] for seed in SEEDS],dtype=float)
        ok=np.isfinite(cube[0,0]); assert np.all(np.isfinite(cube)==ok)
        cube=cube[:,:,ok]
        delta=cube[:,1]-cube[:,0]
        diff=delta.mean(axis=0)
        rng=np.random.default_rng(20261121)
        ix=rng.integers(len(diff),size=(10000,len(diff)))
        ci=np.quantile(diff[ix].mean(axis=1),[.025,.975])*100
        row=dict(metric=metric,images=len(diff),baseline=100*cube[:,0].mean(),candidate=100*cube[:,1].mean(),
                 delta_pp=100*diff.mean(),ci95_pp=ci.tolist(),by_seed_pp=(100*delta.mean(axis=1)).tolist(),
                 improved_numeric=int((diff>1e-10).sum()),declined_numeric=int((diff< -1e-10).sum()),
                 loo_range_pp=[100*(diff.sum()-diff.max())/(len(diff)-1),100*(diff.sum()-diff.min())/(len(diff)-1)])
        rows.append(row)
        for name,d in zip(np.array(images)[ok],diff): by_image.append(dict(metric=metric,image=name,delta_pp=100*d))
    (folder/(prefix+"_summary.json")).write_text(json.dumps(rows,indent=2))
    with (folder/(prefix+"_per_image.csv")).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(by_image[0]));w.writeheader();w.writerows(by_image)
    for r in rows: print(r["metric"],round(r["delta_pp"],4),np.round(r["ci95_pp"],4),np.round(r["by_seed_pp"],4))


if __name__=="__main__": summarize()
