"""Image-paired uncertainty for frozen-weight experiments, conditionally on the seeds."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def summarize(folder):
    files=sorted(folder.glob("fold*.json"))
    assert len(files)==9
    results={}
    for point in ("m0matched","matched"):
        rows=[(s["fold"],s["seed"],r) for f in files for s in [json.loads(f.read_text())]
              for r in s["rows"] if r["point"]==point]
        assert len({(r["image"],seed) for _,seed,r in rows})==162 and len(rows)==162
        stats={}
        for key in rows[0][2]["base"]:
            measure=1000 if key in ("fp_fov","near_fp_density") else 100
            images=defaultdict(list); seeds=defaultdict(list); folds=defaultdict(list); pairs=[]
            for fold,seed,r in rows:
                a,b=r["base"][key],r["candidate"][key]
                if a is None or b is None:
                    assert a==b
                    continue
                delta=(b-a)*measure
                images[r["image"]].append(delta); seeds[seed].append(delta); folds[fold].append(delta)
                pairs.append((a*measure,b*measure))
            vals=np.array([np.mean(v) for v in images.values()])
            boot=np.random.default_rng(20261122).choice(vals,(10000,len(vals))).mean(1)
            stats[key]=dict(base=np.mean(pairs,axis=0)[0],candidate=np.mean(pairs,axis=0)[1],
                delta=vals.mean(),ci=np.percentile(boot,[2.5,97.5]).tolist(),up=int((vals>1e-8).sum()),
                down=int((vals< -1e-8).sum()),images=len(vals),by_seed={k:np.mean(v) for k,v in seeds.items()},
                by_fold={k:np.mean(v) for k,v in folds.items()},by_image={k:np.mean(v) for k,v in images.items()})
        stats["transitions"]={seed:{k:sum(r[k] for f,s,r in rows if s==seed) for k in ("gains","losses","hard_base","hard_cand")} for seed in (2027,2028,2029)}
        results[point]=stats
        print(point,[(k,round(v["candidate"],3),round(v["delta"],3)) for k,v in stats.items() if k!="transitions"])
        print(stats["thin_d1"]["ci"],stats["thin_d1"]["by_seed"],stats["thin_d1"]["by_fold"],stats["transitions"])
    (folder/"summary.json").write_text(json.dumps(results,indent=2))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("folder",type=Path)
    summarize(ap.parse_args().folder)
