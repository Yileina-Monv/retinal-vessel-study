"""Summarize image-paired calibrated inference comparisons."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parent
folder=ROOT/"addback_calibration"
files=sorted(folder.glob("fold*.json"))
assert len(files)==9
metrics={"thin_d1":100,"thin_d0":100,"near_fp_density":1000,"path_d1":100,
         "dice":100,"precision":100,"fp_fov":1000,"far_recall":100}
data=defaultdict(dict)
for f in files:
    run=json.loads(f.read_text())
    for r in run["rows"]:
        data[r["method"]][r["image"],run["seed"]]=r
assert all(len(tab)==162 for tab in data.values())
summary={}
for method,records in data.items():
    summary[method]={}
    for field,scale in metrics.items():
        baseline=data["equal"]
        byimage=defaultdict(list); byseed=defaultdict(list); scores=[]
        for key,r in records.items():
            if r[field] is None: continue
            val=r[field]*scale
            diff=val-baseline[key][field]*scale
            byimage[key[0]].append(diff)
            byseed[key[1]].append(diff)
            scores.append(val)
        assert all(len(vals)==3 for vals in byimage.values())
        values=np.array([np.mean(vals) for vals in byimage.values()])
        draws=np.random.default_rng(20261122).choice(values,(10000,len(values))).mean(axis=1)
        summary[method][field]=dict(mean=float(np.mean(scores)),delta=float(values.mean()),
            ci=np.percentile(draws,[2.5,97.5]).tolist(),up=int((values>1e-9).sum()),
            down=int((values< -1e-9).sum()),images=len(values),seed_delta={s:float(np.mean(v)) for s,v in byseed.items()},
            image_delta={name:float(np.mean(vals)) for name,vals in byimage.items()})
    print(method,[(k,round(s["mean"],3)) for k,s in summary[method].items()])
    for k in ["thin_d1","near_fp_density"]:
        print(k,{kk:v for kk,v in summary[method][k].items() if kk not in ("image_delta","mean")})
(folder/"summary.json").write_text(json.dumps(summary,indent=2))
