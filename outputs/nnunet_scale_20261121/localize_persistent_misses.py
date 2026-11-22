"""Describe the geometry of repeat misses, using only archived held-out masks."""
import json
from prepare import ROOT
import numpy as np
from scipy import ndimage as ndi
import paired

cases,_,_=paired.dataset()
audit=json.loads((ROOT/"thin_recall_audit.json").read_text())
rows=[]
for old in audit:
    name=old["image"]
    d=cases[name]
    target=d["n"]&d["s"]
    if not target.any(): continue
    distances=[]
    ever_hit_path_ids=set()
    for seed in (2027,2028,2029):
        file=ROOT/"eval_teacher"/f'fold{old["fold"]}'/str(seed)/"correction"/(name+".npz")
        with np.load(file) as z:
            mask=np.unpackbits(z["dual_m0matched"]).reshape(target.shape).astype(bool)
        distance=ndi.distance_transform_edt(~mask)
        distances.append(distance[target])
        ever_hit_path_ids.update(np.unique(d["i"][(d["i"]>0)&(distance<=1)]).tolist())
    dist=np.stack(distances)
    persistent=(dist.min(0)>1)
    assert persistent.sum()==old["dual_always_missed"]
    onpath=d["i"][target]>0
    zero_path=onpath & ~np.isin(d["i"][target],list(ever_hit_path_ids))
    rows.append(dict(image=name,total=int(target.sum()),onpath=int(onpath.sum()),
        missed=int(persistent.sum()),missed_onpath=int((persistent&onpath).sum()),
        close_all=int((persistent&(dist.max(0)<=2)).sum()),
        far_all=int((dist.min(0)>5).sum()),
        far_onpath=int(((dist.min(0)>5)&onpath).sum()),
        far_zero_path=int(((dist.min(0)>5)&zero_path).sum()),
        missed_zero_path=int((persistent&zero_path).sum()),
        very_far_all=int((dist.min(0)>10).sum())))
(ROOT/"persistent_miss_distances.json").write_text(json.dumps(rows,indent=2))
print({k:sum(i[k] for i in rows) for k in rows[0] if k!="image"})
