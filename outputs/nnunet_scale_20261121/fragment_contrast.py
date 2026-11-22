"""Pair far-missed thin points with nearby stable hits around the same haemorrhage."""
import json
import numpy as np
from scipy import ndimage as ndi
from prepare import ROOT,split
import paired

folder=ROOT/"far_fragments"
records=json.loads((folder/"segments.json").read_text())
cases=paired.dataset()[0]
rows=[]
for fold in range(3):
    for name in split(fold)["dev"]:
        fragments=[r for r in records if r["image"]==name]
        if not fragments: continue
        d=cases[name]
        ctl=d["s"] & d["n"]
        for seed in (2027,2028,2029):
            f=ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction"/(name+".npz")
            with np.load(f) as archive:
                mask=np.unpackbits(archive["dual_m0matched"]).reshape(ctl.shape).astype(bool)
            ctl &= ndi.binary_dilation(mask,structure=paired.cross)
        controls=np.argwhere(ctl)
        lesions=ndi.label(d["h"],np.ones((3,3)))[0]
        indices=ndi.distance_transform_edt(~d["h"],return_distances=False,return_indices=True)
        lesion=lesions[tuple(indices)]
        off=np.array([(i,j) for i in range(-5,6) for j in range(-5,6) if 4<=i*i+j*j<=25])
        def contrast(pos):
            y,x=pos
            q=pos+off
            q=q[((q>=0)&(q<1024)).all(1)]
            a,b=q.T
            bg=d["f"][a,b]&~d["v"][a,b]&~d["h"][a,b]
            if bg.sum()<5: return None
            return float(np.median(d["rgb"][a[bg],b[bg],1])-d["rgb"][y,x,1])
        for r in fragments:
            for pos in np.array(r["yx"]):
                y,x=pos
                cc=controls[lesion[tuple(controls.T)]==lesion[y,x]]
                if not len(cc): continue
                dd=np.square(cc-pos).sum(1)
                idx=np.argmin(dd)
                if dd[idx]>64**2: continue
                a,b=contrast(pos),contrast(cc[idx])
                if a is not None and b is not None:
                    rows.append(dict(image=name,id=r["id"],point=pos.tolist(),control=cc[idx].tolist(),
                                     distance=float(np.sqrt(dd[idx])),miss_green=a,hit_green=b))
(folder/"contrast_pairs.json").write_text(json.dumps(rows),encoding="utf-8")
