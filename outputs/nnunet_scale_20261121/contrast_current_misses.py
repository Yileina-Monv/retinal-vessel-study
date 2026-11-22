"""Pair current persistent far misses with unique, locally matched stable hits."""
import json
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from prepare import ROOT,split
import paired

OUT=ROOT/"switch_supervision"
OFF=np.array([(i,j) for i in range(-5,6) for j in range(-5,6) if 4<=i*i+j*j<=25])

def signal(d,point):
    y,x=point
    q=point+OFF;q=q[((q>=0)&(q<1024)).all(1)]
    yy,xx=q.T;use=d["f"][yy,xx]&~d["v"][yy,xx]&~d["h"][yy,xx]
    if use.sum()<5:return np.nan
    return float(np.median(d["rgb"][yy[use],xx[use],1])-d["rgb"][y,x,1])

def main():
    cases=paired.dataset()[0];rows=[];miss_total=0
    for fold in range(3):
        for k in split(fold)["dev"]:
            d=cases[k];target=d["s"]&d["n"];miss=target.copy();hit=target.copy()
            if not target.any():continue
            for s in (2027,2028,2029):
                with np.load(OUT/"heldout_masks"/f"fold{fold}_{s}"/(k+".npz")) as z:
                    mask=np.unpackbits(z["m0matched_base"]).reshape(target.shape).astype(bool)
                miss&=ndi.distance_transform_edt(~mask)>5
                hit&=ndi.binary_dilation(mask,structure=paired.cross)
            m,h=np.argwhere(miss),np.argwhere(hit);miss_total+=len(m)
            if not len(m) or not len(h):continue
            cm=np.array([signal(d,i) for i in m]);ch=np.array([signal(d,i) for i in h])
            d_he,nearest=ndi.distance_transform_edt(~d["h"],return_indices=True)
            lesion=ndi.label(d["h"],np.ones((3,3),bool))[0][tuple(nearest)]
            width2=ndi.distance_transform_edt(d["v"])**2
            at=lambda arr,coords:arr[tuple(coords.T)]
            dist=cdist(m,h)
            ok=(dist<=64)&(at(lesion,m)[:,None]==at(lesion,h))&(np.abs(at(d_he,m)[:,None]-at(d_he,h))<=2)
            ok&=np.isclose(at(width2,m)[:,None],at(width2,h))&np.isfinite(cm[:,None])&np.isfinite(ch)
            # All admissible distances combined cost less than one unmatched pair.
            penalty=65*(len(m)+1)
            cost=np.c_[np.where(ok,dist,2*penalty),np.full((len(m),len(m)),penalty)]
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if j>=len(h):continue
                assert ok[i,j]
                rows.append(dict(image=k,miss=m[i].tolist(),hit=h[j].tolist(),separation=float(dist[i,j]),
                                 green_miss=float(cm[i]),green_hit=float(ch[j])))
    assert miss_total==806
    per={}
    for k in sorted({r["image"] for r in rows}):
        group=[r for r in rows if r["image"]==k];a=np.array([r["green_miss"] for r in group]);b=np.array([r["green_hit"] for r in group])
        per[k]=dict(n=len(a),miss_mean=a.mean(),hit_mean=b.mean(),delta=(a-b).mean())
    diff=np.array([v["delta"] for v in per.values()])
    boot=np.random.default_rng(20261122).choice(diff,(10000,len(diff))).mean(1)
    summary=dict(total_far=miss_total,pairs=len(rows),images=len(per),miss=np.mean([v["miss_mean"] for v in per.values()]),
                 hit=np.mean([v["hit_mean"] for v in per.values()]),delta=diff.mean(),ci=np.percentile(boot,[2.5,97.5]).tolist(),
                 lower_images=int((diff<0).sum()),equal_images=int((diff==0).sum()))
    (OUT/"local_contrast.json").write_text(json.dumps(dict(summary=summary,by_image=per,rows=rows),indent=2))
    print(summary)

if __name__=="__main__":main()
