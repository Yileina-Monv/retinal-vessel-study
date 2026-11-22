"""Compare matched-calibre near and far portions on the same held-out images."""
import argparse
import json
from prepare import ROOT,split
import numpy as np
import paired
from analyze import SEEDS


def reference(ids,roi):
    pix=np.flatnonzero((ids>0)&roi)
    _,inv,counts=np.unique(ids.ravel()[pix],return_inverse=True,return_counts=True)
    return pix,inv,counts


def coverage(hit,ref):
    pix,inv,counts=ref
    if not len(counts): return np.nan
    num=np.bincount(inv,weights=hit.ravel()[pix],minlength=len(counts))
    return np.mean(num/counts)


def main(teacher=False):
    cases,_,_=paired.dataset(include_tracks=True)
    pairs=[("standard_native","standard_dual"),("continue_native","branch_aux_native"),
           ("branch_native","branch_aux_native"),("continue_native","branch_native"),
           ("continue_dual","branch_aux_dual"),("standard_dual","branch_aux_dual")]
    if teacher:
        pairs=[("extra_native","correction_native"),("extra_dual","correction_dual"),
               ("continue_native","extra_native"),("standard_native","correction_native")]
    arms=sorted({a for pair in pairs for a in pair})
    values={a:{} for a in arms}
    for fold in range(3):
        for name in split(fold)["dev"]:
            d=cases[name]
            ids=d["all_tracks"]
            keys=np.unique(d["i"][d["i"]>0])
            if not len(keys): continue
            far=d["f"]&~d["n"]&~d["h"]
            rois=[reference(ids,d["n"]),reference(ids,far),reference(ids,far&np.isin(ids,keys))]
            for arm in arms:
                method,view=arm.rsplit("_",1)
                family="paired" if method=="standard" else "teacher" if method in ("extra","correction") else "focus"
                reps=[]
                for s in SEEDS:
                    file=ROOT/("eval_"+family)/f"fold{fold}"/str(s)/method/(name+".npz")
                    with np.load(file) as z:
                        mask=np.unpackbits(z[view+"_matched"]).reshape(ids.shape).astype(bool)
                    hit=mask.copy()
                    hit[1:] |= mask[:-1]; hit[:-1] |= mask[1:]
                    hit[:,1:] |= mask[:,:-1]; hit[:,:-1] |= mask[:,1:]
                    reps.append([coverage(hit,r) for r in rois])
                values[arm][name]=np.mean(reps,axis=0)
    summary={}
    for a,b in pairs:
        diffs=100*np.array([values[b][k]-values[a][k] for k in sorted(values[a])])
        record={}
        for i,key in ((1,"all_far_thin"),(2,"far_same_branches")):
            d=diffs[np.isfinite(diffs[:,i])]
            region_difference=d[:,0]-d[:,i]
            draws=np.random.default_rng(20261121).integers(len(d),size=(10000,len(d)))
            record[key]=dict(images=len(d),near_pp=d[:,0].mean(),far_pp=d[:,i].mean(),
                             region_difference_pp=region_difference.mean(),
                             ci=np.quantile(region_difference[draws].mean(axis=1),[.025,.975]).tolist())
        summary[b+"-"+a]=record
    (ROOT/("results_teacher" if teacher else "results_focus")/"spatial.json").write_text(json.dumps(summary,indent=2))
    print("Spatial comparisons complete.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--teacher",action="store_true")
    main(parser.parse_args().teacher)
