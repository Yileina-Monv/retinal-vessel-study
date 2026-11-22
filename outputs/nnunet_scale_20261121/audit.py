"""Check held-out masks and file identities independently of the evaluation computation."""
import argparse
import csv
import hashlib
import json
from prepare import ROOT,split
import numpy as np
import paired


def main(focus=False,teacher=False):
    from analyze import SEEDS
    cases,_,_=paired.dataset()
    images=[]
    audit=[]
    for fold in range(3):
        train,dev=[split(fold)[k] for k in ("train","dev")]
        assert len(train)==36 and not set(train)&set(dev)
        images.extend(dev)
        for seed in SEEDS:
            suffix="teacher" if teacher else "focus" if focus else "paired"
            staged=ROOT/("models_"+suffix)/f"fold{fold}"/str(seed)
            et=ROOT/("eval_"+suffix)/f"fold{fold}"/str(seed)
            methods=("extra","correction") if teacher else ("continue","branch","branch_aux") if focus else ("standard","mix")
            logs=[json.loads((staged/m/"train.json").read_text()) for m in methods]
            if teacher:
                source=ROOT/"models_focus"/f"fold{fold}"/str(seed)/"continue/weights.pt"
                with source.open("rb") as f: parent=hashlib.file_digest(f,"sha256").hexdigest()
                assert all(r["shared_source_sha"]==parent for r in logs)
                assert logs[0]["batch_trace"]==logs[1]["batch_trace"]
                select=json.loads((staged/"selection.json").read_text())
                assert [x["image"] for x in select["train"]]==train
            elif focus:
                with (ROOT/"models_paired"/f"fold{fold}"/str(seed)/"standard/weights.pt").open("rb") as original:
                    parent=hashlib.file_digest(original,"sha256").hexdigest()
                assert all(r["shared_source_sha"]==parent for r in logs)
                assert logs[0]["focused_crops"]==0<logs[1]["focused_crops"]==logs[2]["focused_crops"]
                mining=json.loads((staged/"mining.json").read_text())
                assert [c["image"] for c in mining]==train
                assert all(h<n for c in mining for h,n in zip(c["hits"],c["den"]))
            else:
                for key in ("init_sha","batch_trace","steps","batch","patch"):
                    assert logs[0][key]==logs[1][key]
                assert logs[0]["zoomed"]==0 and logs[1]["zoomed"]>0
            for method,log in zip(methods,logs):
                folder=et/method
                cal=json.loads((folder/"calibration.json").read_text())
                assert cal["train"]==train
                data=json.loads((folder/"dev.json").read_text())
                assert len(data)==18*3*3
                for name in dev:
                    d=cases[name]
                    n=d["n"];v=d["v"];f=d["f"];thin=n&d["s"];far=f&~(d["n"]|d["h"]);ids=d["i"]
                    keys,den=np.unique(ids[ids>0],return_counts=True)
                    with np.load(folder/(name+".npz")) as z:
                        for row in [r for r in data if r["image"]==name]:
                            pred=np.unpackbits(z[row["view"]+"_"+row["point"]]).reshape(v.shape).astype(bool)
                            assert not (pred&~f).any()
                            hit=pred.copy()
                            hit[1:] |= pred[:-1];hit[:-1] |= pred[1:]
                            hit[:,1:] |= pred[:,:-1];hit[:,:-1] |= pred[:,1:]
                            hits=np.bincount(ids[hit],minlength=int(ids.max()+1))[keys]
                            assert keys.tolist()==row["paths"] and den.tolist()==row["den"] and hits.tolist()==row["hits"]
                            assert int((n&~v&pred).sum())==row["near_fp"] and int(n.sum())==row["near_area"]
                            assert int((thin&hit).sum())==row["thin_hits"] and int(thin.sum())==row["thin_den"]
                            tp=int((pred&v).sum())
                            assert np.isclose(row["dice"],2*tp/(pred.sum()+v.sum()))
                            assert np.isclose(row["precision"],tp/pred.sum())
                            assert np.isclose(row["fp_fov"],(pred&~v).sum()/f.sum())
                            assert np.isclose(row["far_recall"],(pred&v&far).sum()/(v&far).sum())
                weights=staged/method/"weights.pt"
                with weights.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
                audit.append(dict(fold=fold,seed=seed,method=method,images=18,masks=len(data),amp_skips=log["amp_skips"],
                                  updates=log["steps"]-log["amp_skips"],changed_crops=log.get("zoomed",log.get("focused_crops")),
                                  weights_sha256=digest))
    assert len(images)==len(set(images))==54
    with (ROOT/("results_teacher" if teacher else "results_focus" if focus else "results")/"verification.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=audit[0]);w.writeheader();w.writerows(audit)
    print("Verified",sum(r["masks"] for r in audit),"binary masks,",len(audit),"weights.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--focus",action="store_true")
    parser.add_argument("--teacher",action="store_true")
    args=parser.parse_args()
    main(args.focus,args.teacher)
