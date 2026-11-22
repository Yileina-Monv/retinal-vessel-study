"""Recompute paired image-macro and branch transition results from every completed fold."""
import argparse
import csv
import json
from prepare import ROOT,OLD,split
import numpy as np
from analyze import METRICS,SEEDS

ARMS={"A":("standard","native"),"B":("standard","dual"),"C":("mix","native"),"D":("mix","dual"),
      "S2":("standard","large"),"M2":("mix","large")}
PAIRS=[("A","B"),("A","C"),("B","D"),("A","D"),("C","D"),("A","S2"),("S2","B")]


def read_rows(fold,seed,point,oldmodel,run):
    oldstage=OLD if fold==0 else OLD/f"fold_{fold}"
    records={}
    ids=split(fold)["dev"]
    arms=dict(ARMS)
    if run.name=="eval_focus":
        arms={"A":ARMS["A"],"B":ARMS["B"]}
        arms.update({m+suffix:(m,view) for m in ("continue","branch","branch_aux")
                     for suffix,view in (("","native"),("_dual","dual"))})
    if run.name=="eval_teacher":
        arms={"A":ARMS["A"],"B":ARMS["B"]}
        arms.update({m+s:(m,v) for m in ("continue","extra","correction") for s,v in (("","native"),("_dual","dual"))})
    for arm,(method,view) in arms.items():
        directory=ROOT/"eval_paired" if method in ("standard","mix") else run
        if method=="continue": directory=ROOT/"eval_focus"
        rows=json.loads((directory/f"fold{fold}"/str(seed)/method/"dev.json").read_text())
        rows=[r for r in rows if r["view"]==view and r["point"]==point]
        assert [r["image"] for r in rows]==ids
        records[arm]=rows
    if oldmodel:
        for arm,stage,method in (("M0_native","rgb_only","base"),("M0_dual","two_scale","base"),
                                 ("M0_special","two_scale","local")):
            rows=json.loads((oldstage/stage/f"{method}_{seed}"/"dev.json").read_text())
            assert [r["image"] for r in rows]==ids
            records[arm]=rows
    return records


def aggregate(pairs,metric,seeds):
    images=sorted({x[0] for x in pairs})
    raw=np.asarray([[[pairs[name,s][arm][metric] for name in images] for arm in (0,1)] for s in seeds],dtype=float)
    valid=np.isfinite(raw[0,0])
    assert np.all(np.isfinite(raw)==valid)
    raw=raw[:,:,valid]*100
    d=raw[:,1]-raw[:,0]
    mean=d.mean(axis=0)
    names=np.asarray(images)[valid]
    gen=np.random.default_rng(20261121)
    ix=gen.integers(len(mean),size=(10000,len(mean)))
    return dict(metric=metric,images=len(names),base=raw[:,0].mean(),candidate=raw[:,1].mean(),
                delta=mean.mean(),ci=np.quantile(mean[ix].mean(axis=1),[.025,.975]).tolist(),
                seed_delta=dict(zip(map(str,seeds),d.mean(axis=1))),up=int((mean>1e-9).sum()),
                down=int((mean< -1e-9).sum()),loo=[float((sum(mean)-x)/(len(mean)-1)) for x in (max(mean),min(mean))],
                per_image=dict(zip(names,mean)))


def compare(seeds,folds=(0,1,2),source="eval_paired",dest="results"):
    out=ROOT/dest
    out.mkdir(exist_ok=True)
    summaries={}
    details=[]
    cases=[]
    for point in ("matched","fixed","m0matched"):
        run=ROOT/source
        oldmodel=point=="m0matched"
        focus=source=="eval_focus"
        teacher=source=="eval_teacher"
        comparisons=([("A","continue"),("continue","branch"),("branch","branch_aux"),("continue","branch_aux"),("A","branch"),("A","branch_aux"),
                      ("B","continue_dual"),("continue_dual","branch_dual"),("branch_dual","branch_aux_dual"),
                      ("continue_dual","branch_aux_dual"),("B","branch_aux_dual"),("branch_aux","branch_aux_dual")] if focus else list(PAIRS))
        if teacher:
            comparisons=[("continue","extra"),("extra","correction"),("continue","correction"),("A","correction"),
                         ("continue_dual","extra_dual"),("extra_dual","correction_dual"),
                         ("continue_dual","correction_dual"),("B","correction_dual"),("correction","correction_dual"),
                         ("extra","extra_dual"),("B","extra_dual")]
        if oldmodel:
            comparisons += ([("M0_special","correction"),("M0_special","correction_dual"),("M0_special","extra"),("M0_special","extra_dual")] if teacher else
                            [("M0_special","branch_aux"),("M0_special","branch_aux_dual"),("M0_special","continue"),("M0_special","continue_dual")]
                            if focus else [("M0_native","A"),("M0_dual","B"),("M0_special","D"),("M0_native","M0_dual")])
        inputs={(f,s):read_rows(f,s,point,oldmodel,run) for f in folds for s in seeds}
        for left,right in comparisons:
            pairs={}
            transitions={str(s):dict(zero=0,zero_to_hit=0,hit_to_zero=0,zero_to_full=0,total=0) for s in seeds}
            for (fold,seed),data in inputs.items():
                for a,b in zip(data[left],data[right]):
                    image=a["image"]
                    assert b["image"]==image and (image,seed) not in pairs
                    pairs[image,seed]=(a,b)
                    if "hits" not in a or "hits" not in b: continue
                    assert a["paths"]==b["paths"] and a["den"]==b["den"]
                    u,v,den=map(np.asarray,(a["hits"],b["hits"],a["den"]))
                    row=dict(total=len(u),zero=int((u==0).sum()),zero_to_hit=int(((u==0)&(v>0)).sum()),
                             hit_to_zero=int(((u>0)&(v==0)).sum()),zero_to_full=int(((u==0)&(v==den)).sum()))
                    for key,n in row.items(): transitions[str(seed)][key]+=n
                    details.append(dict(point=point,comparison=right+"-"+left,fold=fold,seed=seed,image=image,
                                        fp_diff=b["near_fp"]-a["near_fp"],thin_diff=b["thin_hits"]-a["thin_hits"],**row))
            measures={k:aggregate(pairs,k,seeds) for k in METRICS}
            path=measures["path_d1"]["per_image"]
            near=measures["near_fp_density"]["per_image"]
            for name in sorted({k[0] for k in pairs}):
                change=path.get(name)
                fp=near.get(name)
                cases.append(dict(point=point,comparison=right+"-"+left,image=name,
                                  path_d1_change_pp=change,near_extra_fp_per_1000=None if fp is None else fp*10,
                                  thin_d1_change_pp=measures["thin_d1"]["per_image"].get(name),
                                  dice_change_pp=measures["dice"]["per_image"][name],
                                  far_recall_change_pp=measures["far_recall"]["per_image"][name]))
            byfold={str(f):float(np.mean([path[n] for n in split(f)["dev"] if n in path])) for f in folds}
            summaries[point+"_"+right+"-"+left]=dict(measures=measures,transitions=transitions,by_fold=byfold,
                     gain_more_fp=int(sum(v>1e-9 and near[k]>1e-9 for k,v in path.items())),
                     gain_no_more_fp=int(sum(v>1e-9 and near[k]<=1e-9 for k,v in path.items())),
                     worse_more_fp=int(sum(v< -1e-9 and near[k]>1e-9 for k,v in path.items())))
        if oldmodel and not focus and not teacher:
            scale_gains={}
            for (_,seed),data in inputs.items():
                for a,b,m0,m1 in zip(data["A"],data["B"],data["M0_native"],data["M0_dual"]):
                    assert a["image"]==b["image"]==m0["image"]==m1["image"]
                    def change(before,after):
                        return {k:None if before[k] is None else after[k]-before[k] for k in METRICS}
                    scale_gains[a["image"],seed]=(change(m0,m1),change(a,b))
            summaries["m0matched_scale_gain_difference"]={
                "meaning":"same-image difference of scale gains: nnU-Net (B-A) minus ordinary M0 (dual-native)",
                "measures":{k:aggregate(scale_gains,k,seeds) for k in METRICS}}
    (out/"paired.json").write_text(json.dumps(summaries,indent=2,allow_nan=False))
    with (out/"path_transitions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=details[0]);w.writeheader();w.writerows(details)
    with (out/"image_tradeoffs.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cases[0]);w.writeheader();w.writerows(cases)
    for left,right in comparisons[:4]:
        x=summaries["matched_"+right+"-"+left]["measures"]
        print(right+"-"+left,[(k,round(x[k]["delta"],2)) for k in METRICS])


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    ap.add_argument("--folds",type=int,nargs="+",default=[0,1,2])
    ap.add_argument("--dest",default="results")
    ap.add_argument("--source",choices=("eval_paired","eval_focus","eval_teacher"),default="eval_paired")
    args=ap.parse_args()
    compare(args.seeds,args.folds,source=args.source,dest=args.dest)
