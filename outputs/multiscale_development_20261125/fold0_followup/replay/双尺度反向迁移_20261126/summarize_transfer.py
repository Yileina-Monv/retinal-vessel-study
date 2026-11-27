"""Summarize the frozen transfer and paired image bootstrap (seed 2027)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARMS = ["1", "1.25", "1.5", "1+1.5"]
rows, contrasts, decomposition = [], [], []
for model in ("unet", "nnunet"):
    path = ROOT / model
    cfg = json.loads((path / "run.json").read_text())
    df = pd.read_csv(path / "per_image.csv")
    df = df[(df.point == "matched") & (df.band == "W<=3")]
    assert set(df.image) == set(cfg["target_ids"]) and len(df) == 28 * 4
    assert not df.duplicated(["image", "arm"]).any()
    for arm in ARMS:
        sub = df[df.arm == arm]
        assert np.allclose(sub.threshold, cfg["thresholds"][arm])
        rows.append(dict(model=model, arm=arm, threshold=sub.threshold.iloc[0], images=sub.D1.count(),
            points=sub.n.sum(), D1=100*sub.D1.mean(), far_recall=100*sub.far_recall.mean(),
            dice=100*sub.dice.mean(), near_fp1000=sub.near_fp1000.mean()))
    for arm, ref in [(a, "1") for a in ARMS[1:]] + [("1+1.5", "1.25"), ("1+1.5", "1.5")]:
        for metric in ["D1", "far_recall", "dice", "near_fp1000"]:
            pivot = df.pivot(index="image", columns="arm", values=metric)
            d = (pivot[arm]-pivot[ref]).dropna() * (1 if metric=="near_fp1000" else 100)
            ci = np.percentile(np.random.default_rng(2027).choice(d.to_numpy(), (20000,len(d))).mean(1), [2.5,97.5])
            leave_one = (d.sum()-d)/(len(d)-1)
            contrasts.append(dict(model=model, arm=arm, ref=ref, metric=metric, n=len(d), delta=d.mean(),
                low=ci[0], high=ci[1], gain=(d>0).sum(), tie=(d==0).sum(), loss=(d<0).sum(),
                leave_one_min=leave_one.min(), leave_one_max=leave_one.max()))
    far = pd.read_csv(path / "far_decomposition.csv")
    for (dataset, arm), part in far.groupby(["dataset","arm"]):
        assert len(part) == (18 if dataset=="IDRiD" else 28)
        assert np.allclose(part.total,part.map_change+part.threshold_change)
        if dataset=="MAPLES":
            expected=df[df.arm==arm].set_index("image").far_recall.sort_index()
            assert np.allclose(part.set_index("image").matched.sort_index(),expected)
        decomposition.append(dict(model=model,dataset=dataset,arm=arm,**{
            k:100*part[k].mean() for k in ("map_change","threshold_change","total")}))
for name, data in [("transfer_summary",rows),("paired_intervals",contrasts),("far_summary",decomposition)]:
    pd.DataFrame(data).to_csv(ROOT/(name+".csv"),index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(contrasts).query("arm=='1+1.5' and ref=='1'").round(3).to_string(index=False))
print(pd.DataFrame(decomposition).query("model=='nnunet' and arm in ['1+1.5','1.5']").round(3).to_string(index=False))
anchor = pd.read_csv(ROOT.parent / "retinal-vessel-study/outputs/scale_study_20261123/maples_source_20261124/angular/per_image.csv",dtype={"arm":str})
anchor = anchor.query("dataset=='maples28' and sampling=='nearest' and arm=='1' and W==3").set_index("image").sort_index()
for model in ("unet","nnunet"):
    actual=pd.read_csv(ROOT/model/"per_image.csv",dtype={"arm":str}).query("point=='matched' and arm=='1' and band=='W<=3'").set_index("image").sort_index()
    assert actual.index.equals(anchor.index)
    for col in ("n","v","ring"):
        assert np.array_equal(actual[col],anchor[col]), (model,col)
keys=["image","y","x","width","branch"]
pts=[pd.read_csv(ROOT/m/"matched_w3_points.csv")[keys].sort_values(keys).reset_index(drop=True) for m in ("unet","nnunet")]
assert pts[0].equals(pts[1]) and len(pts[0])==172
print("Ground truth verified per image against the archived angular-grid MAPLES study.")
