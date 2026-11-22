"""Recount paired branch transitions from native baseline masks and dual-scale RGB predictions."""
import csv
import json

import numpy as np
from scipy import ndimage
import torch

import paired
from analyze import SEEDS
from two_scale import predict


def spatial_summary(rows, folder):
    results = {}
    for far_key in ("far_thin_delta","far_same_path_delta"):
        image_deltas = []
        for name in sorted({r["image"] for r in rows}):
            rr = [r for r in rows if r["image"]==name]
            if any(r["near_delta"] is None or r[far_key] is None for r in rr): continue
            image_deltas.append([np.mean([r["near_delta"] for r in rr]),
                                 np.mean([r[far_key] for r in rr])])
        values = np.asarray(image_deltas)*100
        diff = values[:,0]-values[:,1]
        draws = np.random.default_rng(20261121).integers(len(diff),size=(10000,len(diff)))
        results[far_key] = dict(images=len(diff),near_pp=values[:,0].mean(),
                                far_pp=values[:,1].mean(),difference_pp=diff.mean(),
                                difference_ci_pp=np.quantile(diff[draws].mean(axis=1),[.025,.975]).tolist())
    (folder/"spatial_comparison.json").write_text(json.dumps(results,indent=2))
    print(results)


def main():
    torch.set_num_threads(2)
    items, _, _ = paired.dataset(include_tracks=True)
    rows = []
    for stage in (paired.HERE, paired.HERE / "fold_1", paired.HERE / "fold_2"):
        for seed in SEEDS:
            folder = stage / "two_scale" / f"local_{seed}"
            threshold = json.loads((folder / "threshold.json").read_text())["threshold"]
            check = json.loads((folder / "dev.json").read_text())
            model = paired.UNet().cuda().eval()
            weights = stage / "rgb_only" / f"local_{seed}" / "weights.pt"
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            for entry in check:
                name = entry["image"]
                d = items[name]
                _, prob = predict(model, d)
                mask = (prob >= threshold) & d["f"]
                hit = ndimage.distance_transform_edt(~mask) <= 1 if mask.any() else mask
                ids = d["i"]
                keys, den = np.unique(ids[ids > 0], return_counts=True)
                hits = np.bincount(ids[hit], minlength=int(ids.max())+1)[keys]
                with np.load(stage / "rgb_only" / f"base_{seed}" / (name+".npz")) as z:
                    assert np.array_equal(keys, z["branch_ids"]) and np.array_equal(den, z["denominators"])
                    base = z["hits"]
                    base_mask = np.unpackbits(z["prediction"]).reshape(mask.shape).astype(bool)
                base_hit = ndimage.distance_transform_edt(~base_mask) <= 1 if base_mask.any() else base_mask
                far = d["f"] & ~d["n"] & ~d["h"]
                tracks = d["all_tracks"]
                assert np.array_equal(np.where(d["n"],tracks,0),ids)
                def paired_coverage(where):
                    labels = tracks[where & (tracks>0)]
                    denoms = np.bincount(labels, minlength=int(tracks.max())+1)
                    valid = denoms>0
                    if not valid.any(): return None
                    def mean(hit_mask):
                        count = np.bincount(tracks[where & (tracks>0) & hit_mask],minlength=len(denoms))
                        return (count[valid]/denoms[valid]).mean()
                    return float(mean(hit)-mean(base_hit))
                assert (float(np.mean(hits/den)) if len(den) else None) == entry["path_d1"]
                area = int(d["n"].sum())
                fp = int((mask & d["n"] & ~d["v"]).sum())
                assert (fp/area if area else None) == entry["near_fp_density"]
                rows.append(dict(seed=seed,image=name,paths=len(keys),base_zero=int((base==0).sum()),
                                 zero_to_hit=int(((base==0)&(hits>0)).sum()),hit_to_zero=int(((base>0)&(hits==0)).sum()),
                                 zero_to_full=int(((base==0)&(hits==den)).sum()),improved=int((hits>base).sum()),
                                 declined=int((hits<base).sum()),near_fp=fp,near_area=area,
                                 near_delta=paired_coverage(d["n"]),far_thin_delta=paired_coverage(far),
                                 far_same_path_delta=paired_coverage(far & np.isin(tracks,keys))))
    assert len({(r["seed"], r["image"]) for r in rows}) == len(rows) == 162
    out = paired.HERE / "out_of_fold/all/two_scale/vs_native/path_transitions.csv"
    with out.open("w", newline="") as output:
        writer = csv.DictWriter(output,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for seed in SEEDS:
        print(seed,{k:sum(r[k] for r in rows if r["seed"]==seed) for k in list(rows[0])[2:-3]})
    spatial_summary(rows,out.parent)


if __name__ == "__main__":
    main()
