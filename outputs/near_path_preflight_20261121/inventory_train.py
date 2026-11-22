"""Count historical Train candidates inside the frozen near region. CPU, GT only."""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from replay_n0 import HERE, read_csv, spans


def label(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        if a.shape[-1] not in (3,4) or not (a[:,:,:3] == a[:,:,:1]).all():
            raise ValueError("Invalid label channels")
        if a.shape[-1] == 4 and not (a[:,:,3] == 255).all():
            raise ValueError("Unexpected alpha")
        a = a[:,:,0]
    if a.ndim != 2 or not (np.isin(a,[0,1]).all() or np.isin(a,[0,255]).all()):
        raise ValueError("Invalid binary mask")
    return a != 0


def inventory(root, output):
    files = defaultdict(dict)
    for row in read_csv(HERE / "train_manifest.csv"):
        path = root / row["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source drift: {path}")
        files[row["filename"]][row["kind"]] = path
    candidates = defaultdict(list)
    for row in read_csv(files.pop("ALL")["branches"]):
        if row["candidate_B2"] == "True":
            candidates[row["filename"]].append(int(row["branch_id"]))
    context = {r["filename"]:r for r in read_csv(HERE / "train_near_reference.csv")}
    if set(files) != set(context) or len(context) != 110 or sum(map(len,candidates.values())) != 13325:
        raise ValueError("Historical Train selection differs")
    output_rows = []
    for name, f in sorted(files.items()):
        h = label(f["Hemorrhages"])
        with np.load(f["candidate_masks_NOT_FOR_TRAINING"],allow_pickle=False) as z:
            b2 = z["B2_candidate"].astype(bool)
        if b2.shape != h.shape:
            raise ValueError("Native shape mismatch")
        near = np.zeros_like(h)
        if h.any():
            uncertain = label(f["BrightUncertains"]) | label(f["RedUncertains"])
            radius = max(1, (h.shape[1]+99)//100)
            near = (distance_transform_edt(~h)<=radius) & ~h & ~uncertain
        # Only query the frozen B2 coordinates below. They are GT within FOV.
        unique = b2 & near
        with np.load(f["path_data"],allow_pickle=False) as z:
            coords, offsets = z["coords"], z["offsets"]
        reconstructed = np.zeros_like(b2)
        paths = runs = occurrences = 0
        for branch in candidates[name]:
            c = coords[offsets[branch]:offsets[branch+1]]
            reconstructed[tuple(c.T)] = True
            inside = near[tuple(c.T)]
            if inside.any():
                paths += 1
                runs += len(spans(inside))
                occurrences += len(np.unique(c[inside],axis=0))
        if not np.array_equal(b2,reconstructed) or unique.sum() > int(context[name]["near_vessel_pixels"]):
            raise ValueError(f"Train region or candidate binding differs on {name}")
        output_rows.append(dict(filename=name,paths=paths,near_runs=runs,
                                unique_points=int(unique.sum()),path_occurrences=occurrences))
    if sum(r["unique_points"] for r in output_rows) == 0:
        raise ValueError("No eligible training target")
    with output.open("x", encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project_root",type=Path)
    p.add_argument("output_csv",type=Path)
    args=p.parse_args()
    inventory(args.project_root,args.output_csv)
