"""Verify frozen N0 assets and locate common Dev omissions in ordered GT runs (CPU)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
CROSS = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def spans(mask):
    mask = np.asarray(mask, dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(boundaries[::2].tolist(), boundaries[1::2].tolist()))


def scan_run(coords, h1, h2):
    """Classify common misses within this near run only. Stop before the next run."""
    details = []
    for start, end in spans(~(h1 | h2)):
        if end - start == len(coords):
            where = "whole_near_run"
        elif start == 0 or end == len(coords):
            where = "near_run_end"
        else:
            where = "interior"
        left_shared = start > 0 and bool(h1[start-1] and h2[start-1])
        right_shared = end < len(coords) and bool(h1[end] and h2[end])
        segment = coords[start:end]
        details.append(dict(start_offset=start, end_exclusive=end, points=end-start,
                            start_yx=segment[0].tolist(), end_yx=segment[-1].tolist(),
                            arc_px=float(np.linalg.norm(np.diff(segment,axis=0),axis=1).sum()),
                            location=where, shared_hit_both_flanks=left_shared and right_shared))
    return details


def verified_assets(root):
    rows = read_csv(HERE / "asset_manifest.csv")
    assets = {}
    missing = []
    for row in rows:
        p = root / row["relative_path"]
        if not p.is_file():
            missing.append(p)
            continue
        with p.open("rb") as f:
            digest = hashlib.sha256()
            for block in iter(lambda: f.read(1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"Asset SHA mismatch: {p}")
        assets.setdefault(row["filename"], {})[row["kind"]] = p
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(rows)} assets missing. First: {missing[0]}")
    if len(rows) != 84 or len(assets) != 28 or any(set(v) != {"2027","2028","reference"} for v in assets.values()):
        raise ValueError("Expected 28 matched references and two sets of 28 N0 predictions")
    return assets


def replay(root, output):
    expected = {(r["filename"], int(r["branch_id"])): r for r in read_csv(HERE / "path_audit.csv")}
    seen = set()
    segments = []
    for name, files in verified_assets(root).items():
        with np.load(files["reference"], allow_pickle=False) as ref:
            omega = ref["Omega"].astype(bool)
            near = ref["near"].astype(bool) & omega
            coords, offsets, chosen = ref["coords"], ref["offsets"], ref["candidate_B2"]
        hit_maps = []
        for seed in ("2027","2028"):
            prob = np.load(files[seed], mmap_mode="r", allow_pickle=False)
            if prob.shape != omega.shape or prob.dtype != np.float32 or not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:
                raise ValueError(f"Invalid probability array: {name} {seed}")
            hit_maps.append(binary_dilation((prob>=0.5) & omega, structure=CROSS))
        for branch in np.flatnonzero(chosen):
            c = coords[offsets[branch]:offsets[branch+1]].astype(np.int64)
            keep = near[tuple(c.T)]
            if not keep.any():
                continue
            key = name, int(branch)
            exp = expected[key]
            distinct = np.unique(c[keep], axis=0)
            hit_counts = [int(h[tuple(distinct.T)].sum()) for h in hit_maps]
            if [len(distinct), *hit_counts] != [int(exp[k]) for k in ("near_points","hits_2027","hits_2028")]:
                raise ValueError(f"Original point counts disagree: {key}")
            seen.add(key)
            for run, (start,end) in enumerate(spans(keep)):
                points = c[start:end]
                flags = [h[tuple(points.T)] for h in hit_maps]
                for detail in scan_run(points, *flags):
                    segments.append(dict(filename=name,branch_id=int(branch),near_run=run,**detail))
    if seen != set(expected) or len(seen) != 174:
        raise ValueError("Original path membership differs")
    with output.open("x", encoding="utf-8") as f:
        json.dump(dict(paths_verified=len(seen),common_missed_spans=segments),f,indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    replay(args.project_root, args.output_json)
