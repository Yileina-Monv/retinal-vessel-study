#!/usr/bin/env python3
"""Frozen fold-0 inference: native replay, scale arms, angular matched-FP scoring.

No training is launched. Historical Python files remain the inference authority.
Only the declared calibration cohort enters threshold selection.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F

ARMS = {"1": (1.,), "1.25": (1.25,), "1.5": (1.5,), "1+1.5": (1., 1.5)}
GRID = np.arange(129, dtype=np.float64) / 128
CROSS = ndi.generate_binary_structure(2, 1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def avg(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def ratio(a, b):
    return float(a / b) if b else None


def setup(work):
    old = work / "outputs/refined_idrid_20261121"
    nn = work / "outputs/nnunet_scale_20261121"
    sys.path[:0] = [str(nn), str(old), str(work)]
    # prepare.py normally sets these. Set before importing nnU-Net as well.
    for key, sub in (("nnUNet_raw", "raw"), ("nnUNet_preprocessed", "preprocessed"),
                     ("nnUNet_results", "framework_runs")):
        os.environ[key] = str(nn / sub)
    os.environ["nnUNet_compile"] = "false"
    p = importlib.import_module("paired")
    score = importlib.import_module("two_scale").score
    # Extract only the original geometry function, whose module imports absent shrink.run.
    source = work / "angle_reference/run_grid.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "geometry")
    namespace = {"cv2": cv2, "np": np, "ndi": ndi}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return p, score, namespace["geometry"]


def load_dataset(args, p):
    if args.maples_seed:
        data, _, _ = p.dataset(0)
        dev = sorted(data)
        base = args.work / "datasets/maples/dev"
        folders = sorted(f for f in base.iterdir() if f.is_dir() and (f / "image.tif").is_file())
        if len(folders) != 28:
            raise ValueError(f"MAPLES Dev28 requires exactly 28 image folders, got {len(folders)}")
        calibration = []
        for folder in folders:
            name = "MAPLES_" + folder.name
            if name in data:
                raise ValueError("Duplicate dataset identifier")
            def read(file, rgb=False):
                with Image.open(folder / file) as im:
                    return np.asarray(im.convert("RGB" if rgb else "L")).copy()
            rgb = read("image.tif", True)
            fov = read("fov.png") > 0
            valid = fov & ~(read("uncertain.png") > 0)
            # Preserve complete GT geometry; uncertainty only gates scored/support.
            # This matches the frozen angular reference: vessel & FOV, raw lesion.
            vessel, he = (read("vessel.png") > 0) & fov, read("lesion.png") > 0
            if any(x.shape != rgb.shape[:2] for x in (valid, vessel, he)):
                raise ValueError(f"MAPLES array shape mismatch: {name}")
            data[name] = dict(rgb=rgb, v=vessel, f=valid, h=he)
            calibration.append(name)
        if args.maples_target == "maples":
            return data, calibration, calibration, 45., f"maples_seed{args.maples_seed}_dev28_recheck"
        return data, calibration, dev, 50., f"maples_seed{args.maples_seed}_to_idrid54"
    if not args.dataset_manifest:
        data, train, dev = p.dataset(0)
        frozen = read_json(args.work / "outputs/refined_idrid_20261121/split.json")
        if frozen != {"train": train, "dev": dev}:
            raise ValueError("Historical fold-0 split drift")
        return data, train, dev, 50., "idrid_fold0"
    path = args.dataset_manifest.resolve()
    meta = read_json(path)
    train, dev = meta["train"], meta["dev"]
    if not train or not dev or set(train) & set(dev) or len(set(train + dev)) != len(train + dev):
        raise ValueError("Manifest requires disjoint nonempty unique train and dev names")
    data = {}
    for name in train + dev:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Unsafe image identifier")
        row = meta["images"][name]
        def read(key, rgb=False):
            file = Path(row[key])
            if not file.is_absolute():
                file = path.parent / file
            with Image.open(file) as im:
                return np.asarray(im.convert("RGB" if rgb else "L")).copy()
        rgb = read("rgb", True)
        valid = read("valid") > 0
        fov = read("fov") > 0 if "fov" in row else np.ones(valid.shape, bool)
        vessel, he = (read("vessel") > 0) & fov, read("he") > 0
        if any(a.shape != rgb.shape[:2] for a in (valid, vessel, he)):
            raise ValueError(f"Label size mismatch: {name}")
        near = (ndi.distance_transform_edt(~he) <= int(np.ceil(rgb.shape[1] / 100))) & ~he & valid if he.any() else np.zeros_like(he)
        sk = skeletonize(vessel)
        data[name] = dict(rgb=rgb, v=vessel, f=valid, h=he, n=near,
                          s=sk & (2 * ndi.distance_transform_edt(vessel) <= 3),
                          i=np.zeros(valid.shape, np.int32))
    return data, train, dev, float(meta.get("nominal_angle", 45)), meta.get("dataset", "custom")


class Model:
    def __init__(self, kind, args, p):
        self.kind, self.args, self.p = kind, args, p
        self.path = args.unet_weight if kind == "unet" else args.nnunet_weight
        if kind == "unet":
            self.model = p.UNet().to(args.device).eval()
            weights = torch.load(self.path, map_location="cpu", weights_only=True)
            if "model" in weights:
                weights = weights["model"]
            self.model.load_state_dict(weights, strict=True)
        else:
            if args.device != "cuda":
                raise ValueError("Historical nnU-Net requires --device cuda")
            # Reuse original trainer/plans/network/predictor construction, never train_pair().
            from train import trainer
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            t = trainer(0, 2027)
            t.network.load_state_dict(torch.load(self.path, map_location="cpu", weights_only=True))
            t.set_deep_supervision_enabled(False)
            self.pred = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                       device=torch.device("cuda"), allow_tqdm=False)
            self.pred.manual_initialization(t.network.eval(), t.plans_manager, t.configuration_manager,
                                             None, t.dataset_json, type(t).__name__, None)

    @torch.inference_mode()
    def probabilities(self, rgb, scales=(1., 1.25, 1.5)):
        result, costs = {}, {}
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True,
                                        allow_tf32=torch.backends.cudnn.allow_tf32):
            if self.kind == "unet":
                raw = torch.from_numpy(rgb).permute(2, 0, 1)
                if self.args.maples_seed:
                    cc, _ = ndi.label(rgb.max(2) > 10, np.ones((3, 3), bool))
                    field = ndi.binary_fill_holes(cc == 1 + np.argmax(np.bincount(cc.ravel())[1:]))
                    x = ((raw / 255 - self.p.mean) / self.p.std) * torch.from_numpy(field)[None]
                else:
                    x = self.p.normalize(raw)
                for s in scales:
                    start = time.monotonic()
                    a = x if s == 1 else F.interpolate(x[None], size=(round(x.shape[-2]*s), round(x.shape[-1]*s)),
                                                       mode="bilinear", align_corners=False)[0]
                    hh, ww = a.shape[-2:]
                    a = F.pad(a, (0, max(0, 512-ww), 0, max(0, 512-hh)))
                    pr, windows = self.p.tiled_probability(self.model, a, patch=512, stride=256,
                        batch_size=4 if self.args.device == "cuda" else 1,
                        device=self.args.device, amp=self.args.device == "cuda")
                    pr = pr[:hh, :ww]
                    if s != 1:
                        pr = F.interpolate(pr[None, None], size=rgb.shape[:2], mode="area")[0, 0]
                    result[s] = pr.numpy()
                    costs[str(s)] = dict(windows=windows, seconds=time.monotonic()-start)
            else:
                pred = self.pred
                props = {"spacing": (999, 1, 1)}
                cfg = pred.configuration_manager
                a, _, _ = cfg.preprocessor_class(verbose=False).run_case_npy(
                    rgb.transpose(2, 0, 1)[:, None], None, props, pred.plans_manager, cfg, pred.dataset_json)
                if list(pred.plans_manager.transpose_forward) != [0, 1, 2]:
                    raise ValueError("Unsupported nnU-Net transpose")
                x = torch.from_numpy(np.ascontiguousarray(a))
                sl = tuple(slice(*b) for b in props["bbox_used_for_cropping"][1:])
                for s in scales:
                    start = time.monotonic()
                    big = x if s == 1 else F.interpolate(x[:, 0][None], size=(round(x.shape[-2]*s), round(x.shape[-1]*s)),
                                                         mode="bilinear", align_corners=False)[0, :, None]
                    pr = pred.predict_sliding_window_return_logits(big).float().softmax(0)[1, 0].cpu()
                    if s != 1:
                        pr = F.interpolate(pr[None, None], size=x.shape[-2:], mode="area")[0, 0]
                    out = np.zeros(rgb.shape[:2], np.float32)
                    out[sl] = pr.numpy()
                    result[s] = out
                    costs[str(s)] = dict(seconds=time.monotonic()-start)
        if any(not np.isfinite(a).all() or a.shape != rgb.shape[:2] for a in result.values()):
            raise ValueError("Invalid probability map")
        return result, costs


def angular(d, angle, geometry):
    field, coords, roi, geom = geometry(d["rgb"], angle)
    def warp(a, mode=cv2.INTER_NEAREST):
        return cv2.remap(a.astype(np.float32), coords, None, mode,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = warp(d["f"] & field) > .5
    scored = support & roi
    vessel, lesion = warp(d["v"]) >= .5, warp(d["h"]) >= .5
    ring = scored & ~lesion & (ndi.distance_transform_edt(~lesion) <= 9) if lesion.any() else np.zeros_like(scored)
    sk = skeletonize(vessel)
    width = 2 * ndi.distance_transform_edt(vessel)
    near_sk = sk & ring
    neighbours = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    branches, _ = ndi.label(sk & (neighbours <= 2), structure=np.ones((3, 3), bool))
    # Freeze branch IDs from the complete remapped GT skeleton before ring/width clipping.
    return dict(warp=warp, support=support, scored=scored, v=vessel, h=lesion, ring=ring,
                sk=near_sk, width=width, branches=branches,
                geometry=dict(zip("arc_points cx cy axis1 axis2 rotation median_residual p95_residual".split(), map(float, geom)),
                              scored_fraction=float(scored.sum()/roi.sum())))


def maps_for_arms(prob, a):
    warped = {s: a["warp"](pr, cv2.INTER_LINEAR) for s, pr in prob.items()}
    return {arm: sum(warped[s] for s in scales) / len(scales) for arm, scales in ARMS.items()}


def curves(arms, a):
    den = int(a["ring"].sum())
    if not den:
        return None
    sel = a["ring"] & ~a["v"]
    return {arm: ((sel.sum()-np.searchsorted(np.sort(pr[sel]), GRID))/den).tolist() for arm, pr in arms.items()}


def score_angular(arms, a, thresholds):
    rows, branch_rows = [], []
    for point, ts in (("fixed", {arm: .5 for arm in ARMS}), ("matched", thresholds)):
        native = (arms["1"] >= ts["1"]) & a["support"]
        native_hit = ndi.binary_dilation(native, structure=CROSS)
        for arm, pr in arms.items():
            pred = (pr >= ts[arm]) & a["support"]
            hit = ndi.binary_dilation(pred, structure=CROSS)
            valid_v = a["v"] & a["scored"]
            far = valid_v & ~a["h"] & ~a["ring"]
            tp = int((pred & valid_v).sum())
            fp = int((pred & ~a["v"] & a["scored"]).sum())
            common = dict(point=point, arm=arm, threshold=ts[arm], near_fp=int((pred & a["ring"] & ~a["v"]).sum()),
                          ring=int(a["ring"].sum()), tp=tp, fp=fp, v=int(valid_v.sum()),
                          near_fp1000=ratio(1000*(pred & a["ring"] & ~a["v"]).sum(), a["ring"].sum()),
                          far_recall=ratio((pred & far).sum(), far.sum()), dice=ratio(2*tp, tp+fp+valid_v.sum()))
            for band, lo, hi in (("W<=3", 0, 3), ("W<=4", 0, 4), ("W<=5", 0, 5),
                                 ("3<W<=4", 3, 4), ("4<W<=5", 4, 5)):
                points = a["sk"] & (a["width"] > lo) & (a["width"] <= hi)
                n = int(points.sum())
                gained = int((points & hit & ~native_hit).sum())
                lost = int((points & native_hit & ~hit).sum())
                rows.append(dict(**common, band=band, n=n, d0=int((points & pred).sum()), d1=int((points & hit).sum()),
                                 D0=ratio((points & pred).sum(), n), D1=ratio((points & hit).sum(), n),
                                 native_D1=ratio((points & native_hit).sum(), n), delta_D1=ratio(gained-lost, n),
                                 gained_points=gained, lost_points=lost))
                ids = a["branches"][points]
                keys, den = np.unique(ids[ids > 0], return_counts=True)
                for key, denominator in zip(keys, den):
                    pts = points & (a["branches"] == key)
                    bh, bn = int((pts & hit).sum()), int((pts & native_hit).sum())
                    branch_rows.append(dict(point=point, arm=arm, band=band, branch=int(key), n=int(denominator),
                        native_hits=bn, hits=bh, gained=int((pts & hit & ~native_hit).sum()),
                        lost=int((pts & native_hit & ~hit).sum()), delta_hits=bh-bn,
                        outcome="gain" if bh > bn else "loss" if bh < bn else "tie"))
    return rows, branch_rows


def replay(prob, d, name, reference, score):
    got = score(d, prob, .5) if all(k in d for k in ("n", "i", "s")) else None
    ref = next((r for r in reference if r["image"] == name), None)
    if ref is None:
        return dict(status="no_reference", metrics=got)
    differences = {k: dict(expected=ref[k], actual=value) for k, value in got.items()
                   if k in ref and ((value is None) != (ref[k] is None) or
                       value is not None and not np.isclose(value, ref[k], atol=1e-7, rtol=0))}
    return dict(status="pass" if not differences else "drift", differences=differences, metrics=got,
                comparison="original native-grid fixed-0.5 metrics; not angular metrics")


def matched_points(arms, a, thresholds):
    """Common W<=3 angular coordinates for a cross-model (image,y,x) join."""
    points = a["sk"] & (a["width"] <= 3)
    yy, xx = np.nonzero(points)
    masks = {}
    for arm, pr in arms.items():
        pred = (pr >= thresholds[arm]) & a["support"]
        masks["D0_" + arm] = pred
        masks["D1_" + arm] = ndi.binary_dilation(pred, structure=CROSS)
    return [dict(y=int(y), x=int(x), width=float(a["width"][y, x]), branch=int(a["branches"][y, x]),
                 **{key: int(mask[y, x]) for key, mask in masks.items()}) for y, x in zip(yy, xx)]


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(dest, dev):
    records = [read_json(dest / "dev" / f"{n}.json") for n in dev]
    rows = [dict(image=r["image"], **row) for r in records for row in r["rows"]]
    branches = [dict(image=r["image"], **row) for r in records for row in r["branches"]]
    csv_write(dest / "per_image.csv", rows)
    csv_write(dest / "branches.csv", branches)
    csv_write(dest / "matched_w3_points.csv", [dict(image=r["image"], **pt) for r in records for pt in r["points"]])
    csv_write(dest / "geometry.csv", [dict(image=r["image"], **r["geometry"]) for r in records])
    summary = []
    for point in ("fixed", "matched"):
        for arm in ARMS:
            for band in ("W<=3", "W<=4", "W<=5", "3<W<=4", "4<W<=5"):
                rr = [r for r in rows if (r["point"], r["arm"], r["band"]) == (point, arm, band)]
                bb = [r for r in branches if (r["point"], r["arm"], r["band"]) == (point, arm, band)]
                summary.append(dict(point=point, arm=arm, band=band, images=sum(r["n"] > 0 for r in rr),
                    points=sum(r["n"] for r in rr), D0=avg([r["D0"] for r in rr]), D1=avg([r["D1"] for r in rr]),
                    delta_D1=avg([r["delta_D1"] for r in rr]),
                    near_fp1000=avg([r["near_fp1000"] for r in rr]), far_recall=avg([r["far_recall"] for r in rr]),
                    dice=avg([r["dice"] for r in rr]), gained_points=sum(r["gained_points"] for r in rr),
                    lost_points=sum(r["lost_points"] for r in rr), branch_gain=sum(r["outcome"] == "gain" for r in bb),
                    branch_loss=sum(r["outcome"] == "loss" for r in bb), branch_tie=sum(r["outcome"] == "tie" for r in bb)))
    csv_write(dest / "summary.csv", summary)
    atomic_json(dest / "native_replay.json", [{"image": r["image"], **r["replay"]} for r in records])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True, help="Root containing extracted outputs/, retinal_m0/, angle_reference/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=("unet", "nnunet", "both"), default="both")
    ap.add_argument("--unet-weight", type=Path)
    ap.add_argument("--nnunet-weight", type=Path)
    ap.add_argument("--dataset-manifest", type=Path)
    ap.add_argument("--maples-seed", type=int, choices=(2027, 2028), help="Calibrate on MAPLES Dev28; evaluate IDRiD development54")
    ap.add_argument("--maples-target", choices=("idrid", "maples"), default="idrid", help="With --maples-seed: evaluate IDRiD54 or recheck MAPLES Dev28 (same cohort used for calibration)")
    ap.add_argument("--budget", type=float, default=.015, help="Train macro near-FP density; .015 = 15/1000 ring pixels")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--smoke", action="store_true", help="One dev image, all scales, native replay; no threshold fitting or dev report")
    ap.add_argument("--image", help="Dev image for --smoke (default first); not an evaluation subset")
    ap.add_argument("--keep-probabilities", action="store_true")
    ap.add_argument("--strict-replay", action="store_true", help="Stop on native metric differences; default records differences and continues across hardware")
    ap.add_argument("--run-label", default="idrid_fold0_seed2027")
    args = ap.parse_args()
    if args.maples_seed:
        if args.dataset_manifest or args.model == "nnunet":
            ap.error("--maples-seed uses the built-in transfer dataset and U-Net only")
        args.model = "unet"
        args.run_label = f"maples_seed{args.maples_seed}_transfer" if args.run_label == "idrid_fold0_seed2027" else args.run_label
    args.work, args.out = args.work.resolve(), args.out.resolve()
    if not 0 <= args.budget <= 1:
        ap.error("--budget must be in [0,1]")
    if args.image and not args.smoke:
        ap.error("--image requires --smoke; full calibration/evaluation cannot use a subset")
    args.unet_weight = (args.unet_weight or args.work / (f"unet-maples-{args.maples_seed}/state_dict.pt" if args.maples_seed else "unet-idrid/state_dict.pt")).resolve()
    args.nnunet_weight = (args.nnunet_weight or args.work / "outputs/nnunet_scale_20261121/models_paired/fold0/2027/standard/weights.pt").resolve()
    torch.set_num_threads(2)
    p, original_score, geometry = setup(args.work)
    data, train, dev, angle, dataset = load_dataset(args, p)
    kinds = ("unet", "nnunet") if args.model == "both" else (args.model,)
    data_hash = hashlib.sha256()
    for name in train + dev:
        data_hash.update(name.encode())
        for key in ("rgb", "v", "f", "h"):
            data_hash.update(data[name][key].tobytes())
    sources = [Path(__file__), args.work / "retinal_m0/model.py", args.work / "retinal_m0/inference.py",
               args.work / "outputs/refined_idrid_20261121/paired.py", args.work / "outputs/refined_idrid_20261121/two_scale.py", args.work / "angle_reference/run_grid.py",
               args.work / "releases/development_v1/normalization.json"]
    for kind in kinds:
        weight = args.unet_weight if kind == "unet" else args.nnunet_weight
        dest = args.out / kind
        dest.mkdir(parents=True, exist_ok=True)
        source_files = list(sources)
        if kind == "nnunet":
            source_files += [args.work / "outputs/nnunet_scale_20261121" / f for f in (
                "train.py", "evaluate.py", "prepare.py", "preprocessed/Dataset871_RefinedVessel/nnUNetPlans.json",
                "raw/Dataset871_RefinedVessel/dataset.json")]
        config = dict(version=1, model=kind, weight_sha256=sha(weight), dataset=dataset,
            dataset_sha256=data_hash.hexdigest(), train=train, dev=dev, nominal_angle=angle,
            angular_grid=dict(size=1021, center=[510, 510], pixels_per_degree=20, scoring_radius=450,
                              point_coordinates="zero-based array y,x indices; join on image,y,x"),
            branch_definition="8-connected components of full angular GT skeleton with <=2 eight-neighbours; ring/width clipped afterwards; no minimum length filter",
            budget=args.budget, budget_aggregation="macro over calibration images with nonempty angular ring",
            calibration_cohort="MAPLES Dev28" if args.maples_seed else "train",
            evaluation_cohort=("MAPLES Dev28 calibration-cohort recheck" if args.maples_target == "maples" else "IDRiD development54") if args.maples_seed else "dev",
            arms=ARMS, source_sha256={str(f.relative_to(args.work)) if f.is_relative_to(args.work) else f.name: sha(f) for f in source_files},
            device=args.device, torch=torch.__version__, python=platform.python_version(),
            numpy=np.__version__, opencv=cv2.__version__, cudnn=torch.backends.cudnn.version(),
            cuda=torch.version.cuda, allow_tf32=torch.backends.cudnn.allow_tf32,
            keep_probabilities=args.keep_probabilities, strict_replay=args.strict_replay, run_label=args.run_label,
            maples_source_field=bool(args.maples_seed))
        if (dest / "run.json").exists() and read_json(dest / "run.json") != json.loads(json.dumps(config)):
            raise ValueError(f"Run fingerprint changed; choose a fresh --out: {dest}")
        atomic_json(dest / "run.json", config)
        reference_path = (args.work / "unet-idrid/dev_reference.json" if kind == "unet" else
            args.work / "outputs/nnunet_scale_20261121/eval_paired/fold0/2027/standard/dev.json")
        reference = [] if args.dataset_manifest or args.maples_seed or not reference_path.exists() else read_json(reference_path)
        if kind == "nnunet":
            reference = [r for r in reference if r.get("view") == "native" and r.get("point") == "fixed"]
        # Non-IDRiD source weights have no matching historical IDRiD replay reference.
        if kind == "unet" and (args.work / "unet-idrid/model.json").exists():
            if config["weight_sha256"] != read_json(args.work / "unet-idrid/model.json")["state_sha256"]:
                reference = []
        model = None
        def predict(d):
            nonlocal model
            if model is None:
                model = Model(kind, args, p)
            return model.probabilities(d["rgb"])
        if args.smoke:
            name = args.image or dev[0]
            if name not in dev:
                raise ValueError("Smoke image must be in dev")
            d = data[name]
            prob, cost = predict(d)
            a = angular(d, angle, geometry)
            check = replay(prob[1.], d, name, reference, original_score)
            smoke_arms = maps_for_arms(prob, a)
            smoke_rows, smoke_branches = score_angular(smoke_arms, a, {arm: .5 for arm in ARMS})
            atomic_json(dest / f"smoke_{name}.json", dict(image=name, scales=list(prob), cost=cost,
                         geometry=a["geometry"], replay=check,
                         fixed_rows=[r for r in smoke_rows if r["point"] == "fixed"],
                         fixed_branch_rows=sum(r["point"] == "fixed" for r in smoke_branches),
                         fixed_point_rows=len(matched_points(smoke_arms, a, {arm: .5 for arm in ARMS})),
                         calibration_curve_length=len(next(iter(curves(smoke_arms, a).values()))) if a["ring"].any() else 0,
                         scope="single-image execution check, no fitted thresholds"))
            print(kind, "smoke", name, check["status"], flush=True)
            if check["status"] == "drift" and args.strict_replay:
                raise RuntimeError("Native replay drift; inspect smoke JSON")
            continue
        # Only train probabilities enter threshold calibration. Each image is committed atomically.
        for index, name in enumerate(train, 1):
            target = dest / "train" / f"{name}.json"
            if target.exists():
                continue
            d = data[name]
            prob, cost = predict(d)
            a = angular(d, 45. if args.maples_seed else angle, geometry)
            atomic_json(target, dict(image=name, curves=curves(maps_for_arms(prob, a), a),
                                     geometry=a["geometry"], cost=cost))
            print(kind, "calibration", index, len(train), name, flush=True)
        samples = [read_json(dest / "train" / f"{name}.json")["curves"] for name in train]
        samples = [x for x in samples if x is not None]
        if not samples:
            raise ValueError("No nonempty train angular lesion ring")
        thresholds, achieved = {}, {}
        for arm in ARMS:
            curve = np.mean([x[arm] for x in samples], axis=0)
            eligible = np.flatnonzero(curve <= args.budget)
            if not len(eligible):
                raise ValueError(f"Budget unreachable on frozen 0..128 / 128 grid: {arm}")
            i = int(eligible[0])
            thresholds[arm], achieved[arm] = float(GRID[i]), float(curve[i])
        atomic_json(dest / "calibration.json", dict(train=train, effective_images=len(samples), budget=args.budget,
            thresholds=thresholds, achieved=achieved, grid="0..128 / 128", selected="lowest threshold meeting calibration macro budget"))
        for index, name in enumerate(dev, 1):
            target = dest / "dev" / f"{name}.json"
            if target.exists():
                saved = read_json(target)
                if saved["replay"]["status"] == "drift" and args.strict_replay:
                    raise RuntimeError(f"Cached native replay drift: {name}")
                continue
            d = data[name]
            prob, cost = predict(d)
            a = angular(d, angle, geometry)
            arms = maps_for_arms(prob, a)
            check = replay(prob[1.], d, name, reference, original_score)
            rows, branches = score_angular(arms, a, thresholds)
            if args.keep_probabilities:
                cache = dest / "probabilities"
                cache.mkdir(exist_ok=True)
                tmp = cache / f"{name}.tmp.npz"
                np.savez_compressed(tmp, **{f"native_{s}": pr for s, pr in prob.items()},
                                    **{f"angular_{arm}": pr for arm, pr in arms.items()})
                tmp.replace(cache / f"{name}.npz")
            atomic_json(target, dict(image=name, rows=rows, branches=branches, geometry=a["geometry"],
                                     points=matched_points(arms, a, thresholds), replay=check, cost=cost))
            print(kind, "dev", index, len(dev), name, "native replay", check["status"], flush=True)
            if check["status"] == "drift" and args.strict_replay:
                raise RuntimeError("Native replay drift: inspect dev JSON; omit --strict-replay to continue with drift recorded")
        summarize(dest, dev)
        print("Completed", dest, flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
