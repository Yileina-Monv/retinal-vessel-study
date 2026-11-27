"""Frozen IDRiD fold-0 weights and four pre-calibrated scale arms on MAPLES Dev28."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNNER = ROOT / "Colab_训练推理补充_20261126/runner"
WORK = ROOT / ".work/colab_stagecheck/work"
FOLD = ROOT / "首折双尺度_本地GPU_20261126/idrid_fold0"
sys.path.insert(0, str(RUNNER))
import scale_eval as ev


def read_case(folder):
    def load(name, mode="L"):
        with Image.open(folder / name) as im:
            return np.asarray(im.convert(mode)).copy()
    fov = load("fov.png") > 0
    rgb = load("image.tif", "RGB")
    vessel = (load("vessel.png") > 0) & fov
    valid = fov & (load("uncertain.png") == 0)
    lesion = load("lesion.png") > 0
    assert rgb.shape[:2] == vessel.shape == valid.shape == lesion.shape
    return dict(rgb=rgb, v=vessel, f=valid, h=lesion)


def far_decomposition(maps, a, thresholds):
    far = a["scored"] & a["v"] & ~a["h"] & ~a["ring"]
    if not far.any():
        raise ValueError("Empty far-vessel mask")
    base = maps["1"][far]
    t0 = thresholds["1"]
    rows = []
    for arm, probability in maps.items():
        v = probability[far]
        base_rate = (base >= t0).mean()
        same = (v >= t0).mean()
        final = (v >= thresholds[arm]).mean()
        rows.append(dict(arm=arm, n_far=int(far.sum()), native=base_rate, common=same,
                         matched=final, map_change=same-base_rate, threshold_change=final-same,
                         total=final-base_rate))
    return rows


def main():
    torch.set_num_threads(2)
    p, _, geometry = ev.setup(WORK)
    iddata, train, dev = p.dataset(0)
    args = SimpleNamespace(device="cuda", maples_seed=None, unet_weight=WORK / "unet-idrid/state_dict.pt",
        nnunet_weight=WORK / "outputs/nnunet_scale_20261121/models_paired/fold0/2027/standard/weights.pt")
    ids = sorted(ev.read_json(RUNNER / "maples_split.json")["dev"])
    source = WORK / "datasets/maples/dev"
    assert len(ids) == 28 and ids == sorted(f.name for f in source.iterdir() if f.is_dir())
    for kind in ("unet", "nnunet"):
        dest = HERE / kind
        dest.mkdir(exist_ok=True)
        run = ev.read_json(FOLD / kind / "run.json")
        calfile = FOLD / kind / "calibration.json"
        cal = ev.read_json(calfile)
        assert cal["train"] == run["train"] == train and run["dev"] == dev
        weight = getattr(args, kind + "_weight")
        assert ev.sha(weight) == run["weight_sha256"]
        ts = cal["thresholds"]
        cfg = dict(model=kind, source="Refined IDRiD fold0", target="MAPLES Dev28", train=train,
                   target_ids=ids, thresholds=ts, weight_sha256=ev.sha(weight), calibration_sha256=ev.sha(calfile),
                   runner_sha256=ev.sha(RUNNER / "scale_eval.py"), experiment_sha256=ev.sha(__file__),
                   device=torch.cuda.get_device_name(), torch=torch.__version__, angle=45)
        assert cfg["runner_sha256"] in run["source_sha256"].values()
        if (dest / "run.json").exists():
            assert ev.read_json(dest / "run.json") == cfg
        ev.atomic_json(dest / "run.json", cfg)
        model = ev.Model(kind, args, p)
        for i, name in enumerate(ids, 1):
            target = dest / "dev" / (name + ".json")
            if target.exists():
                continue
            d = read_case(source / name)
            prob, cost = model.probabilities(d["rgb"])
            a = ev.angular(d, 45., geometry)
            maps = ev.maps_for_arms(prob, a)
            rows, branches = ev.score_angular(maps, a, ts)
            ev.atomic_json(target, dict(image=name, rows=rows, branches=branches,
                points=ev.matched_points(maps, a, ts), geometry=a["geometry"], cost=cost,
                replay=dict(status="no_reference"), far=far_decomposition(maps, a, ts)))
            print(kind, i, len(ids), name, flush=True)
        ev.summarize(dest, ids)
        del model
        torch.cuda.empty_cache()
        records = []
        for name in dev:
            a = ev.angular(iddata[name], 50., geometry)
            with np.load(FOLD / kind / "probabilities" / (name + ".npz")) as z:
                maps = {arm:z["angular_" + arm] for arm in ev.ARMS}
            rows = far_decomposition(maps, a, ts)
            orig = ev.read_json(FOLD / kind / "dev" / (name + ".json"))["rows"]
            for r in rows:
                ref = next(v for v in orig if v["arm"] == r["arm"] and v["point"] == "matched")
                assert abs(r["matched"] - ref["far_recall"]) < 1e-9
                records.append(dict(dataset="IDRiD", image=name, **r))
        for name in ids:
            records.extend(dict(dataset="MAPLES", image=name, **r)
                           for r in ev.read_json(dest / "dev" / (name + ".json"))["far"])
        ev.csv_write(dest / "far_decomposition.csv", records)


if __name__ == "__main__":
    main()
