"""Protocol primitives for the MAPLES explicit-area 2/1 class split."""
from __future__ import annotations
import json
import time
from pathlib import Path
from PIL import Image
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from retinal_m0.maples_core import (build_training_regions, paired_weighted_crop, paired_crop_transform,
                                    CorePatches, draw_geometry, _decode_label_image)

GROUPS = ("A_V1", "A_V0", "O_V1", "O_V0")
ALLOWED_WEIGHTS = ((1.0, 1.0), (2.0, 2.0), (2.0, 1.0))


def _binary_array(value, name):
    a = np.asarray(value)
    if a.ndim != 2 or not np.isin(a, (0, 1, False, True)).all():
        raise ValueError(f"{name} must be a binary 2D mask")
    return a.astype(bool, copy=False)


def build_class_split_regions(fov, hemorrhages, uncertain, *, image_width):
    """Build the frozen hemorrhage interior+neighborhood A on the full image."""
    r = build_training_regions(fov, hemorrhages, uncertain, image_width=image_width)
    area = (r["rin"] | r["rnear"]).astype(bool, copy=False)
    if np.any(area & ~np.asarray(fov, dtype=bool)):
        raise RuntimeError("A must be inside FOV")
    return {**r, "area": area}


def class_split_weights(fov, vessel, area, *, vessel_weight=2.0, background_weight=1.0):
    """Construct w from explicit geometry A and binary vessel labels; never infer A from w."""
    f = _binary_array(fov, "FOV")
    v = _binary_array(vessel, "V")
    a = _binary_array(area, "A")
    if f.shape != v.shape or f.shape != a.shape:
        raise ValueError("FOV, vessel, and explicit A masks must be aligned")
    if np.any(a & ~f):
        raise ValueError("A must be a subset of FOV")
    pair = (float(vessel_weight), float(background_weight))
    if pair not in ALLOWED_WEIGHTS:
        raise ValueError("Only the approved 1/1, 2/2, and 2/1 weights are supported")
    w = np.zeros(f.shape, dtype=np.float32)
    w[f] = 1.0
    # Keep the A map separately: A_V0 has coefficient 1 in 2/1 but remains in A.
    w[f & a & v] = np.float32(vessel_weight)
    w[f & a & ~v] = np.float32(background_weight)
    expected = np.zeros(f.shape, dtype=np.float32)
    expected[f] = 1.0
    expected[f] += (a[f] * ((pair[0] - 1.0) * v[f] + (pair[1] - 1.0) * (~v[f]))).astype(np.float32)
    if not np.array_equal(w, expected):
        raise RuntimeError("Class-split weight formula mismatch")
    if np.any(w[~f] != 0) or np.any(w[f & ~a] != 1):
        raise RuntimeError("FOV/A weight invariant failed")
    return w


def paired_class_split_crop(image, vessel, fov, area, *, top, left, size,
                            hflip=False, vflip=False, quarter_turns=0,
                            vessel_weight=2.0, background_weight=1.0):
    """Apply identical frozen crop/augmentation to RGB, V, FOV, A, and full-image weights."""
    image = np.asarray(image)
    v = _binary_array(vessel, "V")
    f = _binary_array(fov, "FOV")
    a = _binary_array(area, "A")
    if image.ndim != 3 or image.shape[:2] != v.shape or f.shape != v.shape or a.shape != v.shape:
        raise ValueError("RGB, V, FOV, and A must align on the complete image")
    weights = class_split_weights(f, v, a, vessel_weight=vessel_weight,
                                  background_weight=background_weight)
    out = paired_weighted_crop(image, v.astype(np.uint8), f, weights,
                               top=top, left=left, size=size, hflip=hflip,
                               vflip=vflip, quarter_turns=quarter_turns)
    _, area_crop = paired_crop_transform(image, a.astype(np.uint8), top=top, left=left,
                                         size=size, hflip=hflip, vflip=vflip,
                                         quarter_turns=quarter_turns)
    area_crop = area_crop.astype(bool)
    if not np.array_equal(out["weight"], class_split_weights(
            out["fov"], out["mask"], area_crop,
            vessel_weight=vessel_weight, background_weight=background_weight)):
        raise RuntimeError("A and its weights lost spatial alignment")
    return {**out, "area": area_crop}


def class_split_loss_with_contributions(logits, target, fov, area, *,
                                        vessel_weight=2.0, background_weight=1.0,
                                        epsilon=1e-6):
    """Per-image normalized BCE with explicit A and original unweighted FOV Dice."""
    if logits.shape != target.shape or target.shape != fov.shape or fov.shape != area.shape or logits.ndim != 4:
        raise ValueError("Matching NCHW logits, target, FOV, and explicit A required")
    pair = (float(vessel_weight), float(background_weight))
    if pair not in ALLOWED_WEIGHTS:
        raise ValueError("Only approved class-weight pairs are supported")
    z, y, f, a = logits.float(), target.float(), fov.float(), area.float()
    if not all(torch.isfinite(t).all() for t in (z, y, f, a)):
        raise ValueError("Nonfinite loss input")
    if not torch.all((y == 0) | (y == 1)) or not torch.all((f == 0) | (f == 1)) or not torch.all((a == 0) | (a == 1)):
        raise ValueError("Target, FOV, and A must be binary")
    if torch.any(a > f):
        raise ValueError("A must be a subset of FOV")
    axes = (1, 2, 3)
    mask_a = (a > 0) & (f > 0)
    nF = f.sum(axes)
    nA = mask_a.float().sum(axes)
    nA_V1 = (mask_a.float() * y).sum(axes)
    nA_V0 = (mask_a.float() * (1 - y)).sum(axes)
    D = nF + (pair[0] - 1.0) * nA_V1 + (pair[1] - 1.0) * nA_V0
    if torch.any(D <= 0):
        raise ValueError("empty_fov: no resampling is permitted")
    weights = f * (1 + a * ((pair[0] - 1.0) * y + (pair[1] - 1.0) * (1 - y)))
    # Use the same denominator expression that defines D; validate its equality to sum(F*w).
    denominator_from_weights = (f * weights).sum(axes)
    if not torch.equal(D, denominator_from_weights):
        raise RuntimeError("D differs from sum(F*w)")
    bce_map = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    masks = {
        "A_V1": mask_a.float() * y,
        "A_V0": mask_a.float() * (1 - y),
        "O_V1": f * (1 - mask_a.float()) * y,
        "O_V0": f * (1 - mask_a.float()) * (1 - y),
    }
    parts = {}
    for name, g in masks.items():
        n = g.sum(axes)
        S = (g * bce_map).sum(axes)
        coefficient = pair[0] if name == "A_V1" else pair[1] if name == "A_V0" else 1.0
        W = S * coefficient
        parts[name] = {"n": n.detach(), "S": S.detach(), "W": W.detach(),
                       "C": (W / D).detach(), "mean_S": torch.where(n > 0, S / n.clamp_min(1), torch.nan).detach()}
    bce = (bce_map * f * weights).sum(axes) / D
    probability = z.sigmoid()
    dice = 1 - (2 * (probability * y * f).sum(axes) + epsilon) / (
        (probability * f).sum(axes) + (y * f).sum(axes) + epsilon)
    if not torch.allclose(sum(parts[g]["C"] for g in GROUPS), bce, rtol=1e-5, atol=1e-7):
        raise RuntimeError("Four-group BCE contribution sum differs from BCE")
    return (bce + dice).mean(), bce, dice, {
        "nF": nF.detach(), "nA": nA.detach(), "nA_V1": nA_V1.detach(),
        "nA_V0": nA_V0.detach(), "D": D.detach(), "weights": weights.detach(),
        "area": mask_a.detach(), "groups": parts,
    }


def contribution_records(parts, filenames, seed, step, global_draws, geometries,
                         input_hashes, retries, bce, dice, *, protocol_sha256,
                         code_binding_sha256, run_binding_sha256):
    rows = []
    for i, name in enumerate(filenames):
        groups = {}
        for g in GROUPS:
            p = parts["groups"][g]
            n = int(p["n"][i].item())
            s = float(p["S"][i].item())
            groups[g] = {"n": n, "S": s, "W": float(p["W"][i].item()),
                         "C": float(p["C"][i].item()),
                         "mean_S": float(p["mean_S"][i].item()) if n else None}
        rows.append({"protocol_sha256": protocol_sha256,
                     "code_binding_sha256": code_binding_sha256,
                     "run_binding_sha256": run_binding_sha256,
                     "seed": seed, "vessel_weight": 2.0,
                     "background_weight": 1.0, "method": "w_v2_b1",
                     "step": step, "global_draw": int(global_draws[i]),
                     "batch_slot": i, "filename": name,
                     "geometry": geometries[i], "input_sha256": input_hashes[i],
                     "nF": int(parts["nF"][i].item()),
                     "nA": int(parts["nA"][i].item()),
                     "nA_V1": int(parts["nA_V1"][i].item()),
                     "nA_V0": int(parts["nA_V0"][i].item()),
                     "D": float(parts["D"][i].item()), "groups": groups,
                     "BCE_image": float(bce[i].item()),
                     "Dice_image": float(dice[i].item()),
                     "total_image": float((bce[i] + dice[i]).item()),
                     "amp_retries": int(retries)})
    return rows


def align_contribution_log(path, checkpoint_step, next_global_draw):
    """Keep exactly four rows per saved step/draw and preserve any uncommitted tail."""
    path = Path(path)
    if not path.exists():
        if checkpoint_step or next_global_draw:
            raise RuntimeError("Missing contribution log for checkpoint")
        return {"kept_rows": 0, "backup": None}
    lines = path.read_text(encoding="utf-8").splitlines()
    grouped = {}
    for line in lines:
        row = json.loads(line)
        grouped.setdefault(int(row["step"]), []).append((row, line))
    keep = []
    for step in range(1, checkpoint_step + 1):
        items = grouped.get(step, [])
        if len(items) != 4:
            raise RuntimeError(f"Step {step} must have four contribution rows")
        expected = list(range((step - 1) * 4, step * 4))
        if sorted(int(row["global_draw"]) for row, _ in items) != expected:
            raise RuntimeError(f"Step {step} draw discontinuity")
        slots = sorted(int(row["batch_slot"]) for row, _ in items)
        if slots != [0, 1, 2, 3]:
            raise RuntimeError(f"Step {step} batch_slot discontinuity")
        keep.extend(line for row, line in sorted(items, key=lambda pair: int(pair[0]["batch_slot"])))
    if next_global_draw != checkpoint_step * 4:
        raise RuntimeError("Checkpoint step/global_draw invalid")
    if keep != lines:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        backup = path.with_name(f"{path.name}.tail_after_step_{checkpoint_step:06d}_draw_{next_global_draw:06d}_{stamp}.bak")
        if backup.exists():
            raise RuntimeError("Refusing to overwrite backup")
        backup.write_text("\n".join(lines[len(keep):]) + ("\n" if lines[len(keep):] else ""), encoding="utf-8")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        tmp.replace(path)
    else:
        backup = None
    return {"kept_rows": len(keep), "backup": str(backup) if backup else None}


class ClassSplitPatches:
    """T0-identical MAPLES draws with explicit full-native A and 2/1 weights."""
    def __init__(self, config):
        self.config = config
        self.base = CorePatches(config, "T0")
        self.area_cache = {}

    def __len__(self):
        return len(self.base)

    def _area(self, index, fov, raw):
        if index in self.area_cache:
            return self.area_cache[index]
        root = Path(self.config["data_root"])
        row = self.base.native.rows[index]
        labels = []
        for key in ("label_hemorrhages", "label_brightuncertains", "label_reduncertains"):
            path = root / row[key]
            with Image.open(path) as im:
                im.load()
                labels.append(_decode_label_image(im))
        h, ub, ur = labels
        regions = build_class_split_regions(fov, h, ub | ur, image_width=raw.shape[1])
        area = regions["area"].astype(bool, copy=False)
        self.area_cache[index] = area
        return area

    def __getitem__(self, draw):
        if type(draw) is not int or draw < 0:
            raise IndexError(draw)
        index = draw % len(self.base.names)
        raw, mask, fov, _ = self.base._full_arrays(index)
        area = self._area(index, fov, raw)
        top, left, hf, vf, turns = draw_geometry(
            int(self.config["seed"]), draw, raw.shape[0], raw.shape[1], patch=512)
        item = paired_class_split_crop(raw, mask, fov, area, top=top, left=left,
                                       size=512, hflip=hf, vflip=vf,
                                       quarter_turns=turns)
        image = (item["image"].astype(np.float32) / 255.0 - self.base.mean) / self.base.std
        image *= item["fov"][..., None]
        tensors = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "mask": torch.from_numpy(np.ascontiguousarray(item["mask"][None])).float(),
            "fov": torch.from_numpy(np.ascontiguousarray(item["fov"][None])).float(),
            "weight": torch.from_numpy(np.ascontiguousarray(item["weight"][None])).float(),
            "area": torch.from_numpy(np.ascontiguousarray(item["area"][None])).float(),
            "geometry": torch.tensor([top, left, hf, vf, turns], dtype=torch.int64),
        }
        h = hashlib.sha256()
        for key in ("image", "mask", "fov", "geometry"):
            value = tensors[key].contiguous().numpy()
            h.update(f"{key}:{value.dtype}:{value.shape}".encode())
            h.update(value.tobytes())
        tensors.update(filename=self.base.names[index], draw=draw, input_sha256=h.hexdigest())
        return tensors
