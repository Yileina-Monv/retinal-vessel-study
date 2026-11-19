"""Shared, protocol-bound MAPLES T0/TW data and loss primitives."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from retinal_data.fives import paired_crop_transform
from retinal_data.maples_fov import candidate_fov
from retinal_data.maples_native import NativeMaplesDataset, decode_native_vessel


def build_training_regions(fov, hemorrhages, uncertain, *, image_width):
    """Partition Omega on the full native image, then create protocol weights."""
    fov = _binary_array(fov, "FOV")
    hemorrhages = _binary_array(hemorrhages, "H")
    uncertain = _binary_array(uncertain, "U")
    if fov.ndim != 2 or hemorrhages.shape != fov.shape or uncertain.shape != fov.shape:
        raise ValueError("FOV, H, and U must be aligned native 2D masks")
    if type(image_width) is not int or image_width != fov.shape[1]:
        raise ValueError("image_width must equal the complete native RGB width")
    if not fov.any():
        raise ValueError("empty_fov: no resampling is permitted")
    radius = max(1, int(np.ceil(image_width / 100)))
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    dilated = binary_dilation(hemorrhages, structure=disk)
    omega = fov & ~uncertain
    rin = hemorrhages & omega
    rnear = dilated & ~hemorrhages & omega
    rfar = omega & ~dilated
    weights = np.zeros(fov.shape, dtype=np.float32)
    weights[fov] = 1.0
    weights[rin | rnear] = 2.0
    membership = rin.astype(np.uint8) + rnear.astype(np.uint8) + rfar.astype(np.uint8)
    if not np.array_equal(membership, omega.astype(np.uint8)):
        raise RuntimeError("Rin/Rnear/Rfar do not partition Omega")
    if np.any(weights[uncertain & fov] != 1) or np.any(weights[~fov] != 0):
        raise RuntimeError("Training weight invariant failed")
    return {"omega": omega, "rin": rin, "rnear": rnear, "rfar": rfar,
            "weights": weights, "dilation_radius": radius}


def _binary_array(value, name):
    result = np.asarray(value)
    if result.ndim != 2 or not np.isin(result, (0, 1, False, True)).all():
        raise ValueError(name + " must be a binary 2D array")
    return result.astype(bool, copy=False)


def paired_weighted_crop(image, mask, fov, weights, *, top, left, size,
                          hflip=False, vflip=False, quarter_turns=0):
    """Apply one frozen crop/flip/rotation to RGB, V, FOV, and prebuilt weights."""
    image = np.asarray(image)
    mask = _binary_array(mask, "V")
    fov = _binary_array(fov, "FOV")
    weights = np.asarray(weights, dtype=np.float32)
    if image.ndim != 3 or image.shape[:2] != mask.shape or fov.shape != mask.shape or weights.shape != mask.shape:
        raise ValueError("RGB, V, FOV and weights must align on the full image")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    x, y = paired_crop_transform(image, mask, top=top, left=left, size=size,
                                 hflip=hflip, vflip=vflip, quarter_turns=quarter_turns)
    _, f = paired_crop_transform(image, fov.astype(np.uint8), top=top, left=left,
                                 size=size, hflip=hflip, vflip=vflip,
                                 quarter_turns=quarter_turns)
    _, w = paired_crop_transform(image, weights, top=top, left=left, size=size,
                                 hflip=hflip, vflip=vflip,
                                 quarter_turns=quarter_turns)
    if not f.any():
        raise ValueError("empty_fov: no resampling is permitted")
    return {"image": x, "mask": y, "fov": f.astype(bool), "weight": w.astype(np.float32)}


def draw_geometry(seed, draw, height, width, *, patch=512):
    if type(seed) is not int or type(draw) is not int or draw < 0 or min(height, width) < patch:
        raise ValueError("Invalid seed, draw, or source image")
    rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
    return (int(rng.integers(height - patch + 1)),
            int(rng.integers(width - patch + 1)),
            bool(rng.random() < 0.5), bool(rng.random() < 0.5),
            int(rng.integers(4)))


def weighted_masked_loss(logits, target, fov, weights, epsilon=1e-6):
    """Per-image normalized weighted BCE plus the original unweighted masked Dice."""
    if logits.shape != target.shape or target.shape != fov.shape or fov.shape != weights.shape or logits.ndim != 4:
        raise ValueError("Matching NCHW logits, target, FOV, and weight tensors required")
    logits, target, fov, weights = logits.float(), target.float(), fov.float(), weights.float()
    if not all(torch.isfinite(t).all() for t in (logits, target, fov, weights)):
        raise ValueError("Loss inputs must be finite")
    if not torch.all((target == 0) | (target == 1)) or not torch.all((fov == 0) | (fov == 1)):
        raise ValueError("Target and FOV must be binary")
    if torch.any(weights < 0) or torch.any(weights[fov == 0] != 0):
        raise ValueError("Weights must be nonnegative and zero outside FOV")
    axes = (1, 2, 3)
    valid_weights = fov * weights
    denominator = valid_weights.sum(axes)
    if torch.any(denominator <= 0):
        raise ValueError("empty_fov: no resampling is permitted")
    bce_map = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * valid_weights).sum(axes) / denominator
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target * fov).sum(axes) + epsilon) / (
        (probability * fov).sum(axes) + (target * fov).sum(axes) + epsilon)
    return (bce + dice).mean()


class CorePatches(Dataset):
    """Manifest-order four-round training draws with geometry keyed by global draw."""
    def __init__(self, config, method):
        if method not in ("T0", "TW"):
            raise ValueError("method must be T0 or TW")
        self.config, self.method = config, method
        self.native = NativeMaplesDataset(config["data_root"], split="train")
        if self.native.split_counts != {"train": 110, "dev": 28, "official_test": 60}:
            raise ValueError("Frozen MAPLES split counts differ")
        self.names = [row["filename"] for row in self.native.rows]
        if config.get("train_sample_filenames") != self.names:
            raise ValueError("Configured manifest order differs from source manifest")
        self.mean = np.asarray(config["mean"], dtype=np.float32)[None, None, :]
        self.std = np.asarray(config["std"], dtype=np.float32)[None, None, :]
        if self.mean.shape != (1, 1, 3) or self.std.shape != (1, 1, 3) or np.any(self.std <= 0):
            raise ValueError("Invalid frozen normalization")
        self.cache = OrderedDict()
        self.input_hashes = {}

    def __len__(self):
        return len(self.names) * 4

    def _full_arrays(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        item = self.native[index]
        row = self.native.rows[index]
        image = item["image"].permute(1, 2, 0).numpy()
        raw = np.rint(image * 255).astype(np.uint8)
        fov = candidate_fov(raw)
        root = Path(self.config["data_root"])
        h_path = root / row["label_hemorrhages"]
        ub_path = root / row["label_brightuncertains"]
        ur_path = root / row["label_reduncertains"]
        labels = []
        label_hashes = {}
        for role, path in (("H", h_path), ("U_bright", ub_path), ("U_red", ur_path)):
            data = path.read_bytes()
            label_hashes[role] = hashlib.sha256(data).hexdigest()
            with Image.open(path) as im:
                im.load()
                labels.append(_decode_label_image(im))
        hemorrhage, u_bright, u_red = labels
        if any(a.shape != fov.shape for a in labels) or fov.shape != image.shape[:2]:
            raise ValueError("Native RGB/FOV/H/U shapes differ")
        h = hashlib.sha256()
        for value in (raw, fov.astype(np.uint8), hemorrhage, u_bright, u_red):
            h.update(np.asarray(value).tobytes())
        self.input_hashes[row["filename"]] = {
            "raw_rgb_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "fov_sha256": hashlib.sha256(fov.astype(np.uint8).tobytes()).hexdigest(),
            "vessel_sha256": row["label_vessels_sha256"],
            "lesion_label_sha256": label_hashes,
            "region_arrays_sha256": h.hexdigest(),
        }
        if self.method == "TW":
            regions = build_training_regions(fov, hemorrhage, u_bright | u_red,
                                             image_width=raw.shape[1])
            weights = regions["weights"]
        else:
            weights = fov.astype(np.float32)
        result = (raw, item["vessel_mask"][0].numpy().astype(np.uint8), fov, weights)
        self.cache[index] = result
        if len(self.cache) > 110:
            self.cache.popitem(last=False)
        return result

    def __getitem__(self, draw):
        if type(draw) is not int or draw < 0:
            raise IndexError(draw)
        index = draw % len(self.names)
        raw, mask, fov, weights = self._full_arrays(index)
        top, left, hf, vf, turns = draw_geometry(
            int(self.config["seed"]), draw, raw.shape[0], raw.shape[1], patch=512)
        item = paired_weighted_crop(raw, mask, fov, weights, top=top, left=left, size=512,
                                    hflip=hf, vflip=vf, quarter_turns=turns)
        image = (item["image"].astype(np.float32) / 255.0 - self.mean) / self.std
        image *= item["fov"][..., None]
        tensors = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "mask": torch.from_numpy(np.ascontiguousarray(item["mask"][None])).float(),
            "fov": torch.from_numpy(np.ascontiguousarray(item["fov"][None])).float(),
            "weight": torch.from_numpy(np.ascontiguousarray(item["weight"][None])).float(),
            "geometry": torch.tensor([top, left, hf, vf, turns], dtype=torch.int64),
        }
        h = hashlib.sha256()
        for key in ("image", "mask", "fov", "geometry"):
            value = tensors[key].contiguous().numpy()
            h.update(f"{key}:{value.dtype}:{value.shape}".encode())
            h.update(value.tobytes())
        tensors.update(filename=self.names[index], draw=draw, input_sha256=h.hexdigest())
        return tensors


def _decode_label_image(image):
    a = np.asarray(image)
    if a.ndim == 3:
        if a.shape[-1] == 4 and not np.all(a[..., 3] == 255):
            raise ValueError("Label alpha must be fully opaque")
        if not np.all(a[..., :3] == a[..., :1]):
            raise ValueError("Label RGB channels disagree")
        a = a[..., 0]
    if a.ndim != 2 or not (np.isin(a, (0, 1)).all() or np.isin(a, (0, 255)).all()):
        raise ValueError("Lesion masks must be binary")
    return np.ascontiguousarray(a != 0, dtype=np.uint8)
