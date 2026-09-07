"""Strict CPU-side FIVES reader and reproducible paired geometric transforms."""
from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .prepare import ROOT, stable_seed


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("A nonempty manifest with unique sample keys is required")
    for row in rows:
        if row["source_split"] != "train" or row["key"] != "train/" + row["filename"]:
            raise ValueError("This development loader accepts source train records only")
        for name in ("IC", "Blur", "LC", "group_size", "label_is_empty", "height", "width", "foreground_pixels"):
            row[name] = int(row[name])
        if any(row[q] not in (0, 1) for q in ("IC", "Blur", "LC")):
            raise ValueError("Invalid quality value")
        if row["label_is_empty"] != int(row["foreground_pixels"] == 0):
            raise ValueError("Inconsistent empty-label metadata")
    return rows


def decode_image(data):
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image, found {image.mode}")
        result = np.array(image)
    if result.dtype != np.uint8:
        raise ValueError("Expected uint8 image")
    return result


def decode_label(data):
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.mode not in ("RGB", "RGBA", "L"):
            raise ValueError(f"Unexpected label mode: {image.mode}")
        result = np.array(image)
    if result.dtype != np.uint8:
        raise ValueError("Expected uint8 label")
    if result.ndim == 3:
        if result.shape[-1] == 4 and not np.all(result[..., 3] == 255):
            raise ValueError("Label alpha is not uniformly opaque")
        if not np.all(result[..., :3] == result[..., :1]):
            raise ValueError("Label RGB channels disagree")
        result = result[..., 0]
    if result.ndim != 2 or np.any((result != 0) & (result != 255)):
        raise ValueError("Label must contain only 0 and 255")
    return np.ascontiguousarray(result == 255, dtype=np.uint8)


def read_pair(row, root=ROOT, verify_hashes=True):
    root = Path(root).resolve()
    if row["source_split"] != "train":
        raise ValueError("Held-out source records are forbidden in this development reader")
    filename = row["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid filename")
    if row["key"] != "train/" + filename:
        raise ValueError("Sample key and path disagree")
    arrays = []
    for role, folder, decode in (("image", "Original", decode_image), ("label", "Ground truth", decode_label)):
        path = (root / row[f"{role}_path"]).resolve()
        expected = (root / "data_checks/fives_download_2026-09-04/datasets/FIVES/train" / folder / filename).resolve()
        if path != expected or not path.is_relative_to(root):
            raise ValueError(f"Unexpected source path for {row['key']}")
        data = path.read_bytes()
        if verify_hashes and hashlib.sha256(data).hexdigest() != row[f"{role}_sha256"]:
            raise ValueError(f"Source fingerprint changed: {row['key']}/{role}")
        arrays.append(decode(data))
    image, mask = arrays
    if image.shape[:2] != mask.shape or mask.shape != (int(row["height"]), int(row["width"])):
        raise ValueError(f"Image/label dimensions disagree: {row['key']}")
    if int(mask.sum()) != int(row["foreground_pixels"]):
        raise ValueError(f"Foreground count differs from source audit: {row['key']}")
    return image, mask


def paired_crop_transform(image, mask, *, top, left, size, hflip=False, vflip=False, quarter_turns=0):
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions must match")
    if size <= 0 or min(top, left) < 0 or top + size > mask.shape[0] or left + size > mask.shape[1]:
        raise ValueError("Crop is outside the source image")
    if quarter_turns not in (0, 1, 2, 3):
        raise ValueError("Rotation must be a quarter turn")
    image = image[top:top + size, left:left + size]
    mask = mask[top:top + size, left:left + size]
    if hflip:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if vflip:
        image, mask = image[::-1], mask[::-1]
    if quarter_turns:
        image, mask = np.rot90(image, quarter_turns), np.rot90(mask, quarter_turns)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class FivesDataset(Dataset):
    def __init__(self, manifest, *, root=ROOT, view="patch", patch_size=512, repeats=1,
                 augment=False, seed=20260904, cache_size=2, verify_hashes=True, augmentation=None):
        self.rows = read_manifest(manifest)
        if view not in ("patch", "full") or repeats < 1 or patch_size < 1 or cache_size < 0:
            raise ValueError("Invalid dataset options")
        if view == "full" and (augment or repeats != 1):
            raise ValueError("Full-image view has no augmentation or repeated draws")
        if augment and any(r["trial_split"] == "smoke_val" for r in self.rows):
            raise ValueError("Do not augment the smoke validation subset")
        self.root, self.view = Path(root), view
        self.patch_size, self.repeats = patch_size, repeats
        self.augment, self.seed = augment, seed
        self.augmentation = augmentation or {
            "horizontal_flip_probability": 0.5, "vertical_flip_probability": 0.5,
            "quarter_turns": [0, 1, 2, 3], "photometric": False}
        if (self.augmentation["photometric"] or
            any(not 0 <= self.augmentation[q] <= 1 for q in ("horizontal_flip_probability", "vertical_flip_probability")) or
            not self.augmentation["quarter_turns"] or any(k not in (0, 1, 2, 3) for k in self.augmentation["quarter_turns"])):
            raise ValueError("Unsupported augmentation settings")
        self.cache_size, self.verify_hashes = cache_size, verify_hashes
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.rows) * self.repeats

    def _arrays(self, row):
        key = row["key"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arrays = read_pair(row, self.root, self.verify_hashes)
        if self.cache_size:
            self._cache[key] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()  # Do not copy decoded parent arrays to Windows workers.
        return state

    def __getitem__(self, index):
        # Epoch travels with the index, so even persistent worker copies receive it.
        epoch, draw = index if isinstance(index, tuple) else (0, index)
        if not 0 <= draw < len(self) or epoch < 0:
            raise IndexError(index)
        row = self.rows[draw // self.repeats]
        image, mask = self._arrays(row)
        if self.view == "patch":
            height, width = mask.shape
            if min(height, width) < self.patch_size:
                raise ValueError("Source is smaller than the patch; resizing is disabled")
            rng = np.random.default_rng(stable_seed(self.seed, epoch, draw, row["key"]))
            top = int(rng.integers(0, height - self.patch_size + 1))
            left = int(rng.integers(0, width - self.patch_size + 1))
            hflip, vflip, turns = (False, False, 0)
            if self.augment:
                hflip = bool(rng.random() < self.augmentation["horizontal_flip_probability"])
                vflip = bool(rng.random() < self.augmentation["vertical_flip_probability"])
                turns = int(rng.choice(self.augmentation["quarter_turns"]))
            image, mask = paired_crop_transform(image, mask, top=top, left=left, size=self.patch_size,
                                               hflip=hflip, vflip=vflip, quarter_turns=turns)
        else:
            top, left, hflip, vflip, turns = 0, 0, False, False, 0
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
            "quality": torch.tensor([row[q] for q in ("IC", "Blur", "LC")], dtype=torch.int64),
            "key": row["key"], "disease": row["disease"], "group_id": row["group_id"],
            "group_size": row["group_size"], "quality_class": row["quality_class"],
            "source_label_empty": bool(row["label_is_empty"]), "crop_label_empty": not bool(mask.any()),
            "qa_flags": row["qa_flags"], "epoch": epoch, "draw": draw,
            "geometry": torch.tensor([top, left, image.shape[0], image.shape[1], int(hflip), int(vflip), turns]),
        }


class EpochDrawSampler(Sampler):
    def __init__(self, dataset, *, epoch=0, shuffle=False):
        self.dataset, self.epoch, self.shuffle = dataset, epoch, shuffle

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        draws = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.default_rng(stable_seed(self.dataset.seed, self.epoch, "order")).shuffle(draws)
        return iter((self.epoch, int(draw)) for draw in draws)


def make_loader(dataset, *, batch_size=2, workers=0, epoch=0, shuffle=False, pin_memory=False):
    kwargs = {"multiprocessing_context": "spawn", "prefetch_factor": 2, "timeout": 60} if workers else {}
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      sampler=EpochDrawSampler(dataset, epoch=epoch, shuffle=shuffle),
                      pin_memory=pin_memory, persistent_workers=False, **kwargs)


def trial_dataset(subset, *, view=None, config_path=ROOT / "configs/data_trial_v1.json"):
    if subset not in {"smoke_train", "smoke_val", "overfit", "edge_cases", "all_train"}:
        raise ValueError("Unknown development subset")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    summary = json.loads((output / "selection_summary.json").read_text(encoding="utf-8"))
    config_key = config_path.relative_to(ROOT).as_posix()
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != summary["source_sha256"][config_key]:
        raise ValueError("Config changed: rebuild and re-verify the trial manifests")
    manifest = output / "manifests" / f"{subset}.csv"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != summary["manifest_sha256"][manifest.name]:
        raise ValueError("Trial manifest differs from its recorded fingerprint")
    view = view or ("patch" if subset in {"smoke_train", "overfit"} else "full")
    return FivesDataset(manifest, view=view, patch_size=config["patch_size"],
                        repeats=config["crops_per_image"] if view == "patch" else 1,
                        augment=subset == "smoke_train" and view == "patch", seed=config["seed"],
                        cache_size=config["decoded_cache_images_per_worker"], augmentation=config["augmentation"])
