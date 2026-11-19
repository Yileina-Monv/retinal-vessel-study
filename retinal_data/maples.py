"""Train-only MAPLES crop adapter for the bounded engineering interface check."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from retinal_data.fives import paired_crop_transform
from retinal_data.maples_fov import candidate_fov
from retinal_data.maples_native import NativeMaplesDataset


class MaplesTrainPatches(Dataset):
    def __init__(self, config):
        if config.get("scope") != "interface_validation_110_train_only":
            raise ValueError("This adapter is restricted to the 110-train interface check")
        self.native = NativeMaplesDataset(config["data_root"], split="train")
        if self.native.split_counts != {"train": 110, "dev": 28, "official_test": 60}:
            raise ValueError("Unexpected split counts")
        names = config["train_sample_filenames"]
        actual = {row["filename"] for row in self.native.rows}
        if len(names) != 110 or len(set(names)) != 110 or set(names) != actual:
            raise ValueError("Configured train list differs from Task 1 handoff")
        if config.get("patch") != 512 or config.get("microbatch") != 1 or config.get("accumulation") != 1:
            raise ValueError("Only the approved 512-pixel, batch-1 profile is supported")
        self.positions = {row["filename"]: i for i, row in enumerate(self.native.rows)}
        self.names = names
        self.seed = int(config["seed"])
        self.mean = torch.as_tensor(config["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.as_tensor(config["std"], dtype=torch.float32)[:, None, None]
        if self.mean.shape != (3, 1, 1) or self.std.shape != (3, 1, 1) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid normalization statistics")
        source = Path(config["normalization_source"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != config["normalization_source_sha256"]:
            raise ValueError("Normalization source hash differs")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, draw):
        if type(draw) is not int or draw < 0:
            raise IndexError(draw)
        sid = self.names[draw % len(self.names)]
        item = self.native[self.positions[sid]]
        image, target = item["image"], item["vessel_mask"]
        h, w = image.shape[-2:]
        size = 512
        if h < size or w < size:
            raise ValueError("Source image smaller than patch")
        rng = np.random.default_rng(self.seed + draw)
        top = int(rng.integers(h - size + 1))
        left = int(rng.integers(w - size + 1))
        raw = np.rint(image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        fov = candidate_fov(raw)
        x, y = paired_crop_transform(image.permute(1, 2, 0).numpy(), target[0].numpy(), top=top, left=left, size=size)
        _, field = paired_crop_transform(image.permute(1, 2, 0).numpy(), fov.astype(np.uint8), top=top, left=left, size=size)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        y = torch.from_numpy(np.ascontiguousarray(y[None])).float()
        field = torch.from_numpy(np.ascontiguousarray(field[None])).float()
        if field.sum() <= 0:
            raise ValueError("Empty candidate FOV crop")
        if not torch.equal(y, target[:, top:top + size, left:left + size]):
            raise ValueError("Vessel crop lost spatial alignment")
        x = ((x - self.mean) / self.std) * field
        if not torch.isfinite(x).all() or not set(y.unique().tolist()) <= {0.0, 1.0} or not set(field.unique().tolist()) <= {0.0, 1.0}:
            raise ValueError("Invalid crop values")
        return {"image": x, "mask": y, "fov": field,
                "key": item["metadata"]["sample_id"], "filename": sid,
                "geometry": torch.tensor([top, left, size, size], dtype=torch.int64)}
