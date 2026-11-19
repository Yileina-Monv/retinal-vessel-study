"""Native MAPLES inspection interface. No FOV generation or formal normalization."""
from __future__ import annotations

import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from retinal_data.fives import decode_image


def _require(condition, code, detail):
    if not condition:
        raise ValueError(f"{code}: {detail}")


def _sample_path(root, value, expected):
    _require(isinstance(value, str) and value == expected and "\\" not in value
             and ".." not in Path(value).parts and not Path(value).is_absolute(),
             "sample_path", str(value))
    path = (root / value).resolve()
    _require(path.is_relative_to(root) and path.is_file(), "sample_path", value)
    return path


def _verified_bytes(path, expected):
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(expected))), "file_hash", str(path))
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == expected, "file_hash", str(path))
    return data


def decode_native_vessel(data):
    """Support native MAPLES mode 1 and documented L binary encodings."""
    with Image.open(BytesIO(data)) as label:
        label.load()
        _require(label.mode in ("1", "L"), "binary_mask", f"Unexpected mode {label.mode}")
        array = np.array(label)
    values = set(np.unique(array).tolist())
    _require(array.ndim == 2 and (values <= {0, 1} or values <= {0, 255}),
             "binary_mask", "Expected a single-channel binary mask")
    return np.ascontiguousarray(array != 0, dtype=np.uint8)


class NativeMaplesDataset(Dataset):
    """Read Task 1's unchanged manifest at native resolution for inspection.

    RGB / 255 is basic range conversion, not an approved normalization.
    FOV is None until the separately reviewed FOV interface is available.
    Official test requires explicit inspection opt-in and is never training-ready.
    """

    def __init__(self, root, *, split="train", allow_official_test=False):
        self.root = Path(root).resolve()
        _require(split in ("train", "dev", "official_test"), "dataset_split", str(split))
        _require(split != "official_test" or allow_official_test,
                 "held_out_test", "Official test is available only by explicit inspection opt-in")
        manifest = self.root / "data_manifest.csv"
        manifest_bytes = manifest.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"key", "filename", "source_split", "trial_split", "image_path",
                        "label_vessels", "image_sha256", "label_vessels_sha256", "image_bytes",
                        "height", "width", "vessel_pixels", "label_status"}
            _require(reader.fieldnames is not None and required <= set(reader.fieldnames),
                     "manifest_columns", "Missing native handoff columns")
            rows = list(reader)
        _require(bool(rows) and all(None not in row for row in rows),
                 "manifest_rows", "Nonempty well-formed CSV required")
        seen_ids, seen_keys, image_splits = set(), set(), {}
        counts = dict(train=0, dev=0, official_test=0)
        for row in rows:
            sid, key, trial, source = (row[k] for k in ("filename", "key", "trial_split", "source_split"))
            _require(bool(re.fullmatch(r"[A-Za-z0-9_]+", sid)) and sid.endswith("_PP")
                     and sid not in seen_ids and key not in seen_keys,
                     "sample_identity", "Unique case-preserved public image IDs required")
            seen_ids.add(sid)
            seen_keys.add(key)
            _require(source in ("train", "test") and key == f"MAPLES-DR/{source}/{sid}",
                     "sample_identity", key)
            _require(trial in counts, "dataset_split", key)
            _require(source != "test" or trial == "official_test", "held_out_test", key)
            _require(trial != "official_test" or source == "test", "source_partition", key)
            _require(source != "train" or trial in ("train", "dev"), "source_partition", key)
            counts[trial] += 1
            _require(row["label_status"] in ("ok", "missing_anatomy"),
                     "vessel_status", key)
            for field in ("height", "width", "image_bytes", "vessel_pixels"):
                try:
                    row[field] = int(row[field])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"manifest_number: {key}/{field}") from error
            _require(row["height"] > 0 and row["width"] > 0 and row["image_bytes"] > 0
                     and row["vessel_pixels"] >= 0, "manifest_number", key)
            _sample_path(self.root, row["image_path"], f"images/{sid}.tif")
            _sample_path(self.root, row["label_vessels"], f"labels/Vessels/{sid}.png")
            for field in ("image_sha256", "label_vessels_sha256"):
                _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(row[field]))), "file_hash", key)
            fingerprint = row["image_sha256"]
            _require(fingerprint not in image_splits or image_splits[fingerprint] == trial,
                     "split_crossing", key)
            image_splits[fingerprint] = trial
        record_path = self.root / "split_record.json"
        self.split_record_sha256 = hashlib.sha256(record_path.read_bytes()).hexdigest()
        record = json.loads(record_path.read_text(encoding="utf-8"))
        _require(record.get("counts") == counts, "split_record", "Counts differ from handoff")
        dev = record.get("dev_files")
        _require(isinstance(dev, list) and len(dev) == len(set(dev))
                 and set(dev) == {r["filename"] for r in rows if r["trial_split"] == "dev"},
                 "split_record", "Dev membership differs from handoff")
        self.rows = [r for r in rows if r["trial_split"] == split]
        self.split, self.split_counts = split, counts
        _require(bool(self.rows), "empty_split", split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        row = self.rows[index]
        sid = row["filename"]
        image_path = _sample_path(self.root, row["image_path"], f"images/{sid}.tif")
        mask_path = _sample_path(self.root, row["label_vessels"], f"labels/Vessels/{sid}.png")
        image_data = _verified_bytes(image_path, row["image_sha256"])
        _require(len(image_data) == row["image_bytes"], "file_size", row["key"])
        image = decode_image(image_data)  # Reuse FIVES' strict RGB/uint8 decoder.
        mask = decode_native_vessel(_verified_bytes(mask_path, row["label_vessels_sha256"]))
        expected = (row["height"], row["width"])
        _require(image.shape == (*expected, 3) and mask.shape == expected,
                 "sample_shape", row["key"])
        _require(int(mask.sum()) == row["vessel_pixels"], "vessel_count", row["key"])

        # Preserve every source pixel and the same tensor conversion as FIVES.
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255)
        vessel_tensor = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return {
            "image": image_tensor,
            "vessel_mask": vessel_tensor,
            "fov": None,
            "metadata": {
                "sample_id": row["key"], "filename": sid, "split": self.split,
                "source_split": row["source_split"], "image_path": str(image_path),
                "mask_path": str(mask_path), "original_size": expected,
                "manifest_sha256": self.manifest_sha256,
                "split_record_sha256": self.split_record_sha256,
                "fov_status": "not_ready", "normalization_status": "pending",
                "image_conversion": "RGB uint8 -> float32 / 255 only",
                "training_ready": False, "patient_independence": "unknown",
                "reported_group_id": row.get("group_id", "unknown"),
                "label_status": row["label_status"],
            },
        }


def collate_native(samples):
    """Batch native inspection samples without fabricating FOV or resizing."""
    _require(bool(samples), "batch_shape", "Empty batch")
    first = samples[0]
    _require(all(s["image"].shape == first["image"].shape
                 and s["vessel_mask"].shape == first["vessel_mask"].shape for s in samples),
             "batch_shape", "Use batch size 1 or same-sized crops; native resolutions differ")
    _require(all(s["fov"] is None for s in samples), "fov_contract",
             "Formal FOV batching needs its approved contract")
    return {"image": torch.stack([s["image"] for s in samples]),
            "vessel_mask": torch.stack([s["vessel_mask"] for s in samples]),
            "fov": None, "metadata": [s["metadata"] for s in samples]}
