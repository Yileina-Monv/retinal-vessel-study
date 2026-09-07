"""Check archive pairing and three fixed samples; no model or semantic audit."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import json

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

def load_mask(archive, member):
    with Image.open(BytesIO(archive.read(member))) as image:
        image.load()
        array = np.array(image)
        info = {"size_wh": image.size, "stored_mode": image.mode}
    if array.ndim == 3:
        if array.shape[2] != 3 or not np.all(array == array[:, :, :1]):
            raise ValueError("Mask channels are not identical RGB planes")
        array = array[:, :, 0]
        info["normalization"] = "Verified three equal channels, then selected channel 0"
    elif array.ndim == 2:
        info["normalization"] = "Already single channel"
    else:
        raise ValueError("Unexpected mask dimensions")
    info["unique_values"] = np.unique(array).tolist()
    return array, info

with ZipFile(ROOT / "archives/healthy.zip") as images, ZipFile(ROOT / "archives/healthy_manualsegm.zip") as labels, ZipFile(ROOT / "archives/healthy_fovmask.zip") as fovs:
    image_map = {Path(n).stem: n for n in images.namelist() if not n.endswith("/")}
    label_map = {Path(n).stem: n for n in labels.namelist() if not n.endswith("/")}
    fov_map = {Path(n).stem.removesuffix("_mask"): n for n in fovs.namelist() if not n.endswith("/")}
    if set(image_map) != set(label_map) or set(image_map) != set(fov_map):
        raise ValueError("Image, label and FOV identifiers do not match")
    samples = []
    for key in ["01_h", "08_h", "15_h"]:
        with Image.open(BytesIO(images.read(image_map[key]))) as image:
            image.load()
            size, mode = image.size, image.mode
        y, label_info = load_mask(labels, label_map[key])
        f, fov_info = load_mask(fovs, fov_map[key])
        if size != tuple(label_info["size_wh"]) or size != tuple(fov_info["size_wh"]):
            raise ValueError("Image and masks have different dimensions")
        if not set(label_info["unique_values"]).issubset({0, 255}) or not set(fov_info["unique_values"]).issubset({0, 255}):
            raise ValueError("Unexpected values in masks")
        samples.append({"id": key, "image_size_wh": size, "image_mode": mode, "label": label_info, "fov": fov_info, "label_foreground_pixels": int(np.count_nonzero(y)), "fov_foreground_pixels": int(np.count_nonzero(f)), "label_foreground_outside_fov": int(np.count_nonzero((y > 0) & (f == 0))), "dimensions_match": True})

report = {
    "status": "PASS_DOWNLOAD_AND_FORMAT_SMOKE",
    "scope": "HRF healthy subset only; all 15 paired by filename; three fixed samples fully decoded; no patient, semantic label, clinical or model evaluation",
    "paired_count": len(image_map),
    "paired_ids": sorted(image_map),
    "sample_selection": "First, middle and last by fixed ID: 01_h, 08_h, 15_h",
    "initial_check_correction": "The first draft assumed a single-channel FOV and stopped before saving results; actual FOV is RGB with identical planes. Channel equality is now checked explicitly.",
    "samples": samples,
}
(ROOT / "hrf_format_evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
