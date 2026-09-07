"""Accept the data loader using train-only real data and saved visual checks."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import time
import traceback

import numpy as np
import torch

from .fives import FivesDataset, make_loader, read_manifest, read_pair, trial_dataset
from .prepare import ROOT, sha256

OUTPUT = ROOT / "data_preparation"
ARTIFACTS = OUTPUT / "artifacts"


def check_all_train():
    rows = read_manifest(OUTPUT / "manifests/all_train.csv")
    empty, rgba = [], []
    for index, row in enumerate(rows, 1):
        image, mask = read_pair(row)
        if not mask.any():
            empty.append(row["key"])
        if row["label_mode"] == "RGBA":
            rgba.append(row["key"])
        if index % 100 == 0:
            print(f"Train-only source reader: {index}/{len(rows)} pairs passed", flush=True)
    assert len(empty) == 2 and len(rgba) == 8
    return {"pairs_read": len(rows), "source_files_sha256_checked": 2 * len(rows),
            "empty_labels": empty, "opaque_rgba_labels": rgba,
            "image_label_size_and_foreground_count_match_audit": True}


def signatures(dataset, workers):
    result = {}
    empty_crops = 0
    for batch in make_loader(dataset, batch_size=2, workers=workers, epoch=0, shuffle=True):
        assert batch["image"].dtype == torch.float32 and batch["mask"].dtype == torch.float32
        assert tuple(batch["image"].shape[1:]) == (3, 512, 512)
        assert tuple(batch["mask"].shape[1:]) == (1, 512, 512)
        assert torch.isfinite(batch["image"]).all() and 0 <= batch["image"].min() <= batch["image"].max() <= 1
        assert torch.all((batch["mask"] == 0) | (batch["mask"] == 1))
        for index, key in enumerate(batch["key"]):
            draw = int(batch["draw"][index])
            expected = dataset.rows[draw // dataset.repeats]
            assert batch["quality"][index].tolist() == [expected[q] for q in ("IC", "Blur", "LC")]
            assert batch["group_id"][index] == expected["group_id"]
            payload = batch["image"][index].numpy().tobytes() + batch["mask"][index].numpy().tobytes()
            signature = hashlib.sha256(payload).hexdigest()
            result[f"{key}#{draw}"] = {"sha256": signature, "geometry": batch["geometry"][index].tolist()}
            empty_crops += bool(batch["crop_label_empty"][index])
    assert len(result) == len(dataset)
    return result, empty_crops


def check_loader():
    single = trial_dataset("smoke_train")
    one, empty_crops = signatures(single, 0)
    two, other_empty_crops = signatures(trial_dataset("smoke_train"), 2)
    assert one == two and empty_crops == other_empty_crops
    assert not torch.equal(single[(0, 0)]["geometry"], single[(1, 0)]["geometry"])
    first = single[(1, 3)]
    replay = trial_dataset("smoke_train")[(1, 3)]
    assert torch.equal(first["image"], replay["image"]) and torch.equal(first["mask"], replay["mask"])
    validation = trial_dataset("smoke_val")
    for index in range(len(validation)):
        item = validation[index]
        assert item["image"].shape == (3, 2048, 2048) and item["mask"].shape == (1, 2048, 2048)
        assert item["geometry"].tolist() == [0, 0, 2048, 2048, 0, 0, 0]
    edge = FivesDataset(OUTPUT / "manifests/edge_cases.csv", augment=False)
    edge_checks = []
    for index, row in enumerate(edge.rows):
        item = edge[index]
        if row["label_is_empty"]:
            assert item["source_label_empty"] and item["crop_label_empty"] and not item["mask"].any()
        edge_checks.append({"key": row["key"], "flags": row["qa_flags"], "passed": True})
    duplicates = defaultdict(list)
    for row in edge.rows:
        if row["group_size"] > 1:
            duplicates[row["group_id"]].append(row)
    differences = []
    for group in duplicates.values():
        (image_a, mask_a), (image_b, mask_b) = [read_pair(row) for row in group]
        assert np.array_equal(image_a, image_b)
        differing = int(np.count_nonzero(mask_a != mask_b))
        assert differing > 0
        differences.append({"keys": [r["key"] for r in group], "different_label_pixels": differing})
    (OUTPUT / "draw_signatures.json").write_text(json.dumps(one, indent=2), encoding="utf-8")
    return {"patch_draws_per_pass": len(one), "workers_compared": [0, 2],
            "exact_images_masks_geometry_match_across_workers": True, "empty_crops_retained": empty_crops,
            "epoch_changes_draws": True, "epoch_draw_replay_matches": True,
            "full_validation_images_checked": len(validation), "edge_checks": edge_checks,
            "duplicate_annotations_preserved": differences}


def overlay(image, mask):
    result = image.astype(np.float32) / 255
    result[mask.astype(bool)] = 0.55 * result[mask.astype(bool)] + 0.45 * np.array([0., 1., 1.])
    return np.clip(result, 0, 1)


def make_galleries():
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dataset = trial_dataset("smoke_train")
    rows = read_manifest(OUTPUT / "manifests/overfit.csv")
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), dpi=140)
    for index, row in enumerate(rows):
        draw = next(i for i, candidate in enumerate(dataset.rows) if candidate["key"] == row["key"]) * dataset.repeats
        item = dataset[draw]
        image, _ = read_pair(row)
        patch = (item["image"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        mask = item["mask"][0].numpy().astype(np.uint8)
        top, left, height, width, hflip, vflip, turns = item["geometry"].tolist()
        axes[index, 0].imshow(image[::4, ::4])
        axes[index, 0].add_patch(Rectangle((left/4, top/4), width/4, height/4, fill=False, color="cyan", linewidth=1.2))
        axes[index, 0].set_title(f"{row['key']} | crop location", fontsize=9)
        axes[index, 1].imshow(patch)
        axes[index, 1].set_title(f"RGB | H={hflip}, V={vflip}, R={90*turns}", fontsize=9)
        axes[index, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[index, 2].set_title("Paired binary annotation", fontsize=9)
        axes[index, 3].imshow(overlay(patch, mask))
        axes[index, 3].set_title("Annotation overlay (cyan)", fontsize=9)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle("FIVES train only: original-scale crops and synchronized geometry", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(ARTIFACTS / "paired_crops.png")
    plt.close(fig)

    edge_rows = read_manifest(OUTPUT / "manifests/edge_cases.csv")
    selected = [r for r in edge_rows if r["label_is_empty"]]
    selected.append(next(r for r in edge_rows if r["label_mode"] == "RGBA"))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=120)
    for index, row in enumerate(selected):
        image, mask = read_pair(row)
        axes[index, 0].imshow(image[::4, ::4])
        axes[index, 0].set_title(row["key"])
        axes[index, 1].imshow(mask[::4, ::4], cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_title("Source annotation")
        axes[index, 2].imshow(overlay(image[::4, ::4], mask[::4, ::4]))
        axes[index, 2].set_title("Empty label" if row["label_is_empty"] else "Opaque RGBA decoded")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "edge_cases.png")
    plt.close(fig)
    return ["data_preparation/artifacts/paired_crops.png", "data_preparation/artifacts/edge_cases.png"]


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "steps_1_and_2_only", "model_training_executed": False,
              "source_test_image_files_opened": 0, "external_image_files_opened": 0}
    try:
        report["full_train_reader"] = check_all_train()
        print("Checking patch tensors, worker invariance and edge cases...", flush=True)
        report["loader_checks"] = check_loader()
        report["visual_files"] = make_galleries()
        report["visual_review"] = "pending_manual_inspection"
        report["code_and_config_sha256"] = {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in [ROOT / "configs/data_trial_v1.json", *sorted((ROOT / "retinal_data").glob("*.py"))]}
        report["status"] = "passed_automated_checks"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (OUTPUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Data verification: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
