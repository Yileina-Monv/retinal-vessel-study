"""Build a deterministic, provisional trial selection from the audited train split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["key", "source_split", "trial_split", "filename", "disease", "IC", "Blur", "LC",
          "quality_class", "group_id", "group_size", "label_is_empty", "qa_flags",
          "image_path", "label_path", "image_sha256", "label_sha256", "pixel_sha256",
          "image_mode", "label_mode", "height", "width", "foreground_pixels"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def build_selection(config_path: Path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["allowed_source_split"] != "train":
        raise ValueError("This trial is restricted to the source train split")
    source = ROOT / config["source_audit_dir"]
    source_csv = source / "fives_manifest.csv"
    quality_json = source / "quality_evidence.json"
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    audit_rows = [r for r in all_rows if r["split"] == "train"]
    if len(audit_rows) != 600 or len({r["key"] for r in audit_rows}) != 600:
        raise ValueError("Expected exactly 600 unique source train records")
    quality_records = json.loads(quality_json.read_text(encoding="utf-8"))["records"]
    quality = {r["key"]: r for r in quality_records if r["split"] == "train"}
    if len(quality) != 600:
        raise ValueError("Quality metadata is incomplete")
    group_sizes = Counter(r["pixel_sha256"] for r in audit_rows)
    rows = []
    for original in sorted(audit_rows, key=lambda r: r["key"]):
        if original["key"] != "train/" + original["filename"]:
            raise ValueError("Sample key is inconsistent")
        values = [int(original[q]) for q in ("IC", "Blur", "LC")]
        if any(v not in (0, 1) for v in values):
            raise ValueError("Quality scores must be 0/1")
        match = quality[original["key"]]
        if match["disease"] != original["disease"] or values != [match[q] for q in ("IC", "Blur", "LC")]:
            raise ValueError("Audit and quality table disagree")
        empty = original["label_is_empty"] == "True"
        flags = []
        if group_sizes[original["pixel_sha256"]] > 1:
            flags.append("duplicate_image_distinct_labels")
        if empty:
            flags.append("empty_label")
        if original["label_mode"] == "RGBA":
            flags.append("opaque_rgba_label")
        row = {name: original[name] for name in ("key", "filename", "disease", "image_sha256",
               "label_sha256", "pixel_sha256", "image_mode", "label_mode")}
        row.update(source_split="train", trial_split="deferred", IC=values[0], Blur=values[1], LC=values[2],
                   quality_class="good" if all(values) else "poor",
                   group_id="pixel_" + original["pixel_sha256"],
                   group_size=group_sizes[original["pixel_sha256"]], label_is_empty=int(empty),
                   qa_flags=";".join(flags), height=2048, width=2048,
                   foreground_pixels=int(original["label_foreground_pixels_gt_0"]))
        for role, folder in (("image", "Original"), ("label", "Ground truth")):
            expected = Path("datasets") / "FIVES" / "train" / folder / row["filename"]
            audited = Path(original[f"{role}_path"].replace("\\", "/"))
            if audited != expected:
                raise ValueError(f"Unexpected audited path for {row['key']}")
            row[f"{role}_path"] = (source.relative_to(ROOT) / expected).as_posix()
        if row["group_size"] > 1 or empty:
            row["trial_split"] = "edge_only"
        rows.append(row)

    # Rank by metadata and a fixed seed; no image pixels or model scores guide selection.
    for disease in config["diseases"]:
        for quality_class in ("good", "poor"):
            candidates = sorted(
                (r for r in rows if r["disease"] == disease and r["quality_class"] == quality_class
                 and r["trial_split"] == "deferred"),
                key=lambda r: (stable_seed(config["seed"], disease, quality_class, r["key"]), r["key"]))
            offset = 0
            for split in ("smoke_val", "smoke_train"):
                count = config["per_disease"][split][quality_class]
                chosen = candidates[offset:offset + count]
                if len(chosen) != count:
                    raise ValueError(f"Insufficient {disease}/{quality_class} candidates")
                for row in chosen:
                    row["trial_split"] = split
                offset += count
    by_group = defaultdict(set)
    for row in rows:
        by_group[row["group_id"]].add(row["trial_split"])
    if any(len(splits) != 1 for splits in by_group.values()):
        raise ValueError("An image group spans trial partitions")
    subsets = {"all_train": rows}
    for split in ("smoke_train", "smoke_val"):
        subsets[split] = [r for r in rows if r["trial_split"] == split]
    subsets["edge_cases"] = [r for r in rows if r["qa_flags"]]
    subsets["overfit"] = [min((r for r in subsets["smoke_train"] if r["disease"] == disease
                                    and r["quality_class"] == "good"),
                                key=lambda r: (stable_seed(config["seed"], "overfit", r["key"]), r["key"]))
                          for disease in config["diseases"]]
    sources = {str(path.relative_to(ROOT).as_posix()): sha256(path)
               for path in (config_path, source_csv, quality_json, source / "archives" / "quality_primary.xlsx")}
    summary = {
        "protocol_id": config["protocol_id"], "seed": config["seed"], "source_sha256": sources,
        "selection_method": "SHA256 rank within disease and quality; validation selected first",
        "source_train_rows": len(rows), "source_train_groups": len(by_group),
        "source_test_image_files_opened": 0, "external_image_files_opened": 0,
        "source_test_rows_reserved": sum(r["split"] == "test" for r in all_rows),
        "partition_counts": dict(Counter(r["trial_split"] for r in rows)),
        "subset_counts": {key: len(value) for key, value in subsets.items()},
        "qa_flags": dict(Counter(flag for r in rows for flag in r["qa_flags"].split(";") if flag)),
        "strata": {name: dict(sorted(Counter(f"{r['disease']}/{r['quality_class']}" for r in subset).items()))
                   for name, subset in subsets.items()},
        "group_overlap_train_val": 0,
        "notes": ["Provisional engineering subsets, not representative performance estimates.",
                  "overfit is nested in smoke_train; edge_cases can overlap RGBA examples in main subsets.",
                  "Empty full labels and duplicate-image annotations remain intact in edge checks.",
                  "Image duplicate groups are not patient identifiers."]}
    return config, subsets, summary


def write_selection(config_path: Path):
    config, subsets, summary = build_selection(config_path)
    output = ROOT / config["output_dir"]
    manifests = output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    summary["manifest_sha256"] = {}
    for name, rows in subsets.items():
        path = manifests / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary["manifest_sha256"][path.name] = sha256(path)
    (output / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["subset_counts"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data_trial_v1.json")
    write_selection(parser.parse_args().config.resolve())
