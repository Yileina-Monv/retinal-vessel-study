#!/usr/bin/env python3
"""Validate the sanitized public MAPLES training-results tables; no image/model assets required."""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/maples_training_results_20261019")
def read(name):
    with (root / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

configs = read("training_config_summary.csv")
loss = read("loss_curve_200_update_windows.csv")
metrics = read("fixed_endpoint_per_image_metrics.csv")
paired = read("paired_image_deltas.csv")
summary = read("paired_comparison_summary.csv")
audit = json.loads((root / "source_audit.json").read_text(encoding="utf-8"))
assert len(configs) == 17, len(configs)
assert Counter(r["experiment"] for r in configs) == {
    "formal_core": 6, "weight_ablation": 6, "class_split": 3, "budget_extension": 2
}
assert {r["seed"] for r in configs} == {"2026", "2027", "2028"}
assert all(r["optimizer"] == "AdamW" and float(r["learning_rate"]) == 1e-4 for r in configs)
assert all(float(r["protocol_threshold"]) == 0.5 and r["official_test_accessed"].lower() == "false" for r in configs)
assert len(loss) == 190
assert all(int(r["n_updates"]) == 200 for r in loss)
assert len(metrics) == 3836
assert len(paired) == 2131
assert len(summary) == 87
assert all(float(r["threshold"]) == 0.5 for r in metrics + summary)
for r in metrics:
    if r["value"] == "":
        assert r["na_reason"] == "zero_denominator" and r["denominator"] == "0", r
    else:
        assert r["na_reason"] == ""
        v = float(r["value"]); n = float(r["numerator"]); d = float(r["denominator"])
        assert d > 0 and math.isclose(v, n / d, rel_tol=1e-10, abs_tol=1e-12), r
means = defaultdict(list)
for r in paired:
    if r["mean_delta"] != "":
        means[(r["study_block"], r["contrast"], r["metric"])].append(float(r["mean_delta"]))
label_map = {
    ("weight_ablation", "alpha=1.5 minus alpha=1/T0"): "1.5-1",
    ("weight_ablation", "alpha=3.0 minus alpha=1/T0"): "3-1",
    ("class_split", "2/1 minus 1/1/T0"): "2/1-1/1",
    ("class_split", "2/1 minus 2/2/TW"): "2/1-2/2",
}
for r in summary:
    contrast = label_map.get((r["study_block"], r["contrast"]), r["contrast"])
    vals = means[(r["study_block"], contrast, r["metric"])]
    assert len(vals) == int(r["n_images"]), (r, len(vals))
    assert math.isclose(sum(vals) / len(vals), float(r["estimate"]), rel_tol=1e-10, abs_tol=1e-12), r
assert all(c["status"] == "passed" and c["all_passed"] for c in audit["source_checks"])
assert all(not c["official_test_accessed"] for c in audit["source_checks"])
assert audit["official_test_accessed"] is False
print(f"passed: {len(configs)} configs, {len(loss)} loss windows, {len(metrics)} per-image metric rows, {len(paired)} paired image deltas, {len(summary)} contrasts")
