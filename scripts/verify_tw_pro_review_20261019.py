#!/usr/bin/env python3
"""Read pinned public Git objects and check review arithmetic; never load model/data assets.

Requires NumPy. CI rules below are numerical reconstructions, not recovered run code.
No network, training, inference, threshold selection, or writes unless --output is supplied.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SHA = "28a281fa2be1167533bcd0a54eca544a6e1f565d"
ROOT = Path(__file__).resolve().parents[1]
PREFIX = "outputs/maples_training_results_20261019/"
LABELS = {
    "alpha=1.5 minus alpha=1/T0": "1.5-1",
    "alpha=3.0 minus alpha=1/T0": "3-1",
    "2/1 minus 1/1/T0": "2/1-1/1",
    "2/1 minus 2/2/TW": "2/1-2/2",
}


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence destination")
    args = parser.parse_args()
    sources = []

    def raw(path):
        data = subprocess.check_output(["git", "show", f"{SHA}:{path}"], cwd=ROOT)
        sources.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(),
                        "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()})
        return data

    def table(name):
        return list(csv.DictReader(io.StringIO(raw(PREFIX + name).decode("utf-8-sig"))))

    configs = table("training_config_summary.csv")
    loss = table("loss_curve_200_update_windows.csv")
    endpoint = table("fixed_endpoint_per_image_metrics.csv")
    paired = table("paired_image_deltas.csv")
    summary = table("paired_comparison_summary.csv")
    audit = json.loads(raw(PREFIX + "source_audit.json"))
    assert [len(x) for x in (configs, loss, endpoint, paired, summary)] == [17, 190, 3836, 2131, 87]
    assert sum(int(x["successful_updates"]) for x in configs) == 34000
    assert Counter(int(x["starting_step"]) for x in configs) == {0: 15, 2000: 2}
    assert all(x["scheduler"] == "none" for x in configs)
    windows = {}
    duplicates = 0
    for x in loss:
        assert int(x["n_updates"]) == 200
        key = (x["condition"], x["seed"], x["window_start_step"], x["window_end_step"])
        values = (x["mean_total_loss"], x["sd_total_loss"])
        if key in windows:
            assert windows[key] == values
            duplicates += 1
        windows[key] = values
    assert (len(windows), duplicates) == (170, 20)

    lookup = {}
    image_groups = defaultdict(dict)
    valid, na = 0, 0
    for x in endpoint:
        key = (x["study_block"], x["condition"], x["seed"], x["step"], x["image_id"], x["metric"])
        assert key not in lookup, key
        lookup[key] = x
        image_groups[key[:-1]][x["metric"]] = x
        if x["value"]:
            close(float(x["value"]), float(x["numerator"]) / float(x["denominator"]))
            assert not x["na_reason"]
            valid += 1
        else:
            assert x["na_reason"] == "zero_denominator" and float(x["denominator"]) == 0
            na += 1
    assert (len(image_groups), valid, na) == (476, 3103, 733)
    for group in image_groups.values():
        tp = float(group["recall"]["numerator"])
        assert tp == float(group["precision"]["numerator"])
        pred, ref = float(group["precision"]["denominator"]), float(group["recall"]["denominator"])
        assert float(group["dice"]["numerator"]) == 2 * tp
        assert float(group["dice"]["denominator"]) == pred + ref
        assert float(group["fp_density"]["numerator"]) == pred - tp

    # Explicit endpoint terms: coefficient, study block, condition, step.
    t0 = ("formal", "T0", "2000")
    tw = ("formal", "TW", "2000")
    t4 = ("budget_exploratory", "T0", "4000")
    w4 = ("budget_exploratory", "TW", "4000")
    cls = ("class_split_exploratory", "w_v2_b1", "2000")
    expressions = {
        "TW-T0": [(1, tw), (-1, t0)],
        "1.5-1": [(1, ("weight_ablation", "alpha=1.5", "2000")), (-1, t0)],
        "3-1": [(1, ("weight_ablation", "alpha=3", "2000")), (-1, t0)],
        "2/1-1/1": [(1, cls), (-1, t0)],
        "2/1-2/2": [(1, cls), (-1, tw)],
        "T04000_minus_T02000": [(1, t4), (-1, t0)],
        "TW4000_minus_TW2000": [(1, w4), (-1, tw)],
        "TW4000_minus_T04000": [(1, w4), (-1, t4)],
        "interaction_4000_minus_2000": [(1, w4), (-1, t4), (-1, tw), (1, t0)],
    }
    groups = defaultdict(list)
    bridge_rows = bridge_cells = 0
    complete_groups = defaultdict(list)
    for x in paired:
        key = (x["study_block"], x["contrast"], x["metric"])
        groups[key].append(x)
        seed_vals = [float(x[f"seed_{seed}"]) for seed in (2026, 2027, 2028) if x[f"seed_{seed}"]]
        if seed_vals:
            close(statistics.mean(seed_vals), float(x["mean_delta"]))
        else:
            assert not x["mean_delta"]
        row_complete = True
        active_seeds = (2026,) if x["study_block"] == "budget_extension" else (2026, 2027, 2028)
        for seed in active_seeds:
            if not x[f"seed_{seed}"]:
                missing_sources = [lookup.get((block, condition, str(seed), step, x["image_id"], x["metric"]))
                                   for _, (block, condition, step) in expressions[x["contrast"]]]
                if any(source is None for source in missing_sources):
                    row_complete = False
                else:
                    assert all(not source["value"] for source in missing_sources)
                continue
            terms = []
            for coefficient, (block, condition, step) in expressions[x["contrast"]]:
                source = lookup.get((block, condition, str(seed), step, x["image_id"], x["metric"]))
                if source is None or not source["value"]:
                    row_complete = False
                    break
                terms.append(coefficient * float(source["value"]))
            else:
                close(sum(terms), float(x[f"seed_{seed}"]))
                bridge_cells += 1
        if row_complete:
            bridge_rows += 1
        complete_groups[key].append(row_complete)

    cache, formal_rng = {}, np.random.default_rng(20261016)
    max_ci_error = 0.0
    checked_sd = checked_complete = 0
    formal = {}
    for x in summary:
        assert x["bootstrap_replicates"] == "1000" and x["rng"] == "PCG64(20261016)"
        contrast = LABELS.get(x["contrast"], x["contrast"])
        key = (x["study_block"], contrast, x["metric"])
        rows = [r for r in groups[key] if r["mean_delta"]]
        values = np.array([float(r["mean_delta"]) for r in rows])
        assert len(values) == int(x["n_images"])
        close(float(values.mean()), float(x["estimate"]))
        if all(complete_groups[key]):
            checked_complete += 1
        sd_vals = []
        for seed in (2026, 2027, 2028):
            vals = [float(r[f"seed_{seed}"]) for r in rows if r[f"seed_{seed}"]]
            if vals:
                sd_vals.append(statistics.mean(vals))
                if x[f"seed_{seed}"]:
                    close(sd_vals[-1], float(x[f"seed_{seed}"]))
        if x["seed_sd"] and x["seed_sd"] != "reported_in_source" and len(sd_vals) >= 2:
            close(statistics.stdev(sd_vals), float(x["seed_sd"]))
            checked_sd += 1
        n = len(values)
        if x["study_block"] == "formal_core":
            if n not in cache:
                cache[n] = formal_rng.integers(0, n, size=(1000, n))
            indexes = cache[n]
            formal[x["metric"]] = {k: float(x[k]) for k in ("estimate", "ci95_low", "ci95_high")}
        else:
            indexes = np.random.default_rng(20261016).integers(0, n, size=(1000, n))
        ci = np.quantile(values[indexes].mean(axis=1), [0.025, 0.975], method="linear")
        errors = np.abs(ci - np.array([float(x["ci95_low"]), float(x["ci95_high"])]))
        max_ci_error = max(max_ci_error, float(errors.max()))
        assert float(errors.max()) < 1e-12, (key, ci, x)
    assert (checked_complete, bridge_rows, bridge_cells, checked_sd) == (63, 1619, 3173, 40), (checked_complete, bridge_rows, bridge_cells, checked_sd)

    totals, denominator_by_image = {}, {}
    seed_macros = {}
    for condition in ("T0", "TW", "w_v2_b1"):
        rows = [r for r in endpoint if r["condition"] == condition and r["step"] == "2000" and r["metric"] == "near_thin_recall_d1"]
        good = [r for r in rows if r["value"]]
        assert len(good) == 27
        totals[condition] = sum(int(float(r["numerator"])) for r in good)
        assert sum(int(float(r["denominator"])) for r in good) == 231
        seed_macros[condition] = {str(seed): statistics.mean(float(r["value"]) for r in good if r["seed"] == str(seed)) for seed in (2026, 2027, 2028)}
        for r in good:
            d = int(float(r["denominator"]))
            assert denominator_by_image.setdefault(r["image_id"], d) == d
    assert sorted(denominator_by_image.values()) == [1, 1, 3, 4, 6, 8, 13, 14, 27]
    assert totals == {"T0": 106, "TW": 104, "w_v2_b1": 106}
    assert audit["human_visual_review_count"] == 0 and audit["ai_visual_review_exported"] is False
    assert audit["official_test_accessed"] is False

    result = {
        "source_sha": SHA,
        "scope": "public-table arithmetic only; no image, probability, execution binding or clinical validation",
        "numpy_version": np.__version__,
        "row_counts": dict(zip(("config", "loss", "endpoint", "paired", "summary"), (17, 190, 3836, 2131, 87))),
        "fresh_trajectories": 15, "continuation_segments": 2, "successful_updates_recorded": 34000,
        "unique_loss_windows": 170, "repeated_prefix_windows": 20,
        "endpoint_valid": valid, "endpoint_na": na, "condition_image_groups": len(image_groups),
        "endpoint_paired_rows_checked": bridge_rows, "endpoint_seed_cells_checked": bridge_cells,
        "complete_endpoint_paired_summary_comparisons": checked_complete,
        "paired_summary_only_comparisons": 87 - checked_complete,
        "summary_means_checked": 87, "numeric_seed_sd_checked": checked_sd,
        "ci_numerical_reconstructions_checked": 87, "ci_max_abs_error": max_ci_error,
        "ci_rule_status": "numerical reconstruction only; historical bootstrap source unavailable",
        "formal_core_ratio_units": formal,
        "formal_main_seed_deltas_pp": {s: 100 * (seed_macros["TW"][s] - seed_macros["T0"][s]) for s in seed_macros["T0"]},
        "main_hits_across_three_seeds": totals, "main_reference_counts": denominator_by_image,
        "human_visual_review_count_in_source": 0,
        "plan_b_joint_matrix_independently_verified": False,
        "test_access_statement": "source declares false; underlying access history not independently audited",
        "sources": sources,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
