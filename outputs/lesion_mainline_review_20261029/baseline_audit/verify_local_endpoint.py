"""Verify the descriptive two-seed local endpoint using public CSVs only.

No image, model, GPU, network, or Test access. Run from any directory.
Writes local_endpoint_summary.json beside this script.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for source in json.loads((ROOT / "provenance.json").read_text()):
    content = (ROOT / source["path"]).read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(content)).encode() + b"\0" + content
    ).hexdigest()
    assert blob == source["git_blob_sha"], source["path"]

pairs = pd.read_csv(ROOT / "new/paired_image_deltas.csv",
                    float_precision="round_trip")
metrics = pd.read_csv(ROOT / "new/fixed_endpoint_per_image_metrics.csv",
                      float_precision="round_trip")
metric = "near_thin_recall_d1"
subsets = {
    seed: pairs[(pairs.seed == seed) & (pairs.metric == metric)]
    .sort_values("pair_index").reset_index(drop=True)
    for seed in (2027, 2028)
}
for sub in subsets.values():
    assert len(sub) == 9 and sub.pair_index.tolist() == list(range(9))
assert subsets[2027].image.tolist() == subsets[2028].image.tolist()

rows = []
for image in subsets[2027].image:
    record = {"image": image}
    for seed in (2027, 2028):
        for method in ("A0", "NNUNET"):
            subset = metrics[(metrics.seed == seed)
                             & (metrics.method == method)
                             & (metrics.metric == metric)
                             & (metrics.image == image)]
            assert len(subset) == 1
            row = subset.iloc[0]
            assert row.denominator > 0
            record[f"{method}_{seed}_hits"] = int(row.numerator)
            if "denominator" in record:
                assert record["denominator"] == row.denominator
            else:
                record["denominator"] = int(row.denominator)
            assert abs(row.value - row.numerator / row.denominator) < 1e-12
            paired_value = subsets[seed].set_index("image").loc[image, method]
            assert abs(row.value - paired_value) < 1e-12
    rows.append(record)
assert sum(r["denominator"] for r in rows) == 77
for seed in (2027, 2028):
    assert sum(r[f"NNUNET_{seed}_hits"] for r in rows) == 59
    for method in ("A0", "NNUNET"):
        arm = metrics[(metrics.seed == seed) & (metrics.method == method)
                      & (metrics.metric == metric)]
        assert len(arm) == 28 and arm.value.isna().sum() == 19

delta = (subsets[2027].delta.to_numpy()
         + subsets[2028].delta.to_numpy()) / 2
indices = np.asarray(json.loads(
    (ROOT / "new/bootstrap_indices_by_n.json").read_text())["9"])
assert indices.shape == (1000, 9)
low, high = np.quantile(delta[indices].mean(axis=1),
                        [.025, .975], method="linear")
leave_one_out = [float(np.delete(delta, i).mean() * 100) for i in range(9)]
result = {
    "scope": "Descriptive image bootstrap conditional on fixed seeds 2027/2028.",
    "metric": metric,
    "n_images": 9,
    "reference_positions": 77,
    "per_seed_NNUNET_hits": {"2027": 59, "2028": 59},
    "mean_delta_pp": float(delta.mean() * 100),
    "lower_pp": float(low * 100),
    "upper_pp": float(high * 100),
    "positive_zero_negative_images": [
        int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())
    ],
    "leave_one_out_mean_pp": leave_one_out,
    "counts": rows,
    "limitations": [
        "Repeated Dev; not independent validation.",
        "Does not cover seed population, patient clustering, or selection uncertainty.",
        "Counts do not reveal overlap of missed positions or establish connectivity.",
        "The 18 published input blobs are checked; no pixel evaluator is rerun."
    ],
}
assert abs(result["mean_delta_pp"] - 14.0650) < .0001
assert abs(result["lower_pp"] - (-6.9618)) < .0001
assert abs(result["upper_pp"] - 32.1898) < .0001
(ROOT / "local_endpoint_summary.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in result.items()
                  if k not in ("counts", "leave_one_out_mean_pp")}, indent=2))
