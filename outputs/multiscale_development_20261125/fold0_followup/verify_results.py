"""Cross-check the published cohort tables, in percentage points."""
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-7):
        raise ValueError(f"{actual} differs from {expected}")


def main():
    checks = read("audit/sensitivity.csv")
    interactions = read("audit/interaction.csv")
    angles = read("audit/angle_summary.csv")
    for cohort, path, factor in [("IDRiD", "fold0/model_comparison.csv", 100),
                                 ("MAPLES", "transfer/transfer_summary.csv", 1)]:
        source = read(path)
        effects = {}
        for model in ("unet", "nnunet"):
            arms = {r["arm"]: r for r in source if r["model"] == model}
            assert set(arms) == {"1", "1.25", "1.5", "1+1.5"} and len(source) == 8
            for metric in ("D1", "far_recall", "dice"):
                delta = factor * (float(arms["1+1.5"][metric]) - float(arms["1"][metric]))
                row, = [r for r in checks if (r["dataset"], r["model"], r["point"], r["band"], r["metric"])
                        == (cohort, model, "matched", "W<=3", metric)]
                close(delta, row["mean"])
                effects[model, metric] = delta
                if metric == "D1":
                    close(arms["1"]["images"], row["n"])
                    if cohort == "IDRiD":
                        anchor, = [r for r in angles if r["model"] == model and float(r["nominal_angle_deg"]) == 50]
                        close(delta, anchor["mean_delta_D1_pp"])
                        close(arms["1"]["points"], anchor["points"])
        for metric in ("D1", "far_recall", "dice"):
            row, = [r for r in interactions if (r["dataset"], r["point"], r["metric"]) == (cohort, "matched", metric)]
            close(effects["unet", metric] - effects["nnunet", metric], row["mean"])
    print("Verified four model/cohort contrasts, paired interactions and 50-degree replay.")


if __name__ == "__main__":
    main()
