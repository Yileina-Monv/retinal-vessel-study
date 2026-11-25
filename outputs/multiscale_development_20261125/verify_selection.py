"""Recompute the constrained source choices and headline effects from cohort means."""
import csv
from pathlib import Path

with Path(__file__).with_name("scale_grid.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))

expected = {
    "2027": {"maples": (19.74, -0.02, 0.53), "idrid": (6.69, 0.55, 1.13)},
    "2028": {"maples": (19.15, 0.45, 0.42), "idrid": (4.42, 0.73, 0.80)},
}
for seed, domains in expected.items():
    arms = {r["arm"]: r for r in rows if r["seed"] == seed}
    assert len(arms) == 15
    native = arms["1"]
    for kind, wanted in [(False, "1.25" if seed == "2027" else "1"), (True, "1+1.5")]:
        eligible = [
            r for name, r in arms.items()
            if ("+" in name) == kind and float(r["maples_fp"]) <= 15
            and all(float(r[k]) >= float(native[k]) - 1 for k in ("maples_far", "maples_Dice"))
        ]
        best = min(eligible, key=lambda r: (-float(r["maples_D1"]), float(r["windows"]), r["arm"]))
        assert best["arm"] == wanted
    for domain, target in domains.items():
        keys = [domain + "_" + m for m in ("D1", "far", "Dice")]
        delta = [float(arms["1+1.5"][k]) - float(native[k]) for k in keys]
        assert all(abs(x-y) < 0.0051 for x, y in zip(delta, target))
        print(seed, domain, *(f"{x:+.2f}" for x in delta))
