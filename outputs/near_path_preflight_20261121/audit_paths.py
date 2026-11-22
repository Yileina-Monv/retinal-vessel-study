"""CPU audit of the published N0 path table. Run with Python 3, no dependencies."""
import csv
import hashlib
import io
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_SHA256 = "8d2b86babfd312392c7ded6029cfe4dba84486d172ee299d0402f315429cfa77"


def edge_count(length):
    # Each native 8-neighbour edge has length 1 or sqrt(2).
    matches = []
    for diagonal in range(math.floor(length / math.sqrt(2) + 1e-8) + 1):
        axial = length - diagonal * math.sqrt(2)
        if abs(axial - round(axial)) < 1e-8 and round(axial) >= 0:
            matches.append(round(axial) + diagonal)
    if len(matches) != 1:
        raise ValueError(f"Ambiguous native edge count for {length}")
    return matches[0]


def classify(row):
    size, hits, runs = (int(row[key]) for key in ("near_points", "near_hits_d1", "valid_near_runs"))
    edges = edge_count(float(row["edge_length"]))
    hit_edges = edge_count(float(row["edge_hit_length"]))
    if hits == size:
        return "full", 0
    if not hits:
        return "zero", 0
    if edges + runs != size:
        return "repeat_coordinates", None
    hit_runs = hits - hit_edges
    if not 1 <= hit_runs <= hits:
        raise ValueError("Invalid number of hit runs")
    gaps_min = max(0, hit_runs - runs)
    if gaps_min:
        return "internal_gap", gaps_min
    if runs == 1:
        return "terminal_only", 0
    return "multiple_runs_unresolved", 0


def main():
    source = (ROOT / "historical_paths.csv").read_text(encoding="utf-8-sig")
    canonical = ("\ufeff" + source.rstrip("\n").replace("\n", "\r\n") + "\r\n").encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source does not match the published archival SHA-256")
    rows = list(csv.DictReader(io.StringIO(source)))
    paired = {}
    for row in rows:
        if row["method"] != "N0":
            continue
        key = row["filename"], int(row["branch_id"])
        seed = row["seed"]
        if seed in paired.setdefault(key, {}):
            raise ValueError("Duplicate path")
        paired[key][seed] = row
    if len(rows) != 1740 or len(paired) != 174:
        raise ValueError("Incomplete archive")
    audit = []
    for (image, branch), seeds in sorted(paired.items()):
        first, second = seeds["2027"], seeds["2028"]
        for field in ("near_points", "valid_near_runs", "edge_length"):
            if first[field] != second[field]:
                raise ValueError("Reference differs between seeds")
        size = int(first["near_points"])
        h1, h2 = (int(row["near_hits_d1"]) for row in (first, second))
        a, b = classify(first), classify(second)
        audit.append(dict(filename=image, branch_id=branch, near_points=size, near_runs=first["valid_near_runs"],
                          hits_2027=h1, hits_2028=h2, class_2027=a[0], class_2028=b[0],
                          min_internal_gaps_2027=a[1], min_internal_gaps_2028=b[1],
                          min_shared_missed_points=max(0, size-h1-h2),
                          max_shared_missed_points=size-max(h1, h2)))
    with (ROOT / "path_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0]))
        w.writeheader()
        w.writerows(audit)
    print("N0 path classes (2027, 2028):")
    print(Counter((r["class_2027"], r["class_2028"]) for r in audit))
    for kind in ("internal_gap", "terminal_only", "zero"):
        both = [r for r in audit if r["class_2027"] == r["class_2028"] == kind]
        print(kind, len(both), "paths in", len({r["filename"] for r in both}), "images")
        print([(r["filename"], r["branch_id"], r["hits_2027"], r["hits_2028"], r["near_points"]) for r in both[:5]])
    partial = [r for r in audit if all(0 < r["hits_" + s] < r["near_points"] for s in ("2027","2028"))]
    for group in (audit, partial):
        print(len(group), "paths; shared missed position occurrences in",
              [sum(r[k] for r in group) for k in ("min_shared_missed_points", "max_shared_missed_points")])


if __name__ == "__main__":
    main()
