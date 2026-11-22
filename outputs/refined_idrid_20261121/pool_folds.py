"""Pool held-out predictions; count each unique image once per seed."""
import argparse
import json
from pathlib import Path
from analyze import HERE, SEEDS, summarize


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--calibrated",action="store_true")
    ap.add_argument("--reverse",action="store_true")
    ap.add_argument("--two-scale",action="store_true")
    ap.add_argument("--vs-native",action="store_true")
    args=ap.parse_args()
    if args.vs_native and not args.two_scale: ap.error("--vs-native requires --two-scale")
    if args.two_scale and (args.calibrated or args.reverse): ap.error("Use one comparison at a time")
    stages=[HERE,HERE/"fold_1",HERE/"fold_2"]
    for required in ([1,2],[0,1,2]):
        out=HERE/"out_of_fold"/("confirmation" if len(required)==2 else "all")
        if args.calibrated: out=out/"calibrated"
        if args.reverse: out=out/"reverse_local"
        if args.two_scale: out=out/"two_scale"
        if args.vs_native: out=out/"vs_native"
        splits=[json.loads((s/"split.json").read_text()) for s in stages]
        ids=sorted(name for j in required for name in splits[j]["dev"])
        if len(ids)!=len(set(ids)) or len(ids)!=18*len(required): raise ValueError("Duplicate holdout")
        for seed in SEEDS:
            for method in ("base","local"):
                rows=[]
                for j in required:
                    split=splits[j]
                    if set(split["train"]) & set(split["dev"]): raise ValueError("Split overlap")
                    f=stages[j]/("two_scale" if args.two_scale else "rgb_only")
                    if args.vs_native: f=f/"vs_native"
                    if args.calibrated: f=f/"calibrated"
                    if args.reverse: f=f/"reverse_local"
                    part=json.loads((f/f"{method}_{seed}"/"dev.json").read_text())
                    if [r["image"] for r in part]!=split["dev"]: raise ValueError("Wrong fold")
                    rows.extend(dict(fold=j,**r) for r in part)
                rows.sort(key=lambda r:r["image"])
                d=out/f"{method}_{seed}";d.mkdir(parents=True,exist_ok=True)
                (d/"dev.json").write_text(json.dumps(rows,indent=2))
        print(required)
        summarize(out,images=ids)


if __name__=="__main__": main()
