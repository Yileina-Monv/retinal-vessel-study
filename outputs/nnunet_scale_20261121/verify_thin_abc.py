"""Audit training provenance and recalculate the main endpoint from stored point hits."""
import hashlib
import json
import numpy as np
from prepare import ROOT,split
from mine_thin_nested import OUT
from report_paired_cycle import summarize

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    counts={r["image"]:r["thin_points"] for r in read(ROOT/"thin_recall_audit.json")}
    audit=[]
    for fold in range(3):
        tr=set(split(fold)["train"]);dev=set(split(fold)["dev"])
        assert not tr&dev
        for seed in (2027,2028,2029):
            source=sha(ROOT/"models_focus"/f"fold{fold}"/str(seed)/"continue/weights.pt")
            traces={}
            for arm in ("A","B","C","D"):
                p=OUT/f"fold{fold}"/str(seed)/arm
                meta=read(p/"train.json"); ev=read(p/"dev.json")
                assert source==meta["parent_sha"] and len(meta["loss"])==meta["steps"]==400
                assert len(ev["rows"])==72 and {r["image"] for r in ev["rows"]}==dev
                for r in ev["rows"]:
                    n=counts[r["image"]]
                    pt=np.unpackbits(np.frombuffer(bytes.fromhex(r["hit"]),dtype=np.uint8))
                    assert len(pt)>=n and not pt[n:].any()
                    assert (float(pt[:n].mean()) if n else None)==r["metrics"]["thin_d1"]
                traces[arm]=meta["trace"]
                audit.append(dict(fold=fold,seed=seed,arm=arm,weight_sha=sha(p/"weights.pt"),
                    parent=source,updates=400-meta["amp_skips"],targeted=meta["targeted"]))
            assert traces["B"]==traces["C"]==traces["D"] and traces["A"]!=traces["B"]
    for sub in OUT.glob("*_vs_*"):
        summarize(sub)
    (OUT/"verification.json").write_text(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
