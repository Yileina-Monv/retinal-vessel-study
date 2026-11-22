"""Compare every new arm to the earlier 1200-step research candidate on identical photos."""
import json
import numpy as np
from scipy import ndimage as ndi
from prepare import ROOT,split
from mine_thin_nested import OUT
from report_paired_cycle import summarize
import paired

def read(path):
    return json.loads(path.read_text())

def main():
    cases=paired.dataset()[0]
    references=read(ROOT/"far_fragments/segments.json")
    for fold in range(3):
        for seed in (2027,2028,2029):
            prior=ROOT/"eval_teacher"/f"fold{fold}"/str(seed)/"correction"
            lookup={(r["image"],r["point"]):r for r in read(prior/"dev.json") if r["view"]=="dual"}
            rows={m:[] for m in "ABCD"}
            data={m:{(r["image"],r["point"]):r for r in read(OUT/f"fold{fold}"/str(seed)/m/"dev.json")["rows"]
                     if r["view"]=="dual"} for m in rows}
            for name in split(fold)["dev"]:
                d=cases[name]; thin=d["s"]&d["n"]; far=d["s"]&d["f"]&~(d["h"]|d["n"])
                xy=[pos for r in references if r["image"]==name for pos in r["yx"]]
                with np.load(prior/(name+".npz")) as z:
                    for pt in ("m0matched","matched"):
                        mask=np.unpackbits(z["dual_"+pt]).reshape(thin.shape).astype(bool)
                        h=ndi.binary_dilation(mask,structure=paired.cross)
                        hard=int(h[tuple(np.array(xy).T)].sum()) if xy else 0
                        for m,tab in data.items():
                            r=tab[name,pt]
                            b=np.unpackbits(np.frombuffer(bytes.fromhex(r["hit"]),dtype=np.uint8))[:thin.sum()].astype(bool)
                            a=h[thin]
                            src=lookup[name,pt]
                            base={k:src[k] for k in r["metrics"] if k in src}
                            base["thin_d0"]=float(mask[thin].mean()) if thin.any() else None
                            base["far_thin_d1"]=float(h[far].mean()) if thin.any() and far.any() else None
                            assert (float(a.mean()) if a.size else None)==base["thin_d1"]
                            rows[m].append(dict(image=name,point=pt,base=base,candidate=r["metrics"],hard_base=hard,
                                hard_cand=r["hard"],gains=int((b&~a).sum()),losses=int((a&~b).sum())))
            for m,rec in rows.items():
                path=OUT/(m+"_vs_correction_dual");path.mkdir(exist_ok=True)
                (path/f"fold{fold}_{seed}.json").write_text(json.dumps(dict(fold=fold,seed=seed,rows=rec)))
    for m in "ABCD": summarize(OUT/(m+"_vs_correction_dual"))

if __name__=="__main__":
    main()
