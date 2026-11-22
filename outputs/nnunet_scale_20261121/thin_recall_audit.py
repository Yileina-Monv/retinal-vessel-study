"""Single-core audit of archived masks. Never imports torch or runs a model."""
import ast
import io
import json
import os
import tarfile
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import psutil
proc = psutil.Process()
if os.name == "nt":
    proc.nice(psutil.IDLE_PRIORITY_CLASS)
proc.cpu_affinity(proc.cpu_affinity()[-1:])

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

root = Path(__file__).resolve().parent
old = root.parent / "refined_idrid_20261121"
tree = ast.parse((old/"paired.py").read_text(encoding="utf-8"))
feature_code = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "features")
exec(compile(ast.Module(body=[feature_code], type_ignores=[]), "paired.py:features", "exec"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run():
    rows = []
    with tarfile.open(old/"Train.tar", "r:") as source:
        for fold in range(3):
            folder = root/"eval_teacher"/f"fold{fold}"
            lookup = {}
            for seed in (2027, 2028, 2029):
                dev = read(folder/str(seed)/"correction/dev.json")
                lookup[seed] = {(r["image"], r["view"]): r for r in dev if r["point"] == "m0matched"}
            for name in read((old if fold==0 else old/f"fold_{fold}")/"split.json")["dev"]:
                start = time.process_time()
                label = np.array(Image.open(io.BytesIO(source.extractfile(f"Train/Labels/{name}_vessel.png").read())))
                vessel, fov, ring, thin, ids = features(label)
                target = ring & thin
                yy, xx = np.where(target)
                rows_image = []
                n_hits, d_hits = [], []
                for seed in (2027, 2028, 2029):
                    counts, masks = {}, {}
                    with np.load(folder/str(seed)/"correction"/(name+".npz")) as z:
                        for view in ("native","large","dual"):
                            mask = np.unpackbits(z[view+"_m0matched"]).reshape(label.shape).view(bool)
                            ref = lookup[seed][name,view]
                            padded = np.pad(mask,1)
                            hit = np.logical_or.reduce([padded[yy+1+i,xx+1+j] for i,j in ((0,0),(-1,0),(1,0),(0,-1),(0,1))])
                            assert len(hit)==ref["thin_den"] and int(hit.sum())==ref["thin_hits"]
                            assert int((ring & ~vessel & mask).sum())==ref["near_fp"]
                            counts[view] = hit
                            masks[view] = mask
                    a,b,c = (counts[k] for k in ("native","large","dual"))
                    n_hits.append(a)
                    d_hits.append(c)
                    rows_image.append(dict(seed=seed, native=int(a.sum()), large=int(b.sum()), dual=int(c.sum()),
                        gained=int((~a&c).sum()), lost=int((a&~c).sum()),
                        only_native_vs_large=int((a&~b).sum()), only_large_vs_native=int((~a&b).sum()),
                        neither_view_nor_dual=int((~a&~b&~c).sum()),
                        dual_fp=ref["near_fp"], union_fp=int((ring&~vessel&(masks["native"]|masks["dual"])).sum())))
                ns,ds=np.stack(n_hits),np.stack(d_hits)
                rows.append(dict(image=name, fold=fold, thin_points=len(yy), points_on_training_paths=int((target & (ids>0)).sum()),
                    path_points=int((ids>0).sum()), ring_area=int(ring.sum()),
                    native_always_missed_dual_always_hit=int((~ns.any(0)&ds.all(0)).sum()),
                    dual_always_missed=int((~ds.any(0)).sum()), records=rows_image))
                time.sleep(max(.3, 5*(time.process_time()-start)))
    assert len(rows)==54 and sum(r["thin_points"] for r in rows)==3632
    (root/"thin_recall_audit.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

if __name__=="__main__":
    run()
