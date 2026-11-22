"""Check the standalone RGB prediction entry point against stored held-out masks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from prepare import ROOT
import paired
from evaluate import make_predictor,probabilities
from two_scale import score


def main(teacher=False):
    torch.set_num_threads(2)
    cases,_,dev=paired.dataset(0)
    name=next(n for n in dev if cases[n]["i"].any())
    d=cases[name]
    records=[]
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp)
        img=directory/(name+".png")
        Image.fromarray(d["rgb"]).save(img)
        subjects=(("correction","native"),("extra","dual")) if teacher else (("standard","dual"),("branch","native"),("branch_aux","dual"))
        for method,view in subjects:
            group="teacher" if teacher else "paired" if method=="standard" else "focus"
            output=directory/(method+".png")
            subprocess.run([sys.executable,"-B",str(ROOT/"predict_image.py"),str(img),str(output),"--fold","0",
                            "--seed","2027","--method",method,"--view",view],check=True)
            with Image.open(output) as imgout: mask=np.asarray(imgout)>0
            folder=ROOT/("eval_"+group)/"fold0/2027"/method
            threshold=json.loads((folder/"calibration.json").read_text())["thresholds"][view]
            pr=probabilities(make_predictor(0,2027,method),d["rgb"],native_only=view=="native")[view]
            assert np.array_equal(mask,pr>=threshold)
            with np.load(folder/(name+".npz")) as saved:
                prior=np.unpackbits(saved[view+"_matched"]).reshape(mask.shape).astype(bool)
            delta=(mask!=prior)&d["f"]
            if teacher: assert not delta.any()
            records.append(dict(image=name,method=method,view=view,sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                                archived_changed_pixels=int(delta.sum()),fov=int(d["f"].sum()),
                                max_changed_probability_margin=float(np.max(abs(pr[delta]-threshold),initial=0)),
                                current=score(d,pr,threshold),archived=score(d,prior,.5)))
    (ROOT/("results_teacher" if teacher else "results_focus")/"rgb_verification.json").write_text(json.dumps(records,indent=2))
    print("Standalone label-free RGB and independent deterministic replay agree bit-for-bit; archived differences recorded.")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher",action="store_true")
    main(ap.parse_args().teacher)
