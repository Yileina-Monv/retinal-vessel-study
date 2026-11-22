"""Exercise both the cross-validation and full-data RGB inference artifacts."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import paired
from two_scale import score


def main():
    torch.set_num_threads(2)
    root = paired.HERE
    items, _, dev = paired.dataset()
    name = next(n for n in dev if items[n]["i"].any())
    case = items[name]
    bundle = root / "research_model"
    manifest = json.loads((bundle/"calibration.json").read_text())
    assert hashlib.sha256((bundle/"local_weights.pt").read_bytes()).hexdigest() == manifest["sha256"]
    heldout = root / "two_scale/local_2027"
    expected = next(r for r in json.loads((heldout/"dev.json").read_text()) if r["image"]==name)
    info = {}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        rgb = folder / "input.png"
        Image.fromarray(case["rgb"]).save(rgb)
        for tag, weights, cal in [
            ("heldout",root/"rgb_only/local_2027/weights.pt",heldout/"threshold.json"),
            ("refit",bundle/"local_weights.pt",bundle/"calibration.json"),
        ]:
            output = folder/(tag+".png")
            subprocess.run([sys.executable,"-B",str(root/"predict.py"),str(weights),str(rgb),
                            str(output),"--two-scale-calibration",str(cal)],check=True)
            mask = np.asarray(Image.open(output))
            assert mask.shape==(1024,1024) and set(np.unique(mask))=={0,255}
            if tag=="heldout":
                checks = score(case,mask,128)
                assert all(value==expected[key] for key,value in checks.items())
            info[tag] = dict(image=name, foreground=int((mask>0).sum()),
                             mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    (bundle/"inference_verification.json").write_text(json.dumps(info,indent=2))
    print("Verified",name,info)


if __name__ == "__main__":
    main()
