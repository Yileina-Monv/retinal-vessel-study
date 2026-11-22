"""Refit a fixed seed on all 54 development images, then calibrate using only those images."""
import hashlib
import json

import numpy as np
import torch

import paired as p
from matched_fp import GRID, calibration
from two_scale import predict


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(method, items, names, folder, seed):
    dest = folder / (method + "_weights.pt")
    model = p.UNet().cuda()
    if dest.exists():
        model.load_state_dict(torch.load(dest,map_location="cpu",weights_only=True))
        return model
    model.load_state_dict(torch.load(p.CHECKPOINT,map_location="cpu",weights_only=True)["model"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
    for _ in range(p.STEPS):
        x, vessel, fov, near, ids, _ = p.draw(items,names,rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss = p.loss_fn(model(x),vessel,fov,near,ids,method)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    tmp = dest.with_suffix(".tmp")
    torch.save(model.state_dict(),tmp)
    tmp.replace(dest)
    return model


def main():
    torch.set_num_threads(2)
    assert sha(p.CHECKPOINT) == p.SHA
    items, _, _ = p.dataset()
    names = sorted(items)
    folder = p.HERE / "research_model"
    folder.mkdir(exist_ok=True)
    seed = 2027
    base = train("base",items,names,folder,seed)
    target = float(calibration(base,items,names)[499])
    del base
    model = train("local",items,names,folder,seed)
    curves = []
    for name in names:
        d = items[name]
        if not d["n"].any():
            continue
        _, prob = predict(model,d)
        bg = np.sort(prob[d["n"] & ~d["v"]])
        curves.append((len(bg)-np.searchsorted(bg,GRID))/d["n"].sum())
    curve = np.mean(curves,axis=0)
    idx = np.flatnonzero(curve <= target)[0]
    info = dict(seed=seed,threshold=float(GRID[idx]),native_base_train_fp=target,
                local_train_fp=float(curve[idx]),scales=[1,2],steps=p.STEPS,train=names,
                sha256=sha(folder/"local_weights.pt"),base_sha256=sha(folder/"base_weights.pt"),
                initial_sha256=p.SHA)
    (folder/"calibration.json").write_text(json.dumps(info,indent=2))
    print("Refit complete",info["threshold"],info["sha256"],flush=True)


if __name__ == "__main__":
    main()
