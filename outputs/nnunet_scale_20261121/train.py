import argparse
import hashlib
import json
import random
import time
from prepare import ROOT, split, stage
import numpy as np
import torch
from torch.nn import functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


def trainer(fold,seed):
    pre=ROOT/"preprocessed"/stage(fold)
    plans=json.loads((pre/"nnUNetPlans.json").read_text())
    meta=json.loads((ROOT/"raw"/stage(fold)/"dataset.json").read_text())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t=nnUNetTrainer(plans,"2d",fold,meta,device=torch.device("cuda"))
    t.num_epochs=10
    t.num_iterations_per_epoch=50
    t.initialize()
    return t


def loader(t,fold,names=None):
    rot,dummy,initial,mirrors=t.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    cfg=t.configuration_manager
    aug=t.get_training_transforms(cfg.patch_size,rot,t._get_deep_supervision_scales(),mirrors,dummy,
                                 use_mask_for_norm=cfg.use_mask_for_norm,foreground_labels=t.label_manager.foreground_labels,
                                 ignore_label=t.label_manager.ignore_label)
    ds=t.dataset_class(t.preprocessed_dataset_folder,split(fold)["train"] if names is None else names)
    return nnUNetDataLoader(ds,t.batch_size,initial,cfg.patch_size,t.label_manager,
                            oversample_foreground_percent=t.oversample_foreground_percent,transforms=aug,
                            probabilistic_oversampling=t.probabilistic_oversampling)


def zoom_selected(batch,selected):
    x=batch["data"]
    y=batch["target"][0]
    h,w=x.shape[-2:]
    crop=(slice(h//4,3*h//4),slice(w//4,3*w//4))
    for i in np.flatnonzero(selected):
        sl=slice(int(i),int(i)+1)
        x[sl]=F.interpolate(x[sl,:,crop[0],crop[1]],(h,w),mode="bilinear",align_corners=False)
        z=F.interpolate(y[sl,:,crop[0],crop[1]].float(),(h,w),mode="nearest-exact")
        for target in batch["target"]:
            target[sl]=F.interpolate(z,target.shape[-2:],mode="nearest-exact").to(target.dtype)


def train_pair(fold,seed):
    dest=ROOT/"models_paired"/f"fold{fold}"/str(seed)
    methods=("standard","mix")
    if all((dest/m/"train.json").is_file() for m in methods): return
    arms={m:trainer(fold,seed) for m in methods}
    init=hashlib.sha256()
    for a,b in zip(arms["standard"].network.state_dict().values(),arms["mix"].network.state_dict().values()):
        assert torch.equal(a,b)
        init.update(a.cpu().numpy().tobytes())
    dl=loader(arms["standard"],fold)
    trace=hashlib.sha256()
    start=time.monotonic()
    losses={m:[] for m in methods}
    skips={m:0 for m in methods}
    zoomed=0
    for step in range(500):
        key=seed*10000+step
        np.random.seed(key)
        torch.manual_seed(key)
        random.seed(key)
        batch=next(dl)
        trace.update(np.asarray(batch["data"][:,:,::32,::32]).tobytes())
        flags=np.random.default_rng(key+9999999).random(len(batch["data"]))<.5
        for method,t in arms.items():
            if step%50==0:
                t.current_epoch=step//50
                t.lr_scheduler.step(t.current_epoch)
                t.network.train()
            if method=="mix":
                zoom_selected(batch,flags)
                zoomed+=int(flags.sum())
            before=t.grad_scaler.get_scale()
            loss=float(t.train_step(batch)["loss"])
            t.optimizer.zero_grad(set_to_none=True)
            skips[method]+=int(t.grad_scaler.get_scale()<before)
            if not np.isfinite(loss): raise ValueError("Nonfinite loss")
            losses[method].append(loss)
        if (step+1)%50==0:
            print(fold,seed,step+1,{m:round(float(np.mean(losses[m][-50:])),4) for m in methods},
                  "seconds",round(time.monotonic()-start),flush=True)
    for m,t in arms.items():
        folder=dest/m
        folder.mkdir(exist_ok=True,parents=True)
        torch.save(t.network.state_dict(),folder/"weights.tmp")
        (folder/"weights.tmp").replace(folder/"weights.pt")
        info=dict(fold=fold,seed=seed,method=m,loss=losses[m],steps=len(losses[m]),amp_skips=skips[m],
                  zoomed=zoomed if m=="mix" else 0,init_sha=init.hexdigest(),batch_trace=trace.hexdigest(),
                  pair_seconds=time.monotonic()-start,patch=t.configuration_manager.patch_size,batch=t.batch_size)
        (folder/"train.tmp").write_text(json.dumps(info))
        (folder/"train.tmp").replace(folder/"train.json")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold",type=int,required=True)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    torch.set_num_threads(2)
    train_pair(args.fold,args.seed)
