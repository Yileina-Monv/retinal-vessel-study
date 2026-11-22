"""Train-fold-only nnU-Net plans and native RGB preprocessing."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "refined_idrid_20261121"
sys.path.insert(0,str(OLD))
for key, name in (("nnUNet_raw","raw"),("nnUNet_preprocessed","preprocessed"),("nnUNet_results","framework_runs")):
    os.environ[key] = str(ROOT/name)
os.environ["nnUNet_compile"] = "false"
os.environ["nnUNet_n_proc_DA"] = "0"


def stage(fold):
    return "Dataset"+str(871+fold)+"_RefinedVessel"


def split(fold):
    folder=OLD if fold==0 else OLD/f"fold_{fold}"
    return json.loads((folder/"split.json").read_text())


def main():
    import numpy as np
    import torch
    import paired
    from PIL import Image
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    os.environ["nnUNet_n_proc_DA"] = "1"
    torch.set_num_threads(2)
    items, _, _ = paired.dataset()
    for fold in range(3):
        name=stage(fold)
        raw=ROOT/"raw"/name
        pre=ROOT/"preprocessed"/name
        done=pre/"ready.json"
        if done.is_file(): continue
        for sub in ("imagesTr","labelsTr"): (raw/sub).mkdir(parents=True,exist_ok=True)
        tr=split(fold)["train"]
        assert len(tr)==36 and not set(tr)&set(split(fold)["dev"])
        for k in tr:
            d=items[k]
            Image.fromarray(d["rgb"]).save(raw/"imagesTr"/(k+"_0000.png"))
            seg=np.where(~d["f"],2,d["v"]).astype(np.uint8)
            Image.fromarray(seg).save(raw/"labelsTr"/(k+".png"))
        meta=dict(channel_names={str(i):"rgb_to_0_1" for i in range(3)},labels=dict(background=0,vessel=1,ignore=2),
                  numTraining=len(tr),file_ending=".png",overwrite_image_reader_writer="NaturalImage2DIO")
        (raw/"dataset.json").write_text(json.dumps(meta))
        fpe=DatasetFingerprintExtractor(871+fold,1)
        fpe.num_foreground_voxels_for_intensitystats=180000
        fpe.run(overwrite_existing=False)
        plan_experiment_dataset(871+fold,gpu_memory_target_in_gb=2)
        plans_file=pre/"nnUNetPlans.json"
        plans=json.loads(plans_file.read_text())
        plans["configurations"]["2d"]["data_identifier"]="native_numpy_2d"
        plans_file.write_text(json.dumps(plans))
        pm=PlansManager(plans)
        cfg=pm.get_configuration("2d")
        pp=cfg.preprocessor_class(verbose=False)
        out=pre/cfg.data_identifier
        out.mkdir(exist_ok=True)
        for k in tr:
            data,seg,props=pp.run_case([str(raw/"imagesTr"/(k+"_0000.png"))],str(raw/"labelsTr"/(k+".png")),pm,cfg,meta)
            nnUNetDatasetNumpy.save_case(data,seg,props,str(out/k))
            np.save(out/(k+".npy"),data)
            np.save(out/(k+"_seg.npy"),seg)
        done.write_text(json.dumps(dict(train=tr,dev=split(fold)["dev"])))
        print("Prepared fold",fold,flush=True)


if __name__=="__main__": main()
