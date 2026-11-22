"""Compare archived fold-0 models, from decoded RGB to a CPU binary mask."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from prepare import OLD, ROOT, split, stage
import paired
import two_scale
from evaluate import probabilities
from predict_image import PlansManager, nnUNetPredictor, nnUNetTrainer


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    m = paired.UNet()
    m.load_state_dict(torch.load(OLD/"rgb_only/local_2027/weights.pt", map_location="cpu", weights_only=True))
    m.eval().cuda()
    plans = PlansManager(str(ROOT/"preprocessed"/stage(0)/"nnUNetPlans.json"))
    cfg = plans.get_configuration("2d")
    meta = read(ROOT/"raw"/stage(0)/"dataset.json")
    n = nnUNetTrainer.build_network_architecture(cfg.network_arch_class_name, cfg.network_arch_init_kwargs,
        cfg.network_arch_init_kwargs_req_import, 3, 2, enable_deep_supervision=True)
    n.load_state_dict(torch.load(ROOT/"models_teacher/fold0/2027/correction/weights.pt", map_location="cpu", weights_only=True))
    n.decoder.deep_supervision = False
    pred = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                           device=torch.device("cuda"), allow_tqdm=False)
    pred.manual_initialization(n.eval().cuda(), plans, cfg, None, meta, "nnUNetTrainer", None)
    torch.backends.cudnn.benchmark = False
    thresholds = read(ROOT/"eval_teacher/fold0/2027/correction/calibration.json")["m0_thresholds"]
    mt = next(r["threshold"] for r in read(OLD/"rgb_only/calibrated/thresholds.json")
              if r["method"] == "local" and r["seed"] == 2027)
    md = read(OLD/"two_scale/local_2027/threshold.json")["threshold"]

    def predict(model, view, rgb):
        if model == "nnUNet":
            prob = probabilities(pred, rgb, native_only=view == "native")[view]
            threshold = thresholds[view]
        elif view == "native":
            x = paired.normalize(torch.from_numpy(rgb).permute(2, 0, 1))
            prob = paired.tiled_probability(m, x, patch=512, stride=256, batch_size=4)[0].numpy()
            threshold = mt
        else:
            prob = two_scale.predict(m, {"rgb": rgb})[1]
            threshold = md
        return (prob >= threshold).astype(np.uint8) * 255

    images = {}
    for name in split(0)["dev"]:
        path = next((ROOT/"raw").glob("*/imagesTr/" + name + "_0000.png"))
        with Image.open(path) as im:
            images[name] = np.array(im.convert("RGB"))
    nets = {"UNet": m, "nnUNet": n}
    plans_to_time = [(model, view) for model in nets for view in ("native", "dual")]
    counts = {}
    for model, view in plans_to_time:
        result = {"tiles": 0, "conv_macs": 0}
        def tiles(module, args, output):
            result["tiles"] += args[0].shape[0]
        def macs(module, args, output):
            size = args[0] if isinstance(module, torch.nn.ConvTranspose2d) else output
            channels = module.out_channels if isinstance(module, torch.nn.ConvTranspose2d) else module.in_channels
            result["conv_macs"] += int(size.numel()*channels*np.prod(module.kernel_size)//module.groups)
        hooks = [nets[model].register_forward_hook(tiles)]
        hooks.extend(module.register_forward_hook(macs) for module in nets[model].modules()
                     if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)))
        try:
            predict(model, view, next(iter(images.values())))
        finally:
            for hook in hooks:
                hook.remove()
        counts[model+"_"+view] = result
    # Second warm pass without hooks, then interleave all four paths per image.
    for model, view in plans_to_time:
        predict(model, view, next(iter(images.values())))
    rng = random.Random(2027)
    records = []
    for repeat in range(2):
        names = list(images)
        rng.shuffle(names)
        for name in names:
            order = plans_to_time.copy()
            rng.shuffle(order)
            for model, view in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                start = time.perf_counter()
                mask = predict(model, view, images[name])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start)*1000
                records.append(dict(model=model, view=view, image=name, repeat=repeat,
                                    ms=elapsed, extra_peak_bytes=torch.cuda.max_memory_allocated()-before,
                                    positive_pixels=int(np.count_nonzero(mask))))
    summary = {}
    for model, view in plans_to_time:
        subset = [r for r in records if (r["model"], r["view"]) == (model, view)]
        assert len(subset) == len(images)*2
        for name in images:
            pixels = [r["positive_pixels"] for r in subset if r["image"] == name]
            assert min(pixels)>0 and min(pixels)==max(pixels)
        times = [r["ms"] for r in subset]
        summary[model+"_"+view] = dict(median_ms=float(np.median(times)), mean_ms=float(np.mean(times)),
            p90_ms=float(np.percentile(times,90)), max_extra_peak_MiB=max(r["extra_peak_bytes"] for r in subset)/2**20,
            stored_parameters=sum(p.numel() for p in nets[model].parameters()), **counts[model+"_"+view])
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, fold=0, seed=2027, repeats=2,
                  unit="decoded 1024px RGB to CPU uint8 mask, warm weights, one photo at a time",
                  structural_counts_image=next(iter(images)), summary=summary, records=records)
    (ROOT/"inference_budget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
