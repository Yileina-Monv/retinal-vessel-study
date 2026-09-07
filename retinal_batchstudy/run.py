from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

import numpy as np
import torch

from retinal_data.fives import EpochDrawSampler, trial_dataset
from retinal_data.prepare import ROOT, sha256
from retinal_m0.inference import tiled_probability
from retinal_m0.model import UNet
from retinal_m0.objectives import pixel_loss, binary_metrics
from retinal_m0.state import atomic_save, rng_state, restore_rng, state_digest

CONFIG = ROOT / "configs/batch_study_v1.json"


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def setup(config, seed):
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binding():
    paths = [CONFIG, ROOT / "configs/data_trial_v1.json", ROOT / "data_preparation/selection_summary.json"]
    paths += [ROOT / "data_preparation/manifests" / f"{name}.csv" for name in ("smoke_train", "smoke_val", "overfit")]
    for module in ("retinal_data", "retinal_m0", "retinal_batchstudy"):
        paths += sorted((ROOT / module).glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def hardware():
    import PIL
    import scipy
    return {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "pillow": PIL.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()}


def schedule_lr(config, consumed):
    fraction = consumed / config["training_draws"]
    return config["minimum_learning_rate"] + 0.5 * (config["learning_rate"] - config["minimum_learning_rate"]) * (1 + math.cos(math.pi * fraction))


def sequence(dataset, count):
    order = []
    epoch = 0
    while len(order) < count:
        order.extend(list(EpochDrawSampler(dataset, epoch=epoch, shuffle=True)))
        epoch += 1
    return order[:count]


def get_batch(dataset, plan, start, count):
    samples = [dataset[index] for index in plan[start:start + count]]
    inputs = torch.stack([s["image"] for s in samples])
    targets = torch.stack([s["mask"] for s in samples])
    # Canonical sample-wise hashing is independent of how a batch is partitioned.
    signatures = []
    for item in samples:
        digest = hashlib.sha256()
        digest.update(item["image"].numpy().tobytes())
        digest.update(item["mask"].numpy().tobytes())
        signatures.append({"key": item["key"], "epoch": item["epoch"], "draw": item["draw"],
                           "geometry": item["geometry"].tolist(), "tensor_sha256": digest.hexdigest()})
    return inputs, targets, signatures


def initialized(config, seed, destination):
    setup(config, seed)
    model = UNet(config["widths"], config["group_norm_groups"])
    path = destination / f"initial_seed_{seed}.pt"
    signature = state_digest(model.state_dict())
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["state_digest"] != signature:
            raise ValueError("Initial weights differ; do not silently replace the portable reference")
        model.load_state_dict(saved["model"])
    else:
        atomic_save({"seed": seed, "model": model.state_dict(), "state_digest": signature}, path)
    return model.cuda(), signature


@torch.inference_mode()
def evaluate(model, validation, config):
    model.eval()
    rows = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(len(validation)):
        item = validation[index]
        probability, _ = tiled_probability(model, item["image"], patch=config["patch"],
                                          stride=config["inference_stride"], batch_size=config["inference_batch"])
        rows.append({"key": item["key"], "disease": item["disease"],
                     **binary_metrics(probability, item["mask"][0], config["threshold"])})
    torch.cuda.synchronize()
    return {"macro_" + metric: float(np.mean([r[metric] for r in rows if r[metric] is not None]))
            for metric in ("dice", "precision", "recall")} | {"rows": rows, "seconds": time.perf_counter() - started}


def flat_grad(model):
    return torch.cat([p.grad.detach().float().flatten().cpu() for p in model.parameters()])


def flat_parameters(model):
    return torch.cat([p.detach().float().flatten().cpu() for p in model.parameters()])


def differences(actual, reference):
    delta = actual.double() - reference.double()
    return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
            "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))}


def numerical_probe(config, train, root):
    destination = root / "numerical"
    if (destination / "summary.json").exists():
        return
    destination.mkdir(exist_ok=True)
    train.seed = config["seeds"][0]
    plan = sequence(train, 4)
    inputs, targets, signatures = get_batch(train, plan, 0, 4)
    atomic_save({"inputs": inputs, "targets": targets, "signatures": signatures}, destination / "fixed_inputs.pt")
    result = {"input_signatures": signatures, "comparisons": [], "claim": "Numerical characterization; no posthoc floating-point pass threshold"}
    for amp in (False, True):
        reference = None
        for microbatch in (1, 2, 4):
            model, init_digest = initialized(config, config["seeds"][0], root)
            before = flat_parameters(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False)
            scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=config["amp_initial_scale"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, outputs = [], []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for i in range(0, 4, microbatch):
                x, y = inputs[i:i+microbatch].cuda(), targets[i:i+microbatch].cuda()
                with torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext():
                    logits = model(x)
                    loss, _ = pixel_loss(logits, y, config["dice_epsilon"])
                scaler.scale(loss * microbatch / 4).backward()
                losses.append(float(loss.detach()) * microbatch / 4)
                outputs.append(logits.detach().float().cpu())
            scaler.unscale_(optimizer)
            gradient = flat_grad(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise RuntimeError("Numerical probe AMP step skipped")
            torch.cuda.synchronize()
            vectors = {"outputs": torch.cat(outputs), "gradient": gradient, "parameter_update": flat_parameters(model) - before}
            name = f"{'amp' if amp else 'fp32'}_micro{microbatch}"
            atomic_save(vectors, destination / f"{name}.pt")
            entry = {"name": name, "loss": sum(losses), "gradient_norm": float(norm), "initial_digest": init_digest,
                     "seconds": time.perf_counter() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30}
            if reference is None:
                reference = vectors
            entry["vs_micro1_same_precision"] = {k: differences(v, reference[k]) for k, v in vectors.items()}
            result["comparisons"].append(entry)
            del model, optimizer, scaler, gradient, vectors, outputs
            gc.collect()
            torch.cuda.empty_cache()
    dump(destination / "summary.json", result)


def train_arm(config, arm, seed, train, validation, root, fingerprints, deadline, stop_after=None):
    destination = root / f"seed_{seed}" / arm["id"]
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "result.json").exists():
        return True
    model, initial_digest = initialized(config, seed, root)
    train.seed = seed
    plan = sequence(train, config["training_draws"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False)
    scaler = torch.amp.GradScaler("cuda", init_scale=config["amp_initial_scale"])
    consumed, step, history, signatures = 0, 0, [], []
    updates_seconds, validation_seconds, clipped = 0., 0., 0
    checkpoint_path = destination / "last.pt"
    segments = []
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["binding"] != fingerprints or saved["arm"] != arm or saved["seed"] != seed:
            raise ValueError("Checkpoint binding/config mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        consumed, step = saved["consumed"], saved["step"]
        history, signatures = saved["history"], saved["signatures"]
        updates_seconds, validation_seconds, clipped = saved["updates_seconds"], saved["validation_seconds"], saved["clipped"]
        segments = saved["segments"]
        restore_rng(saved["rng"])
        del saved
    elif (root / f"initial_eval_{seed}.json").exists():
        validation_seconds = read(root / f"initial_eval_{seed}.json")["seconds"]
    else:
        initial_eval = evaluate(model, validation, config)
        dump(root / f"initial_eval_{seed}.json", initial_eval)
        validation_seconds = initial_eval["seconds"]
    microbatch, accumulation = arm["microbatch"], arm["accumulation"]
    effective = microbatch * accumulation
    if config["training_draws"] % effective:
        raise ValueError("Incomplete effective batch")
    torch.cuda.reset_peak_memory_stats()
    segment_started = time.perf_counter()
    segment_start_draw = consumed
    segment_peak = 0.

    def save():
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "rng": rng_state(), "consumed": consumed, "step": step, "history": history, "signatures": signatures,
                     "updates_seconds": updates_seconds, "validation_seconds": validation_seconds, "clipped": clipped,
                     "segments": segments, "binding": fingerprints, "config": config, "arm": arm, "seed": seed,
                     "initial_digest": initial_digest, "lr_next": schedule_lr(config, consumed)}, checkpoint_path)

    try:
        while consumed < config["training_draws"]:
            if time.perf_counter() > deadline or (stop_after is not None and consumed >= stop_after):
                segments.append({"start_draw": segment_start_draw, "end_draw": consumed, "gpu": torch.cuda.get_device_name(),
                                 "seconds": time.perf_counter()-segment_started, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                                 "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30})
                save()
                return False
            torch.cuda.synchronize()
            started = time.perf_counter()
            inputs, targets, records = get_batch(train, plan, consumed, effective)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = schedule_lr(config, consumed)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses = []
            for offset in range(0, effective, microbatch):
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(inputs[offset:offset+microbatch].cuda())
                    loss, _ = pixel_loss(logits, targets[offset:offset+microbatch].cuda(), config["dice_epsilon"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                scaler.scale(loss / accumulation).backward()
                losses.append(float(loss.detach()))
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                raise RuntimeError("AMP skipped update; no silent retry or sample replacement")
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            updates_seconds += duration
            clipped += int(norm > config["gradient_clip_norm"])
            consumed += effective
            step += 1
            signatures.extend(records)
            history.append({"step": step, "draw": consumed, "loss": float(np.mean(losses)), "lr": lr,
                            "gradient_norm": float(norm), "seconds": duration, "amp_scale": scaler.get_scale()})
            if consumed in config["evaluation_draws"]:
                evaluation = evaluate(model, validation, config)
                validation_seconds += evaluation["seconds"]
                dump(destination / f"eval_draw_{consumed}.json", evaluation)
                print(f"seed={seed} {arm['id']} draws={consumed} Dice={evaluation['macro_dice']:.5f}", flush=True)
            if consumed % config["save_every_draws"] == 0:
                save()
                print(f"progress seed={seed} {arm['id']} draws={consumed}/{config['training_draws']}", flush=True)
        segments.append({"start_draw": segment_start_draw, "end_draw": consumed, "gpu": torch.cuda.get_device_name(),
                         "seconds": time.perf_counter()-segment_started, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                         "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30})
        save()
        result = {"seed": seed, "arm": arm, "initial_digest": initial_digest, "draws": consumed, "updates": step,
                  "sample_sequence_digest": hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest(),
                  "updates_seconds": updates_seconds, "validation_seconds": validation_seconds,
                  "draws_per_training_second": consumed / updates_seconds, "gradient_clip_fraction": clipped/step,
                  "amp_skipped_updates": 0, "segments": segments,
                  "final": read(destination / f"eval_draw_{config['training_draws']}.json")}
        dump(destination / "result.json", result)
        return True
    except Exception:
        dump(destination / "failure.json", {"consumed_before_failure": consumed, "step": step, "traceback": traceback.format_exc()})
        raise
    finally:
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--mode", choices=("probe", "train", "all"), default="all")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--stop-after-draws", type=int)
    parser.add_argument("--max-seconds", type=int)
    args = parser.parse_args()
    if not args.device_label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Use a simple device label")
    config = read(CONFIG)
    setup(config, config["seeds"][0])
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required; CPU is not a GPU substitute")
    root = ROOT / "runs/batch_study_v1" / args.device_label
    root.mkdir(parents=True, exist_ok=True)
    current_binding = binding()
    if (root / "binding.json").exists() and read(root / "binding.json") != current_binding:
        raise ValueError("Study code or inputs changed; preserve original results and use a new label")
    dump(root / "binding.json", current_binding)
    dump(root / "config.json", config)
    hardware_path = root / "hardware.json"
    this_hardware = hardware()
    if hardware_path.exists() and read(hardware_path) != this_hardware:
        raise ValueError("Hardware/runtime changed for this label; do not combine silently")
    dump(hardware_path, this_hardware)
    train, validation = trial_dataset("smoke_train"), trial_dataset("smoke_val")
    # Bounded, single-process uint8 cache: 40 paired originals, about 0.625 GiB.
    # No model/data RNG is consumed by workers; augmentation is keyed by sample index.
    for dataset in (train, validation):
        dataset.cache_size = len(dataset.rows)
        for row in dataset.rows:
            dataset._arrays(row)
    dump(root / "data_check.json", {"train_images": len(train.rows), "validation_images": len(validation.rows),
         "all_source_hashes_verified": True, "test_or_external_images_opened": 0,
         "group_overlap": len({r['group_id'] for r in train.rows} & {r['group_id'] for r in validation.rows})})
    deadline = time.perf_counter() + (args.max_seconds or config["max_suite_seconds"])
    if args.mode in ("probe", "all"):
        numerical_probe(config, train, root)
    if args.mode == "probe":
        return
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Seed not in frozen protocol")
    plan = []
    for seed in config["seeds"]:
        indices = np.random.default_rng(config["run_order_seed"] + seed).permutation(len(config["arms"]))
        plan += [(seed, config["arms"][int(i)]) for i in indices]
    dump(root / "run_order.json", [{"seed": seed, "arm": arm["id"]} for seed, arm in plan])
    for seed, arm in plan:
        if seed not in seeds or (args.arm and arm["id"] != args.arm):
            continue
        complete = train_arm(config, arm, seed, train, validation, root, current_binding, deadline, args.stop_after_draws)
        if not complete:
            print("Paused at a full optimizer boundary; rerun the same command to continue.", flush=True)
            return
    print("Requested study matrix completed.", flush=True)


if __name__ == "__main__":
    main()
