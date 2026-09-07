"""One bounded M0 pilot, split into two processes to validate checkpoint resume."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from retinal_data.fives import EpochDrawSampler, trial_dataset
from retinal_data.prepare import ROOT, sha256
from .inference import tiled_probability
from .model import UNet
from .objectives import binary_metrics, pixel_loss
from .state import atomic_save, restore_rng, rng_state, state_digest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def bind_inputs(config_path):
    paths = [config_path, ROOT / "configs/data_trial_v1.json", ROOT / "data_preparation/selection_summary.json"]
    paths += sorted((ROOT / "data_preparation/manifests").glob("*.csv"))
    paths += sorted((ROOT / "retinal_m0").glob("*.py")) + sorted((ROOT / "retinal_data").glob("*.py"))
    paths += [ROOT / "environment/installed-packages.json"]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


class BatchStream:
    def __init__(self, dataset, config, start_step):
        self.dataset, self.config = dataset, config
        self.position = start_step * config["microbatch"] * config["accumulation"]
        self.iterator = self.loader = None
        if len(dataset) % (config["microbatch"] * config["accumulation"]):
            raise ValueError("The trial dataset must contain complete effective batches")

    def next(self):
        if self.iterator is None:
            self.start_iterator()
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.start_iterator()
            batch = next(self.iterator)
        self.position += self.config["microbatch"]
        return batch

    def start_iterator(self):
        epoch, offset = divmod(self.position, len(self.dataset))
        indices = list(EpochDrawSampler(self.dataset, epoch=epoch, shuffle=True))[offset:]
        workers = self.config["workers"]
        options = {"multiprocessing_context": "spawn", "prefetch_factor": 2, "timeout": 60} if workers else {}
        # This separate generator prevents iterator construction from consuming model RNG state.
        generator = torch.Generator().manual_seed(self.config["model_seed"] + epoch)
        self.loader = DataLoader(self.dataset, batch_size=self.config["microbatch"], sampler=indices,
                                 num_workers=workers, pin_memory=True, generator=generator, **options)
        self.iterator = iter(self.loader)

    def close(self):
        self.iterator = self.loader = None
        gc.collect()


def update(model, optimizer, scheduler, scaler, stream, config, step):
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, bces, dices, events, samples = [], [], [], [], []
    waiting = 0.0
    lr = optimizer.param_groups[0]["lr"]
    for _ in range(config["accumulation"]):
        data_started = time.perf_counter()
        batch = stream.next()
        waiting += time.perf_counter() - data_started
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record()
        inputs = batch["image"].to("cuda", non_blocking=True)
        target = batch["mask"].to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(inputs)
            loss, parts = pixel_loss(logits, target, config["dice_epsilon"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite pixel loss")
        scaler.scale(loss / config["accumulation"]).backward()
        last.record()
        events.append((first, last))
        losses.append(float(loss.detach()))
        bces.append(float(parts["bce"].detach()))
        dices.append(float(parts["soft_dice_loss"].detach()))
        for index, key in enumerate(batch["key"]):
            samples.append({"key": key, "epoch": int(batch["epoch"][index]),
                            "draw": int(batch["draw"][index]), "geometry": batch["geometry"][index].tolist(),
                            "empty_crop": bool(batch["crop_label_empty"][index])})
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    first.record()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < old_scale:
        raise RuntimeError("AMP skipped an optimizer update; halt rather than miscount steps")
    scheduler.step()
    last.record()
    events.append((first, last))
    torch.cuda.synchronize()
    return {"step": step, "loss": statistics.mean(losses), "bce": statistics.mean(bces),
            "soft_dice_loss": statistics.mean(dices), "lr_used": lr, "gradient_norm_before_clip": float(norm),
            "data_wait_seconds": waiting, "step_wall_seconds": time.perf_counter() - started,
            "gpu_region_seconds": sum(a.elapsed_time(b) for a, b in events) / 1000,
            "samples": samples, "next_global_draw": stream.position}


def metric_mean(rows, name):
    values = [row[name] for row in rows if row[name] is not None]
    return statistics.mean(values) if values else None


@torch.inference_mode()
def evaluate_fixed(model, samples, config, step, output=None):
    model.eval()
    rows, probabilities, losses = [], [], []
    for sample in samples:
        image = sample["image"][None].to("cuda")
        mask = sample["mask"][None].to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(image)
            loss, _ = pixel_loss(logits, mask, config["dice_epsilon"])
        probability = logits.float().sigmoid().cpu()[0, 0]
        rows.append({"key": sample["key"], **binary_metrics(probability, sample["mask"][0], config["threshold"])})
        probabilities.append(probability.numpy())
        losses.append(float(loss))
    if output:
        np.savez_compressed(output, probabilities=np.stack(probabilities))
    return {"step": step, "scope": "four_fixed_training_patches", "loss": statistics.mean(losses),
            "dice": metric_mean(rows, "dice"), "rows": rows}


@torch.inference_mode()
def evaluate_full(model, dataset, config, output, phase):
    output = Path(output)
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=True)
    model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        probability, tiles = tiled_probability(model, item["image"], patch=config["inference_patch"],
            stride=config["inference_stride"], batch_size=config["inference_batch"])
        torch.cuda.synchronize()
        duration = time.perf_counter() - inference_started
        rows.append({"key": item["key"], "tiles": tiles, "inference_seconds": duration,
                     **binary_metrics(probability, item["mask"][0], config["threshold"])})
        if phase == "final":
            name = item["key"].replace("/", "__").removesuffix(".png")
            np.save(predictions / f"{name}.npy", probability.numpy())
            Image.fromarray(((probability.numpy() >= config["threshold"]) * 255).astype(np.uint8)).save(predictions / f"{name}.png")
        print(f"{phase} development image {index+1}/{len(dataset)}: {item['key']} Dice={rows[-1]['dice']:.4f}", flush=True)
    result = {"scope": "eight_development_images_from_source_train", "pixel_domain": config["pixel_domain"],
              "threshold": config["threshold"], "macro_dice": metric_mean(rows, "dice"),
              "macro_precision": metric_mean(rows, "precision"), "macro_recall": metric_mean(rows, "recall"),
              "elapsed_seconds_including_loading": time.perf_counter() - started, "rows": rows}
    write_json(output / f"validation_{phase}.json", result)
    return result


def checkpoint(model, optimizer, scheduler, scaler, config, binding, step):
    return {"schema_version": 1, "step": step, "next_global_draw": step * config["microbatch"] * config["accumulation"],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(), "config": config, "binding": binding}


def comparable_state(model, optimizer, scheduler, scaler, stream):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "next_global_draw": stream.position, "rng": rng_state()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m0_pilot_v1.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at", type=int)
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.run_dir.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Run output must be within the project's runs directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stop_at = args.stop_at or config["total_optimizer_steps"]
    if not 0 < stop_at <= config["total_optimizer_steps"]:
        raise ValueError("Invalid stop step")
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    binding = bind_inputs(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires the verified CUDA environment")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["model_seed"])
    np.random.seed(config["model_seed"])
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    model = UNet(config["widths"], config["group_norm_groups"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["total_optimizer_steps"], eta_min=config["minimum_learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=config["amp_initial_scale"])
    training = trial_dataset("smoke_train", config_path=ROOT / config["data_config"])
    validation = trial_dataset("smoke_val", config_path=ROOT / config["data_config"])
    fixed_data = trial_dataset("overfit", config_path=ROOT / config["data_config"])
    fixed = [fixed_data[(0, i * fixed_data.repeats)] for i in range(len(fixed_data.rows))]
    start_step = 0
    resume_rng = None
    segment_started = time.perf_counter()
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "resumed": args.resume,
              "status": "running", "pid": os.getpid(), "stopped_at": None}
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if saved["binding"] != binding or saved["config"] != config:
            raise ValueError("Source, code, environment inventory or configuration changed since checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_step = saved["step"]
        if saved["next_global_draw"] != start_step * config["microbatch"] * config["accumulation"]:
            raise ValueError("Checkpoint data cursor is inconsistent")
        existing_steps = [json.loads(line)["step"] for line in (output / "train.jsonl").read_text().splitlines()]
        if existing_steps != list(range(1, start_step + 1)):
            raise ValueError("Log/checkpoint mismatch; preserve evidence and reconcile before resuming")
        resume_rng = saved["rng"]
        del saved
    else:
        write_json(output / "config.json", config)
        write_json(output / "binding.json", binding)
        write_json(output / "fixed_patches.json", [{"key": item["key"], "draw": item["draw"], "epoch": item["epoch"],
                    "geometry": item["geometry"].tolist()} for item in fixed])
        hardware = {"python": os.sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "parameters": sum(p.numel() for p in model.parameters()),
                    "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], text=True).strip()}
        write_json(output / "hardware.json", hardware)
    if start_step >= stop_at:
        raise ValueError("Stop step must be after checkpoint step")
    stream = BatchStream(training, config, start_step)
    expected = json.loads((output / "resume_reference.json").read_text()) if args.resume and (output / "resume_reference.json").exists() else None
    replay_rows = []
    writer = SummaryWriter(str(output / "tensorboard"))
    if resume_rng is not None:
        restore_rng(resume_rng)
    torch.cuda.reset_peak_memory_stats()
    sampled_rss = []
    try:
        if not args.resume:
            probe = evaluate_fixed(model, fixed, config, 0, output / "fixed_before.npz")
            append_json(output / "fixed_probes.jsonl", probe)
            evaluate_full(model, validation, config, output, "initial")
        for step in range(start_step + 1, stop_at + 1):
            if time.perf_counter() - segment_started > config["max_process_wall_seconds"]:
                raise TimeoutError("Bounded pilot exceeded process time budget")
            row = update(model, optimizer, scheduler, scaler, stream, config, step)
            append_json(output / "train.jsonl", row)
            writer.add_scalar("train/pixel_loss", row["loss"], step)
            if expected and step <= expected["end_step"]:
                replay_rows.append(row)
                if step == expected["end_step"]:
                    observed = state_digest(comparable_state(model, optimizer, scheduler, scaler, stream))
                    samples_match = [r["samples"] for r in replay_rows] == [r["samples"] for r in expected["rows"]]
                    differences = [abs(a["loss"] - b["loss"]) for a, b in zip(replay_rows, expected["rows"], strict=True)]
                    comparison = {"status": "passed" if observed == expected["state_digest"] and samples_match else "failed",
                                  "compared_steps": [r["step"] for r in replay_rows], "max_loss_abs_difference": max(differences),
                                  "model_optimizer_scheduler_scaler_rng_cursor_digest_match": observed == expected["state_digest"],
                                  "samples_and_geometry_match": samples_match, "continuous_digest": expected["state_digest"], "resumed_digest": observed}
                    write_json(output / "resume_check.json", comparison)
                    if comparison["status"] != "passed":
                        raise RuntimeError("Independent-process resume differs from uninterrupted reference")
            if step % config["probe_every"] == 0 or step == stop_at:
                probe = evaluate_fixed(model, fixed, config, step,
                                       output / "fixed_after.npz" if step == config["total_optimizer_steps"] else None)
                append_json(output / "fixed_probes.jsonl", probe)
                writer.add_scalar("diagnostic/fixed_train_dice", probe["dice"], step)
                process = psutil.Process()
                rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True) if p.is_running())
                sampled_rss.append(rss)
                print(f"step {step}/{config['total_optimizer_steps']} loss={row['loss']:.4f} fixed_train_dice={probe['dice']:.4f}", flush=True)
            if step % config["checkpoint_every"] == 0 or step == stop_at:
                atomic_save(checkpoint(model, optimizer, scheduler, scaler, config, binding, step), output / "last.pt")
        if stop_at == config["resume_boundary"] and not args.resume:
            # Extra diagnostic updates are not appended to the primary training history.
            reference = [update(model, optimizer, scheduler, scaler, stream, config, s)
                         for s in range(stop_at + 1, stop_at + 1 + config["resume_comparison_steps"])]
            write_json(output / "resume_reference.json", {"end_step": reference[-1]["step"], "rows": reference,
                       "state_digest": state_digest(comparable_state(model, optimizer, scheduler, scaler, stream)),
                       "scope": "four_uninterrupted_reference_updates_after_saved_boundary"})
        stream.close()
        if stop_at == config["total_optimizer_steps"]:
            evaluate_full(model, validation, config, output, "final")
        status.update(status="completed" if stop_at == config["total_optimizer_steps"] else "paused_at_checkpoint",
                      stopped_at=stop_at, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      sampled_process_tree_rss_max_gib=max(sampled_rss, default=0) / 2**30)
    except Exception:
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        stream.close()
        writer.close()
        status["elapsed_seconds"] = time.perf_counter() - segment_started
        write_json(output / ("segment_resume.json" if args.resume else "segment_initial.json"), status)
        print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
