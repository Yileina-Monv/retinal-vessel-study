"""Bounded MAPLES 110-train interface verification through the M0 CLI."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from retinal_data.maples import MaplesTrainPatches
from retinal_m0.model import UNet
from retinal_m0.state import atomic_save, rng_state, state_digest
from retinal_release.data import masked_loss


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(config_path, output, config, args):
    if args.resume or args.stop_at is not None:
        raise ValueError("The 10-step MAPLES interface check is a fresh bounded run")
    if config.get("scope") != "interface_validation_110_train_only" or config.get("max_optimizer_updates") != 10:
        raise ValueError("Only the approved 10-step interface scope is supported")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("The approved physical GPU is 0; set CUDA_VISIBLE_DEVICES=0")
    if not output.is_relative_to(Path("/data/newnas_1/WL/retina_project/code/retinal-vessel-study/runs")):
        raise ValueError("Run output must be within project runs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Fresh run requires a new empty output directory")
    source = Path("/data/newnas_1/WL/retina_project/checkpoints/m0_10000_20261008/last.pt")
    lock_path = Path("/data/newnas_1/WL/retina_project/experiments/task2_runtime/gpu0_smoke.lock")
    started = time.perf_counter()
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()[0]
        values = [part.strip() for part in gpu.split(",")]
        if values[0] != "0" or int(values[2]) >= 512 or int(values[3]) >= 10:
            raise RuntimeError("Physical GPU0 is not available")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                             "--format=csv,noheader,nounits"], text=True, timeout=10)
        uuid = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines()[0].strip()
        if any(line.split(",")[0].strip() == uuid for line in processes.splitlines() if line.strip()):
            raise RuntimeError("Physical GPU0 already has a compute process")
        output.mkdir(parents=True, exist_ok=True)
        binding = {"scope": config["scope"], "config_sha256": digest(config_path),
                   "source_checkpoint_sha256": digest(source),
                   "manifest_sha256": digest(Path(config["data_root"]) / "data_manifest.csv"),
                   "adapter_sha256": digest(Path(__file__).parents[1] / "retinal_data/maples.py"),
                   "runner_sha256": digest(__file__), "gpu_uuid": uuid}
        save_json(output / "config.json", config)
        save_json(output / "binding.json", binding)
        status = {"status": "running", "target_steps": 10, "completed_steps": 0}
        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            dataset = MaplesTrainPatches(config)
            model_config = json.loads((Path(__file__).parents[1] / "configs/development_v1.json").read_text())
            model = UNet(model_config["widths"], model_config["group_norm_groups"])
            saved = torch.load(source, map_location="cpu", weights_only=True)
            if saved["step"] != 10000:
                raise ValueError("Source is not the specified M0 10k checkpoint")
            model.load_state_dict(saved["model"], strict=True)
            del saved
            initial_model_digest = state_digest(model.state_dict())
            model.cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"], foreach=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config["learning_rate"])
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            draws = []
            with (output / "train.jsonl").open("x", encoding="utf-8") as log:
                for step in range(1, 11):
                    if time.perf_counter() - started > config["internal_safe_stop_seconds"]:
                        raise TimeoutError("Bounded interface run exceeded safe window")
                    item = dataset[step - 1]
                    x, y, fov = [item[k].unsqueeze(0).cuda() for k in ("image", "mask", "fov")]
                    if x.shape != (1, 3, 512, 512) or y.shape != fov.shape != (1, 1, 512, 512):
                        raise ValueError("Unexpected MAPLES batch shape")
                    before = model.head.weight.detach().clone()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = masked_loss(logits, y, fov, epsilon=1e-6)
                    if logits.shape != y.shape or not torch.isfinite(loss):
                        raise ValueError("Invalid forward or loss")
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                               config["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    if torch.equal(before, model.head.weight):
                        raise RuntimeError("Parameters did not update")
                    row = {"step": step, "key": item["key"], "geometry": item["geometry"].tolist(),
                           "image_shape": list(x.shape), "mask_shape": list(y.shape),
                           "fov_shape": list(fov.shape), "mask_values": y.unique().tolist(),
                           "fov_values": fov.unique().tolist(), "loss": float(loss.detach()),
                           "gradient_norm": float(grad_norm)}
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    draws.append(row)
                    status["completed_steps"] = step
                    print(f"MAPLES interface step {step}/10 loss={row['loss']:.4f}", flush=True)
            torch.cuda.synchronize()
            payload = {"schema_version": 1, "step": 10, "next_global_draw": 10,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                       "rng": rng_state(), "config": config, "binding": binding}
            atomic_save(payload, output / "last.pt")
            reloaded = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            if set(reloaded) != set(payload) or reloaded["step"] != 10 or reloaded["next_global_draw"] != 10:
                raise RuntimeError("Saved checkpoint format differs")
            if state_digest(reloaded["model"]) != state_digest(model.state_dict()) or initial_model_digest == state_digest(model.state_dict()):
                raise RuntimeError("Saved model state invalid")
            status.update(status="passed", distinct_train_samples=len({row["key"] for row in draws}),
                          checkpoint_sha256=digest(output / "last.pt"), checkpoint_schema_version=1,
                          source_m0_step=10000, source_m0_sha256=binding["source_checkpoint_sha256"],
                          input_shape=[1, 3, 512, 512], output_shape=[1, 1, 512, 512],
                          formal_evaluation=False, fov_status="engineering_candidate",
                          elapsed_seconds=time.perf_counter() - started)
        except Exception:
            status.update(status="failed", error=traceback.format_exc(),
                          elapsed_seconds=time.perf_counter() - started)
            raise
        finally:
            save_json(output / "summary.json", status)
