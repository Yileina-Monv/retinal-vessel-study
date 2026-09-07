"""Bounded environment acceptance using synthetic data only; not a training benchmark."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".runtime" / "deployment_4070" / "compatibility_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib-cache"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        # A top-level class is required by Windows spawn workers.
        return torch.full((3, 64, 64), float(index)), index


class EnvironmentProbeUNet(nn.Module):
    """Representative operators and tensor sizes, not the project's frozen model."""
    def __init__(self):
        super().__init__()
        widths = (32, 64, 128, 256, 512)
        self.encoders = nn.ModuleList()
        previous = 3
        for width in widths:
            self.encoders.append(self.block(previous, width))
            previous = width
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.ups.append(nn.ConvTranspose2d(previous, width, 2, stride=2))
            self.decoders.append(self.block(2 * width, width))
            previous = width
        self.head = nn.Conv2d(widths[0], 1, 1)

    @staticmethod
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(),
        )

    def forward(self, x):
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            if index < len(self.encoders) - 1:
                skips.append(x)
                x = nn.functional.max_pool2d(x, 2)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat((up(x), skip), dim=1))
        return self.head(x)


def check_cpu_stack():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import psutil
    import requests
    import scipy.ndimage as ndi
    from skimage.morphology import skeletonize
    from sklearn.metrics import roc_auc_score
    from PIL import Image
    import yaml
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter

    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[15:113, 60:67] = 1
    mask[60:67, 15:113] = 1
    skeleton = skeletonize(mask.astype(bool))
    radius = ndi.distance_transform_edt(mask)
    assert skeleton.any() and np.all(radius[skeleton] > 0)
    assert not skeletonize(np.zeros_like(mask, dtype=bool)).any()
    rgb = np.repeat((mask * 255)[..., None], 3, axis=2)
    image_path = ARTIFACTS / "合成图像.png"
    Image.fromarray(rgb).save(image_path)
    assert np.array_equal(np.asarray(Image.open(image_path)), rgb)
    decoded = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), rgb)
    tensor = torch.from_numpy(rgb.copy())
    assert np.array_equal(tensor.numpy(), rgb)

    table = pd.DataFrame({"sample": ["synthetic_1"], "quality": [1]})
    xlsx_path = ARTIFACTS / "合成质量表.xlsx"
    table.to_excel(xlsx_path, index=False, engine="openpyxl")
    assert pd.read_excel(xlsx_path, engine="openpyxl").equals(table)
    assert yaml.safe_load("seed: 7\n")["seed"] == 7
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert list(tqdm(range(2), disable=True)) == [0, 1]
    assert requests.Request("GET", "https://example.com/").prepare().method == "GET"
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(skeleton, cmap="gray")
    ax.set_axis_off()
    fig.savefig(ARTIFACTS / "synthetic-skeleton.png")
    plt.close(fig)
    with SummaryWriter(str(ARTIFACTS / "tensorboard")) as writer:
        writer.add_scalar("environment/synthetic_check", 1, 0)
    return {"passed": True, "skeleton_pixels": int(skeleton.sum()),
            "memory_total_gib": round(psutil.virtual_memory().total / 2**30, 2),
            "memory_available_gib": round(psutil.virtual_memory().available / 2**30, 2),
            "unicode_png_and_xlsx_roundtrip": True}


def check_windows_loader():
    loader = DataLoader(SyntheticDataset(), batch_size=2, num_workers=2,
                        multiprocessing_context="spawn", pin_memory=True, timeout=45)
    seen = []
    for images, indices in loader:
        assert images.is_pinned()
        on_gpu = images.to("cuda", non_blocking=True)
        assert on_gpu.shape == (2, 3, 64, 64)
        seen.extend(indices.tolist())
    torch.cuda.synchronize()
    assert seen == [0, 1, 2, 3]
    return {"passed": True, "workers": 2, "samples": len(seen), "pin_memory": True}


def check_gpu():
    from torchvision.ops import nms
    assert torch.cuda.is_available(), "CUDA is unavailable"
    props = torch.cuda.get_device_properties(0)
    arch = f"sm_{props.major}{props.minor}"
    # NVIDIA permits same-major cubins with lower/equal minor on newer GPUs.
    compatible = [int(a[3:]) for a in torch.cuda.get_arch_list() if a.startswith("sm_") and a[3:].isdigit()]
    assert any(a // 10 == props.major and a % 10 <= props.minor for a in compatible), f"No compatible cubin: {arch}"
    device = torch.device("cuda")
    torch.manual_seed(20260904)
    torch.cuda.manual_seed_all(20260904)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    matrix = torch.randn(128, 128)
    assert torch.allclose((matrix.to(device) @ matrix.to(device).T).cpu(),
                          matrix @ matrix.T, atol=1e-3, rtol=1e-3)
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert nms(boxes, scores, 0.5).tolist() == [0]

    torch.cuda.reset_peak_memory_stats()
    model = EnvironmentProbeUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    images = torch.rand(1, 3, 512, 512, device=device)
    labels = (torch.rand(1, 1, 512, 512, device=device) > 0.9).float()
    initial_weight = next(model.parameters()).detach().clone()
    losses, step_seconds = [], []
    for _ in range(3):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        assert logits.shape == labels.shape and torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_start)
        losses.append(float(loss.detach()))
    assert not torch.equal(initial_weight, next(model.parameters()).detach())
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(images)
    assert torch.isfinite(output).all().item()
    checkpoint = ARTIFACTS / "synthetic-environment-checkpoint.pt"
    torch.save({"model": model.state_dict(), "synthetic_only": True}, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert restored["synthetic_only"] is True
    assert all(torch.equal(value.cpu(), restored["model"][name])
               for name, value in model.state_dict().items())
    return {
        "passed": True, "device": props.name, "compute_capability": arch,
        "total_memory_gib": round(props.total_memory / 2**30, 3),
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "torchvision_cuda_nms": True, "amp_fp16_backward_adamw": True,
        "amp_bfloat16_inference": True, "checkpoint_roundtrip": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape": list(images.shape), "optimizer_updates": 3,
        "synthetic_losses": losses, "synthetic_step_seconds": step_seconds,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        "timing_scope": "Three synthetic acceptance steps including startup/checks; not a training benchmark",
    }


def main():
    started = time.perf_counter()
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "scope": "environment_only_synthetic_data_compatible_cubin_check", "status": "running",
              "native_sm_89_in_build": "sm_89" in torch.cuda.get_arch_list(),
              "original_verifier_sha256": "527e52d6e7cd797b483ceb58c1b1f78691f2f89cc6b5a7fcc8842ea56a55251e",
              "compatibility_source": "https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html",
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "checks": {}}
    report = ROOT / ".runtime" / "deployment_4070" / "environment_compatibility.json"
    try:
        names = ("torch", "torchvision", "numpy", "Pillow", "scipy", "scikit-image",
                 "scikit-learn", "opencv-python-headless", "pandas", "openpyxl",
                 "matplotlib", "PyYAML", "tqdm", "tensorboard", "psutil", "requests", "pip")
        result["packages"] = {name: importlib.metadata.version(name) for name in names}
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"], text=True).strip()
        for name, check in (("cpu_stack", check_cpu_stack), ("gpu", check_gpu),
                            ("windows_dataloader", check_windows_loader)):
            print(f"Checking {name}...", flush=True)
            result["checks"][name] = check()
            print(f"PASS {name}", flush=True)
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {report}", flush=True)


if __name__ == "__main__":
    main()
