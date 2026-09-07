from contextlib import nullcontext

import torch


def starts(length, patch, stride):
    if length < patch or not 0 < stride <= patch:
        raise ValueError("Invalid tiled inference dimensions")
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


@torch.inference_mode()
def tiled_probability(model, image, *, patch=512, stride=256, batch_size=4, device="cuda", amp=True):
    if image.ndim != 3 or image.shape[0] != 3 or batch_size < 1:
        raise ValueError("Expected a CHW RGB image and positive inference batch size")
    height, width = image.shape[-2:]
    coords = [(top, left) for top in starts(height, patch, stride) for left in starts(width, patch, stride)]
    axis = (torch.arange(patch, dtype=torch.float32) - (patch - 1) / 2) / (patch / 8)
    weight = torch.exp(-0.5 * (axis[:, None].square() + axis[None, :].square())).clamp_min(0.001)
    total, denominator = torch.zeros((height, width)), torch.zeros((height, width))
    model.eval()
    for offset in range(0, len(coords), batch_size):
        positions = coords[offset:offset + batch_size]
        inputs = torch.stack([image[:, top:top + patch, left:left + patch] for top, left in positions]).to(device)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            logits = model(inputs)
        if logits.shape != (len(positions), 1, patch, patch):
            raise ValueError("Model returned an unexpected segmentation shape")
        probabilities = logits.float().sigmoid().cpu()[:, 0]
        for (top, left), probability in zip(positions, probabilities):
            total[top:top + patch, left:left + patch] += probability * weight
            denominator[top:top + patch, left:left + patch] += weight
    if not torch.all(denominator > 0):
        raise RuntimeError("Tiled prediction contains uncovered pixels")
    result = total / denominator
    if not torch.isfinite(result).all():
        raise RuntimeError("Nonfinite prediction")
    return result, len(coords)
