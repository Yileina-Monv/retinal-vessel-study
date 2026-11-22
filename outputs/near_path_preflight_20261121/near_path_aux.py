"""Prototype auxiliary loss for one already-aligned nnU-Net full-resolution batch."""
import torch
from torch.nn import functional as F


def near_path_aux(logits, gt, near, path_positions):
    """path_positions[b] lists the unique flattened indices in each visible GT path.
    gt and near are boolean BHW after the same geometry as logits. near excludes U.
    Use the existing segmentation loss separately. No model or optimizer is created.
    """
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Expected two-channel BCHW logits")
    if gt.shape != logits[:,0].shape or near.shape != gt.shape or len(path_positions) != len(gt):
        raise ValueError("Native image/target shape mismatch")
    if gt.dtype != torch.bool or near.dtype != torch.bool:
        raise ValueError("Expected boolean masks")
    # Float32 subtraction before BCE, including when the caller uses mixed precision.
    margin = logits[:,1].float() - logits[:,0].float()
    image_terms = []
    for b, paths in enumerate(path_positions):
        if not paths:
            continue
        flat = margin[b].flatten()
        eligible = (near[b] & gt[b]).flatten()
        losses = []
        for indices in paths:
            idx = torch.as_tensor(indices, device=flat.device)
            if idx.dtype != torch.long or idx.ndim != 1 or not idx.numel():
                raise ValueError("Expected a nonempty int64 index vector")
            if idx.min() < 0 or idx.max() >= flat.numel():
                raise ValueError("Invalid target index")
            idx = torch.unique(idx)
            if not eligible[idx].all():
                raise ValueError("Path support must be inside GT and the valid near region")
            losses.append(F.softplus(-flat[idx]).mean())
        bg = near[b] & ~gt[b]
        positive = torch.stack(losses).mean()
        # Keep a valid positive-only term if the crop has no reference background.
        term = (positive + F.softplus(margin[b][bg]).mean()) / 2 if bg.any() else positive
        image_terms.append(term)
    return torch.stack(image_terms).mean() if image_terms else margin.sum() * 0
