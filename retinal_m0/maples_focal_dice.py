"""Isolated Focal-BCE + original foreground Dice primitives; no execution CLI.

2026-10-24 CPU-approved candidate. Gamma 1 is fixed; gamma 0 is only the
exact A0 identity control. No class weights, skeleton term, or loss calibration.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from retinal_m0.maples_core import weighted_masked_loss


def _validate(logits, target, fov, gamma):
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma not in (0, 1):
        raise ValueError('fixed gamma=1 or identity control gamma=0 required')
    if logits.ndim != 4 or logits.shape[1] != 1 or not logits.numel():
        raise ValueError('nonempty single-foreground-channel NCHW required')
    if any(x.shape != logits.shape or x.device != logits.device for x in (target, fov)):
        raise ValueError('aligned same-device target/FOV required')
    if not all(torch.isfinite(x).all() for x in (logits, target, fov)):
        raise ValueError('nonfinite input')
    if not all(torch.all((x == 0) | (x == 1)) for x in (target, fov)):
        raise ValueError('binary target/FOV required')
    if torch.any(fov.sum((1, 2, 3)) <= 0):
        raise ValueError('empty_fov: no resampling permitted')


def focal_dice_terms(logits, target, fov, *, gamma=1):
    """FP32 per-crop terms; the focal modulation remains in the autograd graph.

    sigmoid(-signed_logit) = 1-p_t without subtractive cancellation. Empty
    GT is an ordinary negative crop, never dropped or set to zero.
    """
    _validate(logits, target, fov, gamma)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z, y, f = logits.float(), target.float(), fov.float()
        axes = (1, 2, 3)
        count = f.sum(axes)
        bce_map = F.binary_cross_entropy_with_logits(z, y, reduction='none')
        bce = (bce_map * f).sum(axes) / count
        if gamma == 0:
            focal_map = bce_map
        else:
            focal_map = torch.sigmoid(-(2. * y - 1.) * z) * bce_map
        focal = (focal_map * f).sum(axes) / count
        p = z.sigmoid()
        dice = 1. - (2. * (p * y * f).sum(axes) + 1e-6) / (
            (p * f).sum(axes) + (y * f).sum(axes) + 1e-6)
        total = (focal + dice).mean()
        if not torch.isfinite(total):
            raise ValueError('nonfinite focal-Dice loss')
        return {'loss': total, 'per_crop_focal': focal, 'per_crop_bce': bce,
                'per_crop_dice': dice, 'FOV_count': count,
                'GT_count': (y * f).sum(axes)}


def focal_dice_loss(logits, target, fov, *, gamma=1):
    """Exact original function for gamma-zero; fixed candidate for gamma-one."""
    _validate(logits, target, fov, gamma)
    if gamma == 0:
        with torch.autocast(device_type=logits.device.type, enabled=False):
            return weighted_masked_loss(logits, target, fov, fov, epsilon=1e-6)
    return focal_dice_terms(logits, target, fov, gamma=1)['loss']
