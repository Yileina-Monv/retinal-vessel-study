"""Pinned author asymmetric binary UFL, with the existing crop/FOV reduction.

No model/device/experiment side effects. The single sigmoid head is retained.
Author commit 411c9c5ce43b2ef847f0903d1b841512ad8d2eee; docstring-only
reference repair is recorded separately. No ordinary Dice is added to UFL.
"""
from __future__ import annotations

import torch

PARAMETERS = {'lambda_code': 0.5, 'delta': 0.6, 'gamma': 0.5, 'epsilon': 1e-7}


def ufl_terms(logits, target, fov):
    if logits.ndim != 4 or logits.shape[1] != 1 or not logits.numel():
        raise ValueError('nonempty [B,1,H,W] logits required')
    if not logits.is_floating_point():
        raise ValueError('floating logits required')
    if any(t.shape != logits.shape or t.device != logits.device for t in [target, fov]):
        raise ValueError('aligned same-device logits/target/FOV required')
    if not all(torch.isfinite(t).all() for t in [logits, target, fov]):
        raise ValueError('nonfinite loss input')
    if not all(torch.all((t == 0) | (t == 1)) for t in [target, fov]):
        raise ValueError('binary GT and FOV required')
    if torch.any(fov.sum((1, 2, 3)) <= 0):
        raise ValueError('empty_fov: no dropping or resampling permitted')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        # Float64 is preserved only when supplied by CPU numerical validation.
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        z, y, f = [t.to(dtype=dtype) for t in [logits, target, fov]]
        eps, delta, gamma = 1e-7, 0.6, 0.5
        p = z.sigmoid()
        q = torch.cat([1 - p, p], dim=1).clamp(eps, 1 - eps)
        t = torch.cat([1 - y, y], dim=1)
        n = f.sum((1, 2, 3))
        tp = (f * t * q).sum((2, 3))
        fn = (f * t * (1 - q)).sum((2, 3))
        fp = (f * (1 - t) * q).sum((2, 3))
        error = delta * fn + (1 - delta) * fp
        denominator = tp + error + eps
        ti = (tp + eps) / denominator
        # Exactly 1-TI in real arithmetic; avoids float32 cancellation and
        # the author's r * r**(-gamma) 0*inf evaluation at perfect overlap.
        residual = error / denominator
        bg_ce = -(1 - delta) * (1 - y[:, 0]) * (1 - q[:, 0]).pow(gamma) * q[:, 0].log()
        fg_ce = -delta * y[:, 0] * q[:, 1].log()
        afl = ((bg_ce + fg_ce) * f[:, 0]).sum((1, 2)) / n
        aft = (residual[:, 0] + residual[:, 1].pow(1 - gamma)) / 2
        total = (0.5 * afl + 0.5 * aft).mean()
        if not torch.isfinite(total):
            raise ValueError('nonfinite UFL')
        return {'loss': total, 'per_crop_afl': afl, 'per_crop_aft': aft,
                'per_crop_weighted_afl': .5 * afl, 'per_crop_weighted_aft': .5 * aft,
                'FOV_count': n, 'GT_count': (y * f).sum((1, 2, 3)),
                'TP': tp, 'FN': fn, 'FP': fp, 'TI': ti, 'TI_residual': residual}


def ufl_loss(logits, target, fov):
    return ufl_terms(logits, target, fov)['loss']
