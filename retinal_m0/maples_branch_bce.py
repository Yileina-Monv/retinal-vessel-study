"""Candidate B1/B2 CPU primitives, NOT a training or GPU launch entry.

The target definition remains subject to genuine human review. Reuse the bound
T0 arithmetic; never import the old preparation CLI (it mutates CUDA env vars).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from retinal_m0.maples_core import weighted_masked_loss


def _validate(logits, vessel, fov, selected):
    if logits.ndim != 4 or logits.shape[1] != 1 or not logits.numel():
        raise ValueError("nonempty single-logit NCHW required")
    if any(t.shape != logits.shape or t.device != logits.device
           for t in (vessel, fov, selected)):
        raise ValueError("aligned, same-device NCHW tensors required")
    if not all(torch.isfinite(t).all() for t in (logits, vessel, fov, selected)):
        raise ValueError("nonfinite input")
    if any(not torch.all((t == 0) | (t == 1)) for t in (vessel, fov, selected)):
        raise ValueError("binary GT/FOV/target required")
    if torch.any(selected.bool() & ~(vessel.bool() & fov.bool())):
        raise ValueError("auxiliary target must be a subset of GT and FOV")
    if torch.any(fov.sum((1, 2, 3)) <= 0):
        raise ValueError("empty_fov: no resampling")


def normalized_positive_bce(logits, vessel, fov, selected):
    """FP32 per-crop positive BCE; empty crops remain in the batch mean.

    clamp_min(1) only makes empty divisions safe: nonempty counts are unchanged.
    Empty targets have differentiable exact-zero auxiliary value and gradient.
    """
    _validate(logits, vessel, fov, selected)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z, s = logits.float(), selected.float()
        counts = s.sum((1, 2, 3))
        sums = (F.softplus(-z) * s).sum((1, 2, 3))
        per_crop = sums / counts.clamp_min(1)
        return per_crop.mean()


def branch_bce_loss(logits, vessel, fov, selected, lam=0.1):
    """A0 BCE+Dice plus lambda auxiliary BCE; lambda-zero is an exact bypass."""
    if type(lam) not in (int, float) or not math.isfinite(lam) or lam < 0:
        raise ValueError("finite nonnegative lambda required")
    _validate(logits, vessel, fov, selected)
    base = weighted_masked_loss(logits, vessel, fov, fov, epsilon=1e-6)
    if lam == 0:
        return base
    return base + float(lam) * normalized_positive_bce(logits, vessel, fov, selected)


def transform_native_target(mask, *, top, left, size, hflip, vflip, quarter_turns):
    """Crop and transform an ALREADY native-generated target, without RNG."""
    a = np.asarray(mask)
    if a.ndim != 2 or not np.isin(a, (0, 1, False, True)).all():
        raise ValueError("binary native 2D target required")
    if any(type(v) is not int for v in (top, left, size, quarter_turns)):
        raise ValueError("integer geometry required")
    if size <= 0 or top < 0 or left < 0 or top + size > a.shape[0] or left + size > a.shape[1]:
        raise ValueError("crop outside native target")
    if type(hflip) is not bool or type(vflip) is not bool or quarter_turns not in range(4):
        raise ValueError("frozen flip/quarter-turn geometry required")
    out = a[top:top + size, left:left + size]
    if hflip:
        out = out[:, ::-1]
    if vflip:
        out = out[::-1, :]
    if quarter_turns:
        out = np.rot90(out, quarter_turns)
    return np.ascontiguousarray(out)


def human_review_gate(rows, expected_case_ids):
    """Validate real submitted rows only; never create signatures or accepts.

    This structural check cannot authenticate people or resolve scientific
    uncertainty. Sol must inspect the signed evidence before GPU release.
    """
    expected = set(expected_case_ids)
    if not expected:
        raise ValueError("explicit nonempty case whitelist required")
    reasons, seen, signed = [], set(), 0
    fields = ("reviewer_name", "reviewed_at", "GT_path_matches_intended_branch",
              "RGB_visibility", "visually_fine_branch",
              "junction_or_crossing_uncertainty", "GT_gap_or_artifact_uncertainty",
              "width_scale_definition_acceptable", "B1_B2_target_alignment",
              "accept_pause_uncertain")
    for row in rows:
        cid = row.get("case_id", "")
        if cid not in expected or cid in seen:
            reasons.append(f"unexpected_or_duplicate_case:{cid}")
        seen.add(cid)
        if row.get("reviewer_name", "").strip() and row.get("reviewed_at", "").strip():
            signed += 1
        for field in fields:
            if not row.get(field, "").strip():
                reasons.append(f"missing:{cid}:{field}")
        if row.get("accept_pause_uncertain", "").strip().lower() not in ("accept", "接受", "通过"):
            reasons.append(f"not_accepted:{cid}")
    for cid in sorted(expected - seen):
        reasons.append(f"missing_case:{cid}")
    return {"structurally_complete_and_accepted": not reasons,
            "signed_rows": signed, "expected_cases": len(expected),
            "reasons": reasons,
            "human_authenticity_verified": False,
            "gpu_release": False,
            "notice": "Even complete rows require Sol scientific review and explicit release."}
