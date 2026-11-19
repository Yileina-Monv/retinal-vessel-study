"""CPU-first sparse absolute CE. No model, optimizer, dataset or device discovery."""
import math
import numpy as np
import torch
import torch.nn.functional as F

def sparse_aux(logits, groups):
    if logits.device.type not in ('cpu','cuda'):
        raise ValueError('Unsupported tensor device')
    if logits.ndim != 4 or logits.shape[1] != 2 or len(groups) != logits.shape[0]:
        raise ValueError('Expected [B,2,H,W] and one group list per crop')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite logits')
    z = logits[:, 1].float() - logits[:, 0].float()
    total = z.sum() * 0
    for j, crop in enumerate(groups):
        losses = []
        positives = set()
        for g in crop:
            p = tuple(g['positive']); negatives = [tuple(x) for x in g['negatives']]
            if not negatives or len(negatives) > 2 or len(set(negatives)) != len(negatives):
                raise ValueError('Need one/two distinct negatives')
            if p in positives or p in negatives:
                raise ValueError('duplicate positive or conflicting polarity')
            positives.add(p)
            for y, x in [p] + negatives:
                if not (0 <= y < z.shape[1] and 0 <= x < z.shape[2]):
                    raise ValueError('index out of range')
            neg = torch.stack([F.softplus(z[j,y,x]) for y,x in negatives]).mean()
            losses.append(.5 * F.softplus(-z[j,p[0],p[1]]) + .5 * neg)
        all_neg = {tuple(n) for g in crop for n in g['negatives']}
        if positives & all_neg:
            raise ValueError('cross-group polarity conflict')
        if losses:
            total = total + torch.stack(losses).mean()
    return total / logits.shape[0]

def total_loss(base, logits, groups, coefficient):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('invalid coefficient')
    if coefficient == 0:
        return base
    return base + coefficient * sparse_aux(logits, groups)

def choose_groups(candidates, seed, attempt, crop):
    """Candidates already geometrically validated; private RNG never touches native RNG."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261031,seed,attempt,crop])))
    pool = sorted(candidates, key=lambda c: (c['native_y'],c['native_x']))
    selected = []
    for g in pool:
        negs = sorted(set(map(tuple,g['negatives'])))
        take = min(2,len(negs))
        if not take:
            raise ValueError('invalid candidate with no background')
        ids = sorted(rng.choice(len(negs),take,replace=False))
        selected.append({**g,'negatives':[list(negs[int(i)]) for i in ids]})
    return selected

def map_points(points, bbox_lbs, input_shape, output_shape, center, affine, axes):
    """Native pixel centers -> exact sampled affine -> nearest output pixel, ties to even."""
    p = np.asarray(points,dtype=np.float64).reshape(-1,2) - np.asarray(bbox_lbs[-2:])
    if affine is None:
        q = p - (np.floor(center).astype(int) - np.asarray(output_shape)//2)
    else:
        q = (p - np.asarray(center) + .5) @ np.linalg.inv(np.asarray(affine,dtype=float)).T
        q += (np.asarray(output_shape)-1)/2
    for axis in axes:
        q[:,axis] = output_shape[axis]-1-q[:,axis]
    return np.rint(q).astype(np.int64), q

def inverse_points(output_yx, bbox_lbs, input_shape, output_shape, center, affine, axes):
    q = np.asarray(output_yx,dtype=float).reshape(-1,2).copy()
    for axis in axes:
        q[:,axis] = output_shape[axis]-1-q[:,axis]
    if affine is None:
        p = q + (np.floor(center).astype(int)-np.asarray(output_shape)//2)
    else:
        p = (q-(np.asarray(output_shape)-1)/2) @ np.asarray(affine,dtype=float).T
        p += np.asarray(center)-.5
    return p + np.asarray(bbox_lbs[-2:])
