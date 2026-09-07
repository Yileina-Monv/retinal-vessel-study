"""FP32 per-image pixel loss plus valid-image skeleton recall reduction."""
import math
import torch
from retinal_release.data import masked_loss

def m1_loss(logits, target, fov, tube, *, coefficient, total_images=None, total_valid=None):
    if logits.shape != tube.shape or logits.shape != target.shape or target.shape != fov.shape:
        raise ValueError('Matching NCHW tensors required')
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Invalid coefficient')
    if bool(((tube < 0) | (tube > 1) | (tube > target*fov)).any()):
        raise ValueError('Tube must be a subset of foreground inside FOV')
    counts=tube.float().sum((1,2,3));valid=counts>0
    n=logits.shape[0];v=int(valid.sum())
    if (total_images is None) != (total_valid is None):
        raise ValueError('Supply both effective-batch denominators or neither')
    total_images=n if total_images is None else total_images
    total_valid=v if total_valid is None else total_valid
    if total_images<n or not v<=total_valid<=total_images:
        raise ValueError('Invalid effective-batch denominators')
    pixel=masked_loss(logits,target,fov)*n/total_images
    recall_loss=1-((logits.float().sigmoid()*tube.float()).sum((1,2,3))+1e-6)/(counts+1e-6)
    skeleton=(recall_loss*valid).sum()/max(1,total_valid)
    return pixel+coefficient*skeleton, dict(pixel=pixel.detach(), skeleton=skeleton.detach(),
        empty_skeleton_crops=n-v, crops=n)
