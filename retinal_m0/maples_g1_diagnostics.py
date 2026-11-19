"""Bound real G1 diagnostic math; one forward, four measured gradient calls.

CPU tests use tiny synthetic networks only. Retinal CUDA execution is controlled
by maples_g1_runtime; this module has no launcher or training/update operation.
"""
from __future__ import annotations
import math
import torch
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_branch_bce import branch_bce_loss, normalized_positive_bce
from retinal_m0.maples_g1_safety import GateError, finite_tree


def state_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(f'{name}:{a.dtype}:{a.shape}'.encode()); h.update(a.tobytes())
    return h.hexdigest()


def norm(gradients):
    total = 0.0
    for gradient in gradients:
        if gradient is None or not torch.isfinite(gradient).all():
            raise GateError('missing/nonfinite actual parameter gradient')
        total += float(gradient.detach().double().square().sum().cpu())
    return math.sqrt(total)


def diagnose(model, x, y, fov, selected, *, precision, scaler, record):
    if precision not in ('FP32','AMP'): raise GateError('unknown diagnostic precision')
    if precision == 'AMP' and x.device.type != 'cuda':
        raise GateError('real AMP diagnostics require authorized CUDA, not CPU emulation')
    if not torch.isfinite(x).all(): raise GateError('nonfinite actual input')
    with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                        enabled=precision == 'AMP'):
        z = model(x)  # The ONLY model forward in this diagnostic.
    if z.shape != y.shape: raise GateError('actual model output shape mismatch')
    base = weighted_masked_loss(z, y, fov, fov, epsilon=1e-6)
    zero = branch_bce_loss(z, y, fov, selected, lam=0)
    aux = normalized_positive_bce(z, y, fov, selected)
    total = branch_bce_loss(z, y, fov, selected, lam=.1)
    for value in (base,zero,aux,total):
        if not torch.isfinite(value): raise GateError('nonfinite diagnostic loss')
    counts = selected.float().sum((1,2,3))
    valid = fov.float().sum((1,2,3))
    bce = (torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y.float(),reduction='none')
           * fov.float()).sum((1,2,3)) / valid
    scaler_before = scaler.state_dict()
    scale = float(scaler.get_scale())
    if not math.isfinite(scale) or scale <= 0: raise GateError('invalid loss scale')
    partial = {'precision':precision,'base_loss':float(base.detach()),
               'base_bce':float(bce.mean().detach()),
               'base_dice':float((base-bce.mean()).detach()),
               'aux_unweighted':float(aux.detach()),'aux_weighted':float((.1*aux).detach()),
               'combined_loss':float(total.detach()),'N_FOV':valid.detach().cpu().int().tolist(),
               'N_positive_FOV':(y*fov).sum((1,2,3)).detach().cpu().int().tolist(),
               'N_target':counts.detach().cpu().int().tolist(),
               'scaler_before':scaler_before,'loss_scale':scale,
               'gradient_scaling_removal':'autograd(scaler.scale(loss)) outputs.float()/scale'}
    record('losses',partial)
    parameters = tuple(model.parameters())
    def gradients(loss,label,retain):
        scaled = scaler.scale(loss)
        record('autograd_call',{'objective':label})
        observed = torch.autograd.grad(scaled, parameters+(z,), retain_graph=retain)
        values = tuple(value.detach().float()/scale for value in observed)
        n = norm(values[:-1])
        if not torch.isfinite(values[-1]).all(): raise GateError('nonfinite actual logit gradient')
        record('gradient_measured',{'objective':label,'parameter_L2':n})
        return values[:-1],values[-1]
    gb,zb = gradients(base,'A0',True)
    gz,zz = gradients(zero,'lambda0',True)
    ga,za = gradients(.1*aux,'weighted_aux',True)
    gc,zc = gradients(total,'combined',False)
    zero_value = torch.equal(base.detach(),zero.detach())
    zero_gradient = torch.equal(zb,zz) and all(torch.equal(a,b) for a,b in zip(gb,gz))
    outside = selected == 0
    outside_zero = not bool(torch.count_nonzero(za[outside]).item())
    empty = counts == 0
    empty_zero = not bool(torch.count_nonzero(za[empty]).item()) if bool(empty.any()) else True
    aux_per_crop = (torch.nn.functional.softplus(-z.detach().float())*selected.float()).sum((1,2,3))/counts.clamp_min(1)
    empty_value_zero = bool(torch.all(aux_per_crop[empty] == 0)) if bool(empty.any()) else True
    expected = (.1 / x.shape[0]) * selected.float() * (z.detach().float().sigmoid()-1) / counts.clamp_min(1)[:,None,None,None]
    error = (za-expected).abs()
    analytic = bool(torch.allclose(za,expected,rtol=1e-5,atol=1e-7)) if precision=='FP32' else None
    normalized_error = float((error/(1e-7+1e-5*expected.abs())).max().cpu())
    nb,na,nc = norm(gb),norm(ga),norm(gc)
    dot = sum(float((a.double()*b.double()).sum().cpu()) for a,b in zip(gb,ga))
    cosine = dot/(nb*na) if nb and na else None
    per_layer = {}
    for (name,_),a,b,c in zip(model.named_parameters(),gb,ga,gc):
        per_layer[name]={'A0_L2':norm((a,)),'weighted_aux_L2':norm((b,)),
                         'combined_L2':norm((c,))}
    coefficient = min(1.0, 1.0/(nc+1e-6))
    metrics = {**partial,'zero_lambda_value_exact':zero_value,
        'zero_lambda_gradient_exact':zero_gradient,'empty_target_crop_indices':empty.nonzero().flatten().cpu().tolist(),
        'empty_aux_value_zero':empty_value_zero,'empty_aux_logit_gradient_zero':empty_zero,
        'aux_gradient_outside_target_exact_zero':outside_zero,
        'analytic_FP32_pass':analytic,'analytic_max_absolute_error':float(error.max().cpu()),
        'analytic_max_normalized_error_atol_plus_rtol':normalized_error,
        'grad_l2_base':nb,'grad_l2_aux_weighted':na,'grad_l2_combined':nc,
        'aux_to_base_L2':na/nb if nb else None,
        'aux_to_base_NA_reason':'base_gradient_zero' if not nb else None,
        'cosine_base_aux':cosine,'cosine_NA_reason':'zero_gradient' if cosine is None else None,
        'per_layer_grad_norms':per_layer,'unscaled_gradients':True,
        'clip_max_norm':1.0,'preclip_combined_norm':nc,'clip_coefficient':coefficient,
        'clip_norm_fraction_removed':1.0-coefficient,'clipping_simulated_only':True,
        'would_clip':coefficient<1,'scaler_after':scaler.state_dict(),
        'gradient_review_trigger':na>nb,'autograd_API_calls':4,
        'finite':True,'precision_actually_used':precision}
    record('metrics_observed',metrics)  # Failure evidence survives subsequent validation.
    finite_tree(metrics)
    if not zero_value or not zero_gradient: raise GateError('same-precision lambda0 mismatch')
    if not outside_zero or not empty_zero or not empty_value_zero: raise GateError('empty/off-target auxiliary mismatch')
    if analytic is False: raise GateError('FP32 analytic gradient mismatch')
    if scaler_before != scaler.state_dict(): raise GateError('diagnostic altered scaler state')
    return metrics
