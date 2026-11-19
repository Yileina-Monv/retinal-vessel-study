"""Bounded, same-graph diagnostic rescaling. Never touches the training GradScaler."""
import math
import torch

class DiagnosticFailure(RuntimeError):
    pass

def bounded_gradients(loss, named_parameters, initial_scale, component, record):
    if component not in ("base", "aux"):
        raise DiagnosticFailure("unknown diagnostic component")
    params = tuple(named_parameters)
    if not params or loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
        raise DiagnosticFailure("nonfinite or nonscalar diagnostic loss")
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise DiagnosticFailure("invalid diagnostic scale")
    if any(p.grad is not None for _, p in params):
        raise DiagnosticFailure("diagnostic requires cleared parameter.grad")
    scales = [initial_scale / (2 ** i) for i in range(4 if component == "base" else 1)]
    for index, scale in enumerate(scales):
        raw = torch.autograd.grad(loss * scale, tuple(p for _, p in params),
                                  retain_graph=True, allow_unused=True)
        grads = [None if g is None else g.detach().float().div(scale).cpu() for g in raw]
        del raw
        bad = []
        for (name, _), grad in zip(params, grads):
            if grad is not None and not bool(torch.isfinite(grad).all()):
                bad.append({"parameter": name, "nan": int(torch.isnan(grad).sum()),
                            "inf": int(torch.isinf(grad).sum())})
        row = {"component": component, "diagnostic_try": index + 1, "scale": scale,
               "initial_scale": initial_scale, "finite": not bad, "nonfinite": bad,
               "extra_model_forwards": 0, "optimizer_updates": 0}
        record(row)
        if any(p.grad is not None for _, p in params):
            raise DiagnosticFailure("diagnostic accumulated parameter.grad")
        if not bad:
            return grads, {"initial_scale": initial_scale, "accepted_scale": scale,
                           "tries": index + 1, "fallback_used": index > 0}
        del grads
    raise DiagnosticFailure("nonfinite diagnostic " + component + " after bounded scaling")
