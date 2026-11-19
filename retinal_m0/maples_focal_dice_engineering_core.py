"""D-only bounded update/restore loop; no scientific training interface.

Reuse the already-audited persistent 100-update accounting and checkpoint
primitives without editing candidate C or changing the original A0 loss.
"""
from __future__ import annotations
import os
from pathlib import Path
import torch
from retinal_m0.maples_g1_safety import GateError, atomic_json, file_sha, finite_tree
from retinal_m0.maples_g2_engineering import optimizer_update, require_finite_tensors
from retinal_m0.maples_soft_cldice_engineering_core import (
    Budget as OriginalBudget, empty_budget, rng_state, restore_rng,
    atomic_checkpoint, append, pair_record, norm)
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_focal_dice import focal_dice_terms
from retinal_m0.state import state_digest

SEGMENTS = [('CONT', 0, 50), ('SPLIT', 0, 25), ('SPLIT', 25, 50)]


class Budget(OriginalBudget):
    def _read(self):
        s = super()._read()
        for name in ['charged_events', 'verified_successes', 'failed_events',
                     'uncertain_events', 'optimizer_call_starts', 'forward_calls',
                     'gradient_calls', 'failed_compute', 'gradient_calls_this_step']:
            if type(s.get(name)) is not int or s[name] < 0:
                raise GateError('invalid nonnegative integer counter: ' + name)
        if s['gradient_calls_this_step'] > 3:
            raise GateError('more than three gradient calls per forward')
        if s['status'] not in ['READY', 'FAILED_STOP']:
            raise GateError('unknown persistent engineering state')
        return s


def audit(model, base, delta, scaler, budget, track, step, device):
    ps = tuple(model.parameters())
    before_rng = state_digest(rng_state(device))
    before_scaler = scaler.state_dict()
    scale = float(scaler.get_scale())

    def get(loss):
        budget.before_gradient(track, step)
        g = tuple(x.detach().float() / scale for x in torch.autograd.grad(
            scaler.scale(loss), ps, retain_graph=True, allow_unused=False))
        require_finite_tensors(g)
        return g

    gb, gd = get(base), get(delta)
    n0, nd = norm(gb), norm(gd)
    dot = sum(float((a.double() * d.double()).sum().cpu()) for a, d in zip(gb, gd))
    result = {'A0_L2': n0, 'delta_L2': nd,
              'delta_to_A0': nd / n0 if n0 else None,
              'cosine_A0_delta': dot / (n0 * nd) if n0 and nd else None,
              'pause_ratio_gt2': bool((n0 == 0 and nd > 0) or (n0 and nd / n0 > 2)),
              'unscaled': True, 'preclip': True}
    finite_tree(result)
    if (any(p.grad is not None for p in ps)
            or state_digest(rng_state(device)) != before_rng
            or scaler.state_dict() != before_scaler):
        raise GateError('gradient audit changed parameter.grad/RNG/scaler')
    return result


def run_segment(*, model, optimizer, scaler, dataset, track, start, end, out,
                budget, binding_sha, expected_pairs, device, gamma=1,
                authorize=lambda: None):
    from torch.utils.data import DataLoader
    cuda = str(device).startswith('cuda')
    if (track, start, end) not in SEGMENTS:
        raise GateError('only CONT50 / SPLIT25+25 are approved')
    if type(gamma) is not int or gamma != 1:
        raise GateError('engineering candidate is fixed gamma1, not a scan')
    if cuda:
        from retinal_m0.maples_focal_dice_engineering import worker_authority
        authorize = lambda: worker_authority(track, start, end)
        if authorize()['gamma'] != gamma:
            raise GateError('candidate coefficient drift')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    loader_rng = torch.Generator().manual_seed(2026)
    resume_gen = None
    if start == 0:
        out.mkdir(exist_ok=False)
    else:
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        expected = {'track': track, 'step': start, 'next_draw': start * 4,
                    'binding_sha256': binding_sha, 'gamma': 1,
                    'NOT_FOR_SCIENTIFIC_INITIALIZATION': True}
        if any(type(saved.get(k)) is not type(v) or saved[k] != v for k, v in expected.items()):
            raise GateError('resume checkpoint identity drift')
        if cuda and saved['pid'] == os.getpid():
            raise GateError('production restore needs a new worker process')
        if (saved['budget'] != budget.snapshot()
                or saved['train_sha256'] != file_sha(out / 'train.jsonl')
                or saved['attempts_sha256'] != file_sha(out / 'attempts.jsonl')):
            raise GateError('resume ledger/log prefix mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'], device)
        for key, value in [('model', model.state_dict()), ('optimizer', optimizer.state_dict()),
                           ('scaler', scaler.state_dict()), ('rng', rng_state(device))]:
            if state_digest(value) != saved['digests'][key]:
                raise GateError('restored state mismatch: ' + key)
        resume_gen = saved['loader_generator']
        loader_rng.set_state(resume_gen)
        if state_digest(resume_gen) != saved['digests']['loader_generator']:
            raise GateError('restored loader state mismatch')
        atomic_json(out / f'restore_{start:03d}.json', {
            'passed': True, 'new_pid': os.getpid(), 'checkpoint_pid': saved['pid'],
            'checkpoint_sha256': file_sha(out / 'last.pt'), 'ledger_exact': True,
            'full_log_prefix_exact': True, 'source': 'planned new-process restore; no replay'})
        del saved
    model.train()
    loader = DataLoader(dataset, batch_size=4, sampler=range(start * 4, end * 4),
                        num_workers=0, pin_memory=cuda, drop_last=True, generator=loader_rng)
    iterator = iter(loader)
    if resume_gen is not None:
        loader_rng.set_state(resume_gen)
    active = {'step': None}

    def pre_forward(*_):
        authorize()
        budget.before_forward(track, active['step'])

    def pre_optimizer(*_):
        authorize()
        budget.before_optimizer(track, active['step'])

    fh = model.register_forward_pre_hook(pre_forward)
    oh = optimizer.register_step_pre_hook(pre_optimizer)
    if cuda:
        torch.cuda.reset_peak_memory_stats(0)
    try:
        with (out / 'train.jsonl').open('a' if start else 'x') as log, (out / 'attempts.jsonl').open('a' if start else 'x') as attempts:
            step = start
            for batch in iterator:
                active['step'] = step + 1
                pair = pair_record(batch, step + 1)
                if pair != expected_pairs[step]:
                    raise GateError('sampling/augmentation/input differs from original T0')
                optimizer.zero_grad(set_to_none=True)
                x, y, f = [batch[n].to(device) for n in ['image', 'mask', 'fov']]
                if cuda and (x.shape != (4, 3, 512, 512) or y.shape != (4, 1, 512, 512) or f.shape != y.shape):
                    raise GateError('native crop/batch shape drift')
                with torch.autocast('cuda' if cuda else 'cpu', dtype=torch.float16 if cuda else torch.bfloat16, enabled=cuda):
                    z = model(x)
                base = weighted_masked_loss(z, y, f, f)
                terms = focal_dice_terms(z, y, f, gamma=1)
                loss = terms['loss']  # Direct D, not base + a rounded reconstruction.
                delta = loss - base
                require_finite_tensors([base, delta, loss])
                check = audit(model, base, delta, scaler, budget, track, step + 1, device)
                record = {'step': step + 1, 'track': track, 'pid': os.getpid(), 'pair': pair,
                          'A0_loss': float(base.detach()), 'delta_loss': float(delta.detach()),
                          'loss': float(loss.detach()), 'gamma': 1, 'audit': check,
                          'loss_scale_before': float(scaler.get_scale()),
                          'GT_count': terms['GT_count'].detach().cpu().tolist(),
                          'FOV_count': terms['FOV_count'].detach().cpu().tolist(),
                          'per_crop_losses': {k: terms[k].detach().cpu().tolist() for k in
                                              ['per_crop_bce', 'per_crop_focal', 'per_crop_dice']}}
                append(attempts, record)
                if check['pause_ratio_gt2']:
                    raise GateError('delta/A0 ratio>2; stopped before optimizer update')
                budget.before_gradient(track, step + 1)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                require_finite_tensors([p.grad for p in model.parameters() if p.grad is not None])
                clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                record['preclip_total_L2'] = float(clip)
                outcome = optimizer_update(budget, track, step + 1, 1, optimizer, scaler, amp_enabled=cuda)
                if outcome['outcome'] != 'success':
                    raise GateError('AMP overflow/skip: stop without retry/refund')
                step += 1
                optimizer.zero_grad(set_to_none=True)
                rng = rng_state(device)
                digests = {name: state_digest(value) for name, value in [
                    ('model', model.state_dict()), ('optimizer', optimizer.state_dict()),
                    ('scaler', scaler.state_dict()), ('rng', rng),
                    ('loader_generator', loader_rng.get_state()),
                    ('sampling', {'next_draw': step * 4, 'seed': 2026})]}
                record.update(state_digests=digests, loss_scale_after=float(scaler.get_scale()),
                              GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated(0) / 1048576 if cuda else 0.,
                              GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved(0) / 1048576 if cuda else 0.)
                finite_tree(record)
                append(log, record)
                payload = {'schema': 1, 'NOT_FOR_SCIENTIFIC_INITIALIZATION': True,
                           'track': track, 'step': step, 'next_draw': step * 4, 'pid': os.getpid(),
                           'binding_sha256': binding_sha, 'gamma': 1, 'model': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                           'scheduler': None, 'rng': rng, 'loader_generator': loader_rng.get_state(),
                           'digests': digests, 'budget': budget.snapshot(),
                           'train_sha256': file_sha(out / 'train.jsonl'),
                           'attempts_sha256': file_sha(out / 'attempts.jsonl')}
                atomic_checkpoint(out / 'last.pt', payload, replace=True)
                if step in [25, 50]:
                    atomic_checkpoint(out / f'step_{step:03d}.pt', payload, replace=False)
                if cuda and torch.cuda.max_memory_reserved(0) > 8 * 1024**3:
                    raise GateError('8GiB allocator cap exceeded')
                print('UPDATE', track, step, '/50', 'delta/A0', check['delta_to_A0'], flush=True)
                del z, base, terms, delta, loss, x, y, f, payload
        return {'track': track, 'start': start, 'end': end, 'pid': os.getpid(),
                'last_sha256': file_sha(out / 'last.pt'), 'budget': budget.snapshot()}
    except BaseException as exc:
        budget.stop(repr(exc))
        raise
    finally:
        fh.remove()
        oh.remove()
