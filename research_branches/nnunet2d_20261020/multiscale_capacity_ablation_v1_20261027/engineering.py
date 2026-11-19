"""Bounded four-forward, cont16 versus independent8+8 engineering. No science transition."""
import os, sys, json, time, signal, subprocess, argparse
from pathlib import Path
from common import *


class Counter:
    def __init__(self, path, key):
        self.path = Path(path); self.key = key

    def initialize(self):
        atomic(self.path, {'binding': self.key, 'status': 'READY', 'role': None,
                          'updates_reserved': 0, 'diagnostics_reserved': 0,
                          'forwards': 0, 'backwards': 0, 'optimizer_calls': 0,
                          'successful_updates': 0, 'failed_updates': 0, 'unknown_updates': 0,
                          'pending': None, 'completed_roles': [],
                          'role_counts': {r: 0 for r in ['diag', 'cont', 'split_a', 'split_b']}}, new=True)

    def read(self):
        q = read(self.path)
        if q['binding'] != self.key or not 0 <= q['updates_reserved'] <= 32 or \
           not 0 <= q['diagnostics_reserved'] <= 4:
            raise RuntimeError('engineering ledger binding/cap violated')
        pending = int(q['pending'] is not None and q['pending']['kind'] == 'update')
        if q['updates_reserved'] != q['successful_updates'] + q['failed_updates'] + q['unknown_updates'] + pending:
            raise RuntimeError('engineering persistent accounting mismatch')
        if q['forwards'] > q['updates_reserved'] + q['diagnostics_reserved'] or \
           q['backwards'] > q['updates_reserved'] or q['optimizer_calls'] > q['updates_reserved']:
            raise RuntimeError('unreserved engineering computation')
        return q

    def start_role(self, role):
        q = self.read(); order = ['diag', 'cont', 'split_a', 'split_b']
        if q['status'] != 'READY' or q['pending'] or q['role'] is not None or \
           order[len(q['completed_roles'])] != role: raise RuntimeError('wrong/repeated engineering role')
        q['role'] = role; atomic(self.path, q)

    def reserve(self, kind):
        q = self.read(); role = q['role']; limits = {'diag': 4, 'cont': 16, 'split_a': 8, 'split_b': 8}
        if q['status'] != 'READY' or q['pending'] or role not in limits or \
           q['role_counts'][role] >= limits[role] or (kind == 'diag') != (role == 'diag'):
            raise RuntimeError('no authorized engineering slot')
        field = 'diagnostics_reserved' if kind == 'diag' else 'updates_reserved'
        if q[field] >= (4 if kind == 'diag' else 32): raise RuntimeError('engineering cap exhausted')
        q[field] += 1; q['role_counts'][role] += 1
        q['pending'] = {'kind': kind, 'forward': 0, 'backward': 0, 'optimizer': 0}
        atomic(self.path, q); return q[field]

    def compute(self, kind):
        q = self.read(); p = q['pending']
        if q['status'] != 'READY' or p is None or p[kind] != 0: raise RuntimeError('unbound/extra computation')
        if kind == 'forward': field = 'forwards'
        elif kind == 'backward':
            if p['kind'] != 'update' or not p['forward']: raise RuntimeError('unauthorized backward')
            field = 'backwards'
        elif kind == 'optimizer':
            if p['kind'] != 'update' or not p['backward']: raise RuntimeError('unauthorized optimizer.step')
            field = 'optimizer_calls'
        else: raise RuntimeError('unknown computation')
        p[kind] = 1; q[field] += 1; atomic(self.path, q)

    def resolve(self):
        q = self.read(); p = q['pending']
        want = {'kind': 'diag', 'forward': 1, 'backward': 0, 'optimizer': 0} if p and p['kind'] == 'diag' else \
               {'kind': 'update', 'forward': 1, 'backward': 1, 'optimizer': 1}
        if p != want: raise RuntimeError('engineering requires exact finite no-skip update/diagnostic')
        if p['kind'] == 'update': q['successful_updates'] += 1
        q['pending'] = None; atomic(self.path, q)

    def finish_role(self):
        q = self.read(); role = q['role']
        if q['pending'] or q['role_counts'][role] != {'diag': 4, 'cont': 16, 'split_a': 8, 'split_b': 8}[role]:
            raise RuntimeError('incomplete engineering role')
        q['completed_roles'].append(role); q['role'] = None
        if role == 'split_b': q['status'] = 'COMPLETE_PENDING_REVIEW'
        atomic(self.path, q)

    def stop(self, reason):
        q = self.read()
        if q['pending'] and q['pending']['kind'] == 'update': q['unknown_updates'] += 1
        q['pending'] = None; q['status'] = 'FAILED_STOP'; q.setdefault('reason', str(reason)); atomic(self.path, q)


def worker(mode, role, parent, nonce):
    key, authorize = admit_worker(mode, 'engineering', role, parent, nonce)
    out = STAGE / 'engineering' / mode
    os.environ.update(nnUNet_raw=str(BASE / 'raw'), nnUNet_preprocessed=str(BASE / 'preprocessed'),
                      nnUNet_results=str(out / ('native_' + role)))
    import torch, numpy as np, copy
    from unittest.mock import patch
    from fusion_model import make_trainer, seed_cpu, original_state, legacy_digest, DETAIL_PREFIX
    from fast_observer import FastObserver, Journal, AmpLimits, digest, rng_pack, restore_rng, states_finite
    from seeded_runtime import install_seeded_worker_adapter
    if torch.cuda.is_initialized(): raise RuntimeError('CUDA initialized before engineering admission')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    if torch.cuda.device_count() != 1: raise RuntimeError('exactly one released card must be visible')
    torch.cuda.set_per_process_memory_fraction(10240 / torch.cuda.get_device_properties(0).total_memory * 1048576)
    seed_cpu(); torch.cuda.manual_seed_all(2027)
    counter = Counter(out / 'budget.json', key)
    t = make_trainer(mode, torch.device('cuda:0'), out / ('native_' + role), BASE)
    want = read(N0 / 'initialization.json')['model']
    if legacy_digest(original_state(t.network)) != want: raise RuntimeError('original2027 initialization mismatch')
    t.disable_checkpointing = True
    module = t.network.decoder.stages[-1].detail
    init = {'backbone': want, 'module': digest(module.state_dict()), 'optimizer': digest(t.optimizer.state_dict()),
            'scaler': digest(t.grad_scaler.state_dict()), 'RNG': digest(rng_pack())}
    atomic(out / (role + '_initialization.json'), init, new=True)
    fixtures = read(STAGE / 'cpu' / 'fixed_batches.json')['batches']
    def get_batch(index):
        item = fixtures[index]
        if sha(item['path']) != item['sha256']: raise RuntimeError('fixed Train batch changed')
        batch = torch.load(item['path'], map_location='cpu', weights_only=False)
        if digest(batch['data']) != item['input_sha256'] or digest(batch['target']) != item['target_sha256']:
            raise RuntimeError('fixed input bytes do not match original baseline')
        return batch
    handles = []
    def forward_guard(*_): authorize(); counter.compute('forward')
    def backward_guard(grad): authorize(); counter.compute('backward'); return grad
    def optimizer_guard(*_): authorize(); counter.compute('optimizer')
    try:
        if role == 'diag':
            batch = get_batch(0); data = batch['data'].to('cuda:0', non_blocking=False)
            base = copy.deepcopy(t.network)
            base.decoder.stages[-1] = base.decoder.stages[-1].base
            t.network.train(); base.train()
            # Optimizer is explicitly forbidden throughout the zero-update diagnostic.
            forbid = t.optimizer.register_step_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('diagnostic optimizer forbidden')))
            outputs = {}; report = []
            handles += [base.register_forward_pre_hook(forward_guard), t.network.register_forward_pre_hook(forward_guard)]
            try:
                for amp in [False, True]:
                    for name, network in [('original', base), ('variant', t.network)]:
                        authorize(); counter.reserve('diag')
                        with torch.no_grad(), torch.autocast('cuda', enabled=amp): result = network(data)
                        if not all(torch.isfinite(v).all() for v in result): raise RuntimeError('nonfinite native diagnostic output')
                        outputs[(amp, name)] = [v.detach().cpu() for v in result]
                        counter.resolve(); memory_check()
                    equal = all(torch.equal(a, b) for a, b in zip(outputs[(amp, 'original')], outputs[(amp, 'variant')]))
                    if not equal: raise RuntimeError('initial full-network identity failed')
                    report.append({'AMP': amp, 'all8_outputs_exact': equal,
                                   'shapes': [list(v.shape) for v in outputs[(amp, 'variant')]]})
                if digest(module.state_dict()) != init['module']: raise RuntimeError('diagnostics changed model')
                atomic(out / 'diagnostics.json', {'status': 'PASS', 'results': report, 'updates': 0,
                                                 'reserved_peak_MiB': memory_check(), 'Train_only': True}, new=True)
            finally: forbid.remove()
            counter.finish_role(); return
        trajectory = out / ('cont' if role == 'cont' else 'split')
        trajectory.mkdir(exist_ok=True)
        j = Journal(trajectory / 'budget.json', key, 1, 16, 1, AmpLimits(1, 1))
        start = 8 if role == 'split_b' else 0; length = 16 if role == 'cont' else 8
        if start == 0: j.initialize()
        observer = FastObserver(t, j, trajectory, set(protocol()['train_ids']), authorize)
        if start == 8:
            state = torch.load(out / 'split8.pt', map_location='cpu', weights_only=False)
            for k in state['digests']:
                if digest(state[k]) != state['digests'][k]: raise RuntimeError('split snapshot digest mismatch:' + k)
            if state['budget'] != j.read() or state['budget']['attempts'] != 8: raise RuntimeError('split cursor drift')
            t.network.load_state_dict(state['model']); t.optimizer.load_state_dict(state['optimizer'])
            t.grad_scaler.load_state_dict(state['scaler']); t.lr_scheduler.load_state_dict(state['scheduler'])
            if state['logging'] is not None: t.logger.load_checkpoint(state['logging'])
            t._best_ema = state['best_ema']
            observer.partial_train = state['partial_train']; observer.partial_monitor = state['partial_monitor']
            t.current_epoch = state['native_current_epoch']; restore_rng(state['rng'])
            for k, v in [('model', t.network.state_dict()), ('optimizer', t.optimizer.state_dict()),
                         ('scaler', t.grad_scaler.state_dict()), ('scheduler', t.lr_scheduler.state_dict()), ('rng', rng_pack())]:
                if digest(v) != state['digests'][k]: raise RuntimeError('restored value mismatch:' + k)
            atomic(out / 'restore_verification.json', {'status': 'PASS', 'all_available_states_exact': True,
                    'snapshot_sha256': sha(out / 'split8.pt'), 'native_sampler_resume_claimed': False}, new=True)
        else:
            t.on_train_epoch_start()
        handles += [t.network.register_forward_pre_hook(forward_guard),
                    next(t.network.parameters()).register_hook(backward_guard),
                    t.optimizer.register_step_pre_hook(optimizer_guard)]
        old_clip = torch.nn.utils.clip_grad_norm_
        diagnostics = {}
        def capture(params, limit, *args, **kwargs):
            params = list(params)
            def norm(values):
                return float(torch.stack([p.grad.detach().float().square().sum() for p in values if p.grad is not None]).sum().sqrt().cpu())
            diagnostics['module_unscaled_L2'] = norm(module.parameters())
            diagnostics['projection_unscaled_L2'] = norm(module.project.parameters())
            diagnostics['internal_unscaled_L2'] = norm([p for n, p in module.named_parameters() if not n.startswith('project.')])
            return old_clip(params, limit, *args, **kwargs)
        try:
            for index in range(start, start + length):
                authorize(); counter.reserve('update'); batch = get_batch(index); began = time.monotonic()
                with patch('torch.nn.utils.clip_grad_norm_', capture): t.train_step(batch)
                if j.read()['success'] != index + 1 or j.read()['skip'] or not states_finite(t.network.state_dict(), t.optimizer.state_dict()):
                    raise RuntimeError('engineering nonfinite/skip/no-update')
                if index > 0 and (diagnostics['internal_unscaled_L2'] <= 0 or diagnostics['projection_unscaled_L2'] <= 0):
                    raise RuntimeError('new branch fails to receive real unscaled gradients')
                counter.resolve()
                append(out / 'gradient_diagnostics.jsonl', {'role': role, 'fixed_batch': index + 1, **diagnostics,
                       'seconds': time.monotonic() - began, 'peak_reserved_MiB': memory_check()})
            target = out / ('cont16.pt' if role == 'cont' else 'split8.pt' if role == 'split_a' else 'split16.pt')
            observer.save_bundle(target, key)
            atomic(out / (role + '_complete.json'), {'status': 'PASS', 'updates_this_role': length,
                   'trajectory_successes': j.read()['success'], 'checkpoint_sha256': sha(target),
                   'peak_reserved_MiB': memory_check()}, new=True)
            counter.finish_role()
        finally: observer.close()
    except BaseException as exc:
        counter.stop(repr(exc)); atomic(out / (role + '_failure.json'), {'error': repr(exc), 'budget': counter.read()})
        raise
    finally:
        for handle in handles: handle.remove()


def run_mode(mode):
    validate(mode, 'engineering', full=True)
    out = STAGE / 'engineering' / mode
    with exclusive_lock(out / 'RUN.lock'), exclusive_lock(ROOT / 'experiments' / ('a14_gpu0.lock' if mode == 'local' else 'a14_gpu1.lock')):
        resource = resources(mode)
        if (out / 'controller_session.json').exists(): raise RuntimeError('session consumed; no restart')
        key = validate(mode, 'engineering'); counter = Counter(out / 'budget.json', key)
        if counter.read()['updates_reserved'] or counter.read()['diagnostics_reserved']: raise RuntimeError('engineering budget not fresh')
        atomic(out / 'controller_session.json', {'identity': process_identity(), 'resource': resource}, new=True)
        clock = time.monotonic()
        for role in ['diag', 'cont', 'split_a', 'split_b']:
            counter.start_role(role); nonce = os.urandom(24).hex(); env = worker_env(mode, nonce, out)
            with (out / (role + '.log')).open('x', encoding='utf-8') as log:
                p = subprocess.Popen([sys.executable, '-u', '-B', str(Path(__file__)), '--worker',
                    '--mode', mode, '--role', role, '--parent', str(os.getpid()), '--nonce', nonce],
                    cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic(out / (role + '_session.json'), {'worker': process_identity(p.pid), 'controller': process_identity(),
                        'nonce': nonce, 'mode': mode, 'role': role}, new=True)
                while p.poll() is None:
                    if time.monotonic() - clock >= 1800:
                        os.killpg(p.pid, signal.SIGTERM)
                        try: p.wait(timeout=10)
                        except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                        counter.stop('cumulative engineering30minute cap'); raise RuntimeError('engineering wall cap')
                    time.sleep(1)
            if p.returncode:
                counter.stop('worker exit ' + str(p.returncode)); raise RuntimeError('engineering stopped:' + role)
        atomic(out / 'completion.json', {'status': 'COMPLETE_PENDING_SOL_REVIEW', 'budget': counter.read(),
                'elapsed_seconds': time.monotonic() - clock, 'science_started': False}, new=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['local', 'multi'], required=True)
    g = p.add_mutually_exclusive_group(required=True); g.add_argument('--run-authorized', action='store_true'); g.add_argument('--worker', action='store_true')
    p.add_argument('--role', choices=['diag', 'cont', 'split_a', 'split_b']); p.add_argument('--parent', type=int); p.add_argument('--nonce')
    a = p.parse_args()
    if a.worker: worker(a.mode, a.role, a.parent, a.nonce)
    else: run_mode(a.mode)


if __name__ == '__main__': main()
