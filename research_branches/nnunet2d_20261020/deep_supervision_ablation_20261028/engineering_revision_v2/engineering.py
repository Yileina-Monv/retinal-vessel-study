"""Bounded remaining three-forward, cont16 versus independent8+8 engineering. No science transition."""
import os, sys, json, time, signal, subprocess, argparse
from pathlib import Path
from common import *
sys.path.append(str(PARENT_STAGE))


class Counter:
    def __init__(self, path, key):
        self.path = Path(path); self.key = key

    def initialize(self):
        atomic(self.path, {'binding': self.key, 'status': 'READY', 'role': None,
                          'updates_reserved': 0, 'diagnostics_reserved': 0, 'prior_diagnostics_charged': 2, 'cumulative_diagnostics_charged': 2,
                          'forwards': 0, 'backwards': 0, 'optimizer_calls': 0,
                          'successful_updates': 0, 'failed_updates': 0, 'unknown_updates': 0,
                          'pending': None, 'completed_roles': [],
                          'role_counts': {r: 0 for r in ['diag', 'cont', 'split_a', 'split_b']}}, new=True)

    def read(self):
        q = read(self.path)
        if q['binding'] != self.key or not 0 <= q['updates_reserved'] <= 32 or \
           not 0 <= q['diagnostics_reserved'] <= 3:
            raise RuntimeError('engineering ledger binding/cap violated')
        if q['prior_diagnostics_charged'] != 2 or q['cumulative_diagnostics_charged'] != 2 + q['diagnostics_reserved'] or q['cumulative_diagnostics_charged'] > 5: raise RuntimeError('cumulative V2 diagnostic budget drift')
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
        q = self.read(); role = q['role']; limits = {'diag': 3, 'cont': 16, 'split_a': 8, 'split_b': 8}
        if q['status'] != 'READY' or q['pending'] or role not in limits or \
           q['role_counts'][role] >= limits[role] or (kind == 'diag') != (role == 'diag'):
            raise RuntimeError('no authorized engineering slot')
        field = 'diagnostics_reserved' if kind == 'diag' else 'updates_reserved'
        if q[field] >= (3 if kind == 'diag' else 32): raise RuntimeError('engineering cap exhausted')
        q[field] += 1; q['role_counts'][role] += 1
        q['cumulative_diagnostics_charged'] = 2 + q['diagnostics_reserved']
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
        if q['pending'] or q['role_counts'][role] != {'diag': 3, 'cont': 16, 'split_a': 8, 'split_b': 8}[role]:
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
    from candidate import make_trainer, seed_cpu, legacy_digest
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
    if legacy_digest(t.network.state_dict()) != want: raise RuntimeError('original2027 initialization mismatch')
    t.disable_checkpointing = True
    init = {'backbone': want, 'weights': list(t.loss.weight_factors), 'optimizer': digest(t.optimizer.state_dict()),
            'scaler': digest(t.grad_scaler.state_dict()), 'RNG': digest(rng_pack())}
    atomic(out / (role + '_initialization.json'), init, new=True)
    fixtures = read(BASE / 'multiscale_capacity_ablation_v1_20261027/cpu/fixed_batches.json')['batches']
    def get_batch(index):
        item = fixtures[index]
        if sha(item['path']) != item['sha256']: raise RuntimeError('fixed Train batch changed')
        batch = torch.load(item['path'], map_location='cpu', weights_only=False)
        if digest(batch['data']) != item['input_sha256'] or digest(batch['target']) != item['target_sha256']:
            raise RuntimeError('fixed input bytes do not match original baseline')
        if list(map(str,batch['keys'])) != item['ids'] or not set(item['ids']) <= set(protocol()['train_ids']): raise RuntimeError('Train ID mismatch')
        return batch
    handles = []
    def forward_guard(*_): authorize(); counter.compute('forward')
    def backward_guard(grad): authorize(); counter.compute('backward'); return grad
    def optimizer_guard(*_): authorize(); counter.compute('optimizer')
    try:
        if role == 'diag':
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
            ds = DeepSupervisionWrapper(t.loss.loss, [64/127,32/127,16/127,8/127,4/127,2/127,1/127,0])
            names, params = zip(*t.network.named_parameters())
            forbid = t.optimizer.register_step_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('diagnostic optimizer forbidden')))
            handles.append(t.network.register_forward_pre_hook(forward_guard))
            report = []
            def norm(gs): return sum(float(g.double().square().sum()) for g in gs if g is not None)**.5
            def cos(a,b): return sum(float(x.double().mul(y.double()).sum()) for x,y in zip(a,b) if x is not None and y is not None)/(norm(a)*norm(b))
            try:
                for index in range(1,4):
                    authorize(); batch=get_batch(index); counter.reserve('diag')
                    x=batch['data'].to('cuda:0'); ys=[v.to('cuda:0') for v in batch['target']]
                    with torch.autocast('cuda', enabled=True):
                        outs=t.network(x); l0=t.loss(outs,ys); ld=ds(outs,ys); direct=t.loss.loss(outs[0],ys[0])
                    from diagnostic_logging import loss_diagnostic_evidence
                    evidence=loss_diagnostic_evidence(outs,l0,ld,direct)
                    evidence.update(batch=index+1,scale=float(t.grad_scaler.get_scale()),peak_reserved_MiB=memory_check(),scalar_close=bool(torch.isclose(l0,direct,atol=1e-6,rtol=1e-6)))
                    append(out/'pre_gradient_evidence.jsonl',evidence)
                    if not evidence['all_outputs_and_losses_finite'] or not evidence['scalar_close']:
                        save_torch(out/('failed_output_batch%d.pt'%(index+1)),{'outputs':[v.detach().cpu() for v in outs],'targets':[v.detach().cpu() for v in ys],'evidence':evidence})
                        raise RuntimeError('nonfinite output/loss or scalar difference exceeds approved1e-6 tolerance')
                    scale=float(t.grad_scaler.get_scale())
                    def grads(loss, retain):
                        result=torch.autograd.grad(t.grad_scaler.scale(loss),params,retain_graph=retain,allow_unused=True)
                        cpu=[None if v is None else v.detach().float().div(scale).cpu() for v in result]
                        if not states_finite(cpu): raise RuntimeError('nonfinite true unscaled diagnostic gradient')
                        return cpu
                    g0=grads(l0,True); gd=grads(direct,True)
                    if not all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(g0,gd)): raise RuntimeError('wrapper/direct gradient mismatch')
                    del gd
                    gds=grads(ld,False)
                    ga=[None if a is None and b is None else (torch.zeros_like(b) if a is None else a)-(torch.zeros_like(a) if b is None else (64/127)*b) for a,b in zip(gds,g0)]
                    if any(g is not None for n,g in zip(names,g0) if n.startswith('decoder.seg_layers.') and not n.startswith('decoder.seg_layers.7.')): raise RuntimeError('inactive head got gradient')
                    if norm(g0)==0 or norm(gds)==0: raise RuntimeError('zero parameter gradient')
                    row={'batch':index+1,'AMP':True,'shape':list(x.shape),'scale':scale,'candidate_loss':float(l0.detach()),'DS_loss':float(ld.detach()),'unscaled_candidate_L2':norm(g0),'unscaled_DS_L2':norm(gds),'candidate_over_DS':norm(g0)/norm(gds),'candidate_DS_cosine':cos(g0,gds),'final_aux_cosine':cos(g0,ga),'direct_loss_within_approved_tolerance':True,'direct_gradient_exact':True,'loss_abs_difference':evidence['candidate_direct_abs_difference'],'inactive_heads_none':True,'peak_reserved_MiB':memory_check()}
                    append(out/'diagnostic_batches.jsonl',row); report.append(row)
                    counter.resolve()
                    del x,ys,outs,l0,ld,direct,g0,gds,ga,batch
                if legacy_digest(t.network.state_dict()) != want or t.optimizer.state: raise RuntimeError('diagnostics changed parameters/optimizer')
                if digest(rng_pack()) != init['RNG']: raise RuntimeError('diagnostic changed RNG')
                atomic(out/'diagnostics.json',{'status':'PASS','results':report,'updates':0,'Train_only':True,'reserved_peak_MiB':memory_check()},new=True)
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
            named=list(t.network.named_parameters())
            diagnostics['final_head_unscaled_L2']=norm([p for n,p in named if n.startswith('decoder.seg_layers.7.')])
            diagnostics['total_unscaled_L2']=norm(params)
            diagnostics['inactive_heads_none']=all(p.grad is None for n,p in named if n.startswith('decoder.seg_layers.') and not n.startswith('decoder.seg_layers.7.'))
            if not diagnostics['inactive_heads_none']: raise RuntimeError('inactive head gradient during update')
            return old_clip(params, limit, *args, **kwargs)
        try:
            for index in range(start, start + length):
                authorize(); counter.reserve('update'); batch = get_batch(index); began = time.monotonic()
                with patch('torch.nn.utils.clip_grad_norm_', capture): t.train_step(batch)
                if j.read()['success'] != index + 1 or j.read()['skip'] or not states_finite(t.network.state_dict(), t.optimizer.state_dict()):
                    raise RuntimeError('engineering nonfinite/skip/no-update')
                if diagnostics['final_head_unscaled_L2'] <= 0: raise RuntimeError('final head has no gradient')
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
    with exclusive_lock(out / 'RUN.lock'), exclusive_lock(ROOT / 'experiments' / ('a14_gpu0.lock' if mode == 'final' else 'a14_gpu1.lock')):
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
                    if time.monotonic() - clock + protocol()['prior_wall_upper_seconds'] >= 1800:
                        os.killpg(p.pid, signal.SIGTERM)
                        try: p.wait(timeout=10)
                        except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                        counter.stop('cumulative engineering30minute cap'); raise RuntimeError('engineering wall cap')
                    time.sleep(1)
            if p.returncode:
                counter.stop('worker exit ' + str(p.returncode)); raise RuntimeError('engineering stopped:' + role)
        atomic(out / 'completion.json', {'status': 'COMPLETE_PENDING_SOL_REVIEW', 'budget': counter.read(),
                'elapsed_seconds': time.monotonic() - clock, 'cumulative_wall_upper_seconds': time.monotonic() - clock + protocol()['prior_wall_upper_seconds'], 'science_started': False}, new=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['final'], required=True)
    g = p.add_mutually_exclusive_group(required=True); g.add_argument('--run-authorized', action='store_true'); g.add_argument('--worker', action='store_true')
    p.add_argument('--role', choices=['diag', 'cont', 'split_a', 'split_b']); p.add_argument('--parent', type=int); p.add_argument('--nonce')
    a = p.parse_args()
    if a.worker: worker(a.mode, a.role, a.parent, a.nonce)
    else: run_mode(a.mode)


if __name__ == '__main__': main()
