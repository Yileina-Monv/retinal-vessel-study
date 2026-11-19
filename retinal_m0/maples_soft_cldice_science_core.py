"""Frozen C scientific runner primitives. Importing this module never initializes CUDA."""
from __future__ import annotations
import hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from retinal_m0.maples_g1_safety import GateError, atomic_json, exclusive_lock, read_json, finite_tree
from retinal_m0.maples_soft_cldice import soft_cldice_terms
from retinal_m0.maples_core import weighted_masked_loss
from retinal_m0.maples_soft_cldice_engineering_core import rng_state, restore_rng, atomic_checkpoint, append, pair_record, norm
from retinal_m0.state import state_digest

GLOBAL_CAP = 2000
TRACK_CAPS = {'C': 2000}
FORWARD_CAP = 2000
GRADIENT_CAP = 6000
ETA = 0.15115714003619224
K = 26
SEED = 2026


def validate_scope(scope):
    expected = {'method': 'C', 'seed': SEED, 'updates': GLOBAL_CAP, 'eta': ETA, 'k': K}
    if scope != expected or any(type(scope.get(k)) is not type(v) for k, v in expected.items()):
        raise GateError('scientific scope differs from frozen C protocol')
    return True


def verify_pair_rows(actual, expected):
    if len(actual) != len(expected):
        raise GateError('paired input row count differs')
    for i, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise GateError(f'paired input differs at update {i+1}')
    return True


def _empty_budget(binding, global_cap, track_caps, forward_cap, gradient_cap):
    return {'schema': 1, 'binding_sha256': binding, 'global_cap': global_cap,
            'track_caps': dict(track_caps), 'forward_cap': forward_cap, 'gradient_cap': gradient_cap,
            'status': 'READY', 'charged_events': 0, 'verified_successes': 0,
            'failed_events': 0, 'uncertain_events': 0, 'optimizer_call_starts': 0,
            'forward_calls': 0, 'gradient_calls': 0, 'failed_compute': 0,
            'tracks': {}, 'compute_pending': None, 'gradient_calls_this_step': 0}


class Budget:
    """Persistent, fail-closed ledger shared by production and CPU fixture loops."""
    def __init__(self, path, binding='cpu-fixture', *, global_cap=GLOBAL_CAP,
                 track_caps=None, forward_cap=FORWARD_CAP, gradient_cap=GRADIENT_CAP):
        self.path = Path(path); self.binding = binding
        self.global_cap = global_cap; self.track_caps = dict(track_caps or TRACK_CAPS)
        self.forward_cap = forward_cap; self.gradient_cap = gradient_cap

    def initialize(self):
        with exclusive_lock(str(self.path)+'.lock'):
            if self.path.exists(): raise GateError('budget exists; never reset or rebind')
            atomic_json(self.path, _empty_budget(self.binding, self.global_cap, self.track_caps,
                                                 self.forward_cap, self.gradient_cap))

    def _read(self):
        s = read_json(self.path)
        identity = (s.get('schema') == 1 and s.get('binding_sha256') == self.binding
                    and s.get('global_cap') == self.global_cap and s.get('track_caps') == self.track_caps
                    and s.get('forward_cap') == self.forward_cap and s.get('gradient_cap') == self.gradient_cap)
        if not identity: raise GateError('scientific budget identity/cap drift')
        keys = ('charged_events','verified_successes','failed_events','uncertain_events','optimizer_call_starts')
        totals = {k: 0 for k in keys}
        for name, t in s['tracks'].items():
            if name not in self.track_caps or t['charged_events'] != t['verified_successes'] + t['failed_events'] + t['uncertain_events'] or t['charged_events'] > self.track_caps[name]:
                raise GateError('invalid per-track accounting')
            for k in keys:
                if type(t.get(k)) is not int or t[k] < 0: raise GateError('invalid budget counter')
                totals[k] += t[k]
        if any(s[k] != v for k,v in totals.items()): raise GateError('aggregate budget mismatch')
        if s['charged_events'] > self.global_cap or s['forward_calls'] > self.forward_cap or s['gradient_calls'] > self.gradient_cap:
            raise GateError('scientific budget exceeded')
        return s

    def snapshot(self):
        with exclusive_lock(str(self.path)+'.lock'): return self._read()

    def _change(self, fn):
        with exclusive_lock(str(self.path)+'.lock'):
            s = self._read(); fn(s); atomic_json(self.path, s); self._read()

    def before_forward(self, track, step):
        def change(s):
            if s['status'] != 'READY' or s['compute_pending'] is not None or any(t.get('pending') for t in s['tracks'].values()): raise GateError('stopped or unresolved scientific work')
            if track not in self.track_caps or type(step) is not int: raise GateError('unknown trajectory')
            t=s['tracks'].setdefault(track, {'charged_events':0,'verified_successes':0,'failed_events':0,'uncertain_events':0,'optimizer_call_starts':0,'pending':None})
            if step != t['verified_successes']+1 or t['charged_events'] >= self.track_caps[track] or s['charged_events'] >= self.global_cap or s['forward_calls'] >= self.forward_cap: raise GateError('nonsequential update or forward budget exhausted')
            s['forward_calls'] += 1; s['compute_pending']={'track':track,'step':step}; s['gradient_calls_this_step']=0
        self._change(change)

    def before_gradient(self, track, step, count=1):
        if type(count) is not int or count < 1: raise GateError('invalid gradient reservation')
        def change(s):
            if s['status'] != 'READY' or s['compute_pending'] != {'track':track,'step':step} or s['gradient_calls_this_step']+count > 3 or s['gradient_calls']+count > self.gradient_cap: raise GateError('extra or unbound gradient call')
            s['gradient_calls'] += count; s['gradient_calls_this_step'] += count
        self._change(change)

    def reserve(self, track, step):
        def change(s):
            if s['status'] != 'READY' or s['compute_pending'] != {'track':track,'step':step} or s['gradient_calls_this_step'] != 3: raise GateError('optimizer requested without three gradient calls')
            t=s['tracks'][track]
            if t['pending'] is not None or t['charged_events'] >= self.track_caps[track] or s['charged_events'] >= self.global_cap: raise GateError('optimizer pending or cap exhausted')
            t['pending']={'step':step,'started':False,'reserved_at':time.time()}
        self._change(change)

    def optimizer_started(self, track, step):
        def change(s):
            t=s['tracks'].get(track); p=None if t is None else t.get('pending')
            if s['status'] != 'READY' or p is None or p['step'] != step or p['started']: raise GateError('unauthorized or duplicate optimizer call')
            p['started']=True; t['optimizer_call_starts']+=1; s['optimizer_call_starts']+=1
        self._change(change)

    def resolve(self, track, outcome):
        def change(s):
            t=s['tracks'].get(track); p=None if t is None else t.get('pending')
            if p is None: raise GateError('no unresolved optimizer event')
            if outcome == 'success' and not p['started']: raise GateError('success without real optimizer call')
            if outcome not in ('success','failed','uncertain'): raise GateError('invalid optimizer outcome')
            k={'success':'verified_successes','failed':'failed_events','uncertain':'uncertain_events'}[outcome]
            t[k]+=1; s[k]+=1; t['charged_events']+=1; s['charged_events']+=1
            t['last_outcome']={'outcome':outcome,'pending_record':p}; t['pending']=None; s['compute_pending']=None
            if outcome != 'success': s['status']='FAILED_STOP'
        self._change(change)

    def stop(self, reason):
        def change(s):
            if s['compute_pending'] is not None and s['status'] != 'FAILED_STOP': s['failed_compute'] += 1
            s.setdefault('first_stop_reason', str(reason)); s['status']='FAILED_STOP'; s['stop_reason']=str(reason)
        self._change(change)


def audit_parameter_gradients(model, base_loss, aux_loss, scaler, budget, track, step, device):
    params=tuple(model.parameters()); before=state_digest(rng_state(device)); scale=float(scaler.get_scale())
    def grad(loss):
        budget.before_gradient(track, step)
        result=torch.autograd.grad(scaler.scale(loss), params, retain_graph=True, allow_unused=False)
        values=tuple(g.detach().float()/scale for g in result)
        if any(not torch.isfinite(g).all() for g in values): raise GateError('nonfinite unscaled parameter gradient')
        return values
    g0=grad(base_loss); g1=grad(aux_loss); n0=norm(g0); n1=norm(g1)
    dot=sum(float((a.double()*b.double()).sum().cpu()) for a,b in zip(g0,g1))
    result={'A0_L2':n0,'weighted_aux_L2':n1,'aux_to_A0':n1/n0 if n0 else None,
            'cosine':dot/(n0*n1) if n0 and n1 else None,
            'pause_ratio_gt2':bool((n0==0 and n1>0) or (n0 and n1/n0>2)),
            'unscaled':True,'preclip':True}
    finite_tree(result)
    if any(p.grad is not None for p in params) or state_digest(rng_state(device)) != before: raise GateError('gradient audit changed gradients or RNG')
    return result


def optimizer_hooks(optimizer, budget, track, active_step, authorize=lambda:None):
    def pre(*_): authorize(); budget.optimizer_started(track, active_step['step'])
    return optimizer.register_step_pre_hook(pre)


def install_step_verifier(optimizer, before_parameters):
    """Return a post-hook that records whether the actual optimizer changed parameters."""
    result={'called':False,'changed':False}
    def post(*_):
        result['called']=True
        result['changed']=any(not torch.equal(a.detach(),b.detach()) for a,b in zip(before_parameters, optimizer.param_groups[0]['params']))
    handle=optimizer.register_step_post_hook(post)
    return result, handle

def run_updates(*, model, optimizer, scaler, dataset, track, start, end, out, budget,
                binding_sha, expected_pairs, device, eta=ETA, k=K, authorize=lambda:None, on_update=None):
    """Shared C update loop used by production and CPU resume fixtures."""
    from torch.utils.data import DataLoader
    from retinal_m0.maples_g2_engineering import require_finite_tensors
    if track not in TRACK_CAPS or not (0 <= start < end <= TRACK_CAPS[track]) or eta != ETA or k != K:
        raise GateError('only the frozen C track, budget, eta and k are allowed')
    if len(expected_pairs) < end: raise GateError('expected pair evidence is shorter than requested segment')
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); cuda=str(device).startswith('cuda')
    loader_rng=torch.Generator().manual_seed(SEED); resume_state=None; elapsed0=0.; started_at=time.perf_counter()
    fresh_initial_state={'model':state_digest(model.state_dict()),'optimizer':state_digest(optimizer.state_dict()),'scaler':state_digest(scaler.state_dict()),'rng':state_digest(rng_state(device))}
    initial_state=fresh_initial_state
    if start == 0:
        out.mkdir(exist_ok=False)
    else:
        saved=verify_checkpoint(out/'last.pt',expected_step=start,binding_sha=binding_sha,run_dir=out,ledger=budget)
        if start%200==0:
            milestone=verify_checkpoint(out/f'step_{start:04d}.pt',expected_step=start,binding_sha=binding_sha,run_dir=out)
            if milestone['digests']!=saved['digests'] or milestone['budget']!=saved['budget']: raise GateError('last and matching milestone differ')
            del milestone
        if saved.get('track')!=track or saved.get('step')!=start or saved.get('next_draw')!=start*4 or saved.get('binding_sha256')!=binding_sha or saved.get('eta')!=eta or saved.get('k')!=k:
            raise GateError('resume checkpoint identity or cursor drift')
        if saved.get('budget')!=budget.snapshot() or saved.get('train_sha256')!=sha256_file(out/'train.jsonl') or saved.get('attempts_sha256')!=sha256_file(out/'attempts.jsonl'):
            raise GateError('resume budget or full log prefix mismatch')
        if saved.get('initial_state')!=fresh_initial_state: raise GateError('fresh M0/optimizer/scaler/RNG initialization differs from checkpoint origin')
        initial_state=saved['initial_state']
        model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        restore_rng(saved['rng'],device); loader_rng.set_state(saved['loader_generator']); resume_state=saved['loader_generator']; elapsed0=float(saved['elapsed_seconds'])
        for key,value in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state(device)),('loader_generator',loader_rng.get_state())]:
            if state_digest(value)!=saved['digests'][key]: raise GateError('restored checkpoint state mismatch: '+key)
        atomic_json(out/f'restore_{start:04d}.json',{'step':start,'passed':True,'checkpoint_sha256':sha256_file(out/'last.pt'),'budget_exact':True,'log_prefix_exact':True,'pid':os.getpid()})
        del saved
    model.train(); loader=DataLoader(dataset,batch_size=4,sampler=range(start*4,end*4),num_workers=0,pin_memory=cuda,drop_last=True,generator=loader_rng)
    iterator=iter(loader)
    if resume_state is not None: loader_rng.set_state(resume_state)
    active={'step':None,'evaluation':False}; handles=[]
    def pre_forward(*_):
        if active['evaluation']: return
        authorize(); budget.before_forward(track,active['step'])
    handles.append(model.register_forward_pre_hook(pre_forward))
    probe={'before':None,'called':False,'changed':False}
    def pre_optimizer(*_):
        authorize(); budget.optimizer_started(track,active['step'])
        probe['before']=[p.detach().clone() for p in model.parameters()]; probe['called']=False; probe['changed']=False
    def post_optimizer(*_):
        probe['called']=True
        probe['changed']=any(not torch.equal(a,p.detach()) for a,p in zip(probe['before'],model.parameters()))
    handles.append(optimizer.register_step_pre_hook(pre_optimizer))
    handles.append(optimizer.register_step_post_hook(post_optimizer))
    try:
        with (out/'train.jsonl').open('a' if start else 'x',encoding='utf-8') as log, (out/'attempts.jsonl').open('a' if start else 'x',encoding='utf-8') as attempts:
            step=start
            for batch in iterator:
                step_id=step+1; active['step']=step_id; pair=pair_record(batch,step_id)
                if pair != expected_pairs[step]: raise GateError('actual sample/geometry/input differs from frozen A0 pair')
                authorize()
                if cuda: torch.cuda.reset_peak_memory_stats(0)
                optimizer.zero_grad(set_to_none=True)
                x,y,f=[batch[n].to(device) for n in ('image','mask','fov')]
                with torch.autocast('cuda' if cuda else 'cpu',dtype=torch.float16 if cuda else torch.bfloat16,enabled=cuda): z=model(x)
                base=weighted_masked_loss(z,y,f,f,epsilon=1e-6)
                terms=soft_cldice_terms(z,y,f,k=k); aux=float(eta)*terms['loss']; loss=base+aux
                require_finite_tensors(loss)
                audit=audit_parameter_gradients(model,base,aux,scaler,budget,track,step_id,device)
                attempt={'step':step_id,'track':track,'pid':os.getpid(),'pair':pair,'A0_loss':float(base.detach()),'soft_cldice_loss':float(terms['loss'].detach()),'weighted_aux_loss':float(aux.detach()),'loss':float(loss.detach()),'gradient_audit':audit,'amp_scale_before':float(scaler.get_scale())}
                append(attempts,attempt)
                if audit['pause_ratio_gt2']: raise GateError('aux/A0 gradient ratio exceeds 2; stop before optimizer call')
                budget.before_gradient(track,step_id)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                require_finite_tensors([p.grad for p in model.parameters() if p.grad is not None])
                total=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                budget.reserve(track,step_id)
                scaler.step(optimizer); scaler.update()
                started=budget.snapshot()['tracks'][track]['pending']['started']
                opt_tensors=[]
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value,torch.Tensor): opt_tensors.append(value)
                if any(not torch.isfinite(p).all() for p in model.parameters()) or any(not torch.isfinite(t).all() for t in opt_tensors) or not math.isfinite(float(scaler.get_scale())):
                    budget.resolve(track,'failed'); raise GateError('nonfinite model/optimizer/scaler state after optimizer call')
                if not started or not probe['called'] or not probe['changed']:
                    budget.resolve(track,'failed'); raise GateError('optimizer did not produce a verified parameter update; no retry')
                budget.resolve(track,'success'); optimizer.zero_grad(set_to_none=True); step+=1
                r=rng_state(device); dg={name:state_digest(val) for name,val in [('model',model.state_dict()),('optimizer',optimizer.state_dict()),('scaler',scaler.state_dict()),('rng',r),('loader_generator',loader_rng.get_state()),('sampling',{'next_draw':step*4,'seed':SEED})]}
                row={**attempt,'step':step,'parameters_updated':True,'preclip_total_L2':float(total),'amp_scale_after':float(scaler.get_scale()),'state_digests':dg,'elapsed_seconds':elapsed0+time.perf_counter()-started_at}
                if cuda:
                    row.update(GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated(0)/1048576,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved(0)/1048576)
                    if torch.cuda.max_memory_reserved(0)>8589934592: raise GateError('8GiB allocator cap exceeded')
                finite_tree(row); append(log,row)
                payload={'schema':1,'phase':'C_SOFT_CLDICE_SCIENTIFIC_2000_V1','track':track,'step':step,'next_draw':step*4,'pid':os.getpid(),'binding_sha256':binding_sha,'eta':eta,'k':k,
                         'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':r,'loader_generator':loader_rng.get_state(),'digests':dg,'initial_state':initial_state,'budget':budget.snapshot(),
                         'train_sha256':sha256_file(out/'train.jsonl'),'attempts_sha256':sha256_file(out/'attempts.jsonl'),'elapsed_seconds':elapsed0+time.perf_counter()-started_at}
                atomic_checkpoint(out/'last.pt',payload,replace=True)
                if step % 200 == 0: atomic_checkpoint(out/f'step_{step:04d}.pt',payload,replace=False)
                del z,base,terms,aux,loss
                if on_update is not None:
                    active['evaluation']=True
                    try: on_update(step,payload)
                    finally: active['evaluation']=False
                del payload
            if step != end: raise GateError(f'segment ended at {step}, expected {end}')
        return {'track':track,'start':start,'end':end,'last_sha256':sha256_file(out/'last.pt'),'budget':budget.snapshot()}
    except BaseException as exc:
        state=budget.snapshot()
        pending=state['tracks'].get(track,{}).get('pending')
        if pending is not None:
            budget.resolve(track,'uncertain' if pending['started'] else 'failed')
        budget.stop(repr(exc)); raise
    finally:
        for handle in handles: handle.remove()


def verify_checkpoint(path, *, expected_step, binding_sha, run_dir, ledger=None):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('phase')!='C_SOFT_CLDICE_SCIENTIFIC_2000_V1' or saved.get('track')!='C' or saved.get('step')!=expected_step or saved.get('next_draw')!=expected_step*4 or saved.get('binding_sha256')!=binding_sha or saved.get('eta')!=ETA or saved.get('k')!=K:
        raise GateError('checkpoint phase/step/cursor/binding/eta/k mismatch')
    b=saved['budget']
    if b.get('charged_events')!=expected_step or b.get('verified_successes')!=expected_step or b.get('failed_events') or b.get('uncertain_events'):
        raise GateError('checkpoint budget does not match successful update step')
    if ledger is not None and b!=ledger.snapshot(): raise GateError('checkpoint budget differs from live ledger')
    if b.get('status')!='READY' or b.get('failed_compute') or b.get('compute_pending') or b.get('forward_calls')!=expected_step or b.get('gradient_calls')!=expected_step*3 or b.get('optimizer_call_starts')!=expected_step: raise GateError('checkpoint has unresolved compute or inconsistent calls')
    if b.get('binding_sha256')!=binding_sha: raise GateError('checkpoint budget source binding differs')
    for name,value in [('model',saved['model']),('optimizer',saved['optimizer']),('scaler',saved['scaler']),('rng',saved['rng']),('loader_generator',saved['loader_generator'])]:
        if state_digest(value)!=saved['digests'].get(name): raise GateError('checkpoint state digest mismatch: '+name)
    for filename,key in [('train.jsonl','train_sha256'),('attempts.jsonl','attempts_sha256')]:
        p=Path(run_dir)/filename; lines=p.read_bytes().splitlines(keepends=True)
        if len(lines)<expected_step: raise GateError('checkpoint log prefix is incomplete: '+filename)
        h=hashlib.sha256(b''.join(lines[:expected_step])).hexdigest()
        if h!=saved.get(key): raise GateError('checkpoint log prefix digest mismatch: '+filename)
    logs=[json.loads(line) for line in (Path(run_dir)/'train.jsonl').read_text().splitlines()[:expected_step]]
    if [r.get('step') for r in logs]!=list(range(1,expected_step+1)) or logs[-1].get('state_digests')!=saved['digests']: raise GateError('checkpoint states differ from committed sequential log')
    if state_digest({'next_draw':expected_step*4,'seed':SEED})!=saved['digests'].get('sampling'): raise GateError('sampling cursor digest mismatch')
    def finite_state(v):
        if isinstance(v,torch.Tensor):
            if not torch.isfinite(v).all(): raise GateError('nonfinite saved checkpoint tensor')
        elif isinstance(v,dict):
            for x in v.values(): finite_state(x)
        elif isinstance(v,(tuple,list)):
            for x in v: finite_state(x)
        elif isinstance(v,float) and not math.isfinite(v): raise GateError('nonfinite saved state scalar')
    for key in ('model','optimizer','scaler'): finite_state(saved[key])
    return saved

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()