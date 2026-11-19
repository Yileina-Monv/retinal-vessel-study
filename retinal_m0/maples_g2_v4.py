"""P1 G2 lambda=.03 revision: explicit inherited accounting, no science entry."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from retinal_m0 import maples_g2_engineering as old, maples_g2_runtime as previous
from retinal_m0.maples_g2_engineering import GateError,read_json,file_sha,atomic_json,canonical_sha,lock

ROOT=previous.ROOT;CODE=previous.CODE;BASE=previous.BASE
OUT=previous.G2OUT/'lambda003_revision_v4'
POLICY=previous.DOC/'g2_lambda003_revision_v4.json'
READY=OUT/'ready_binding.json';RECEIPT=OUT/'cpu_review_receipt.json';RELEASE=OUT/'G2_RELEASE.json'
LEDGER=OUT/'g2_budget_ledger.json';STATUS=previous.STATUS
GPU_UUID=previous.GPU_UUID
PARENT_SHA='2b2f5bbad66dbbd2ee5507030b9c524ae284260715eea25acaf05eba11242e50'
REVISION='P1_G2_LAMBDA003_V4_20261022'
LAMBDA=.03;AUDIT_STEPS={1,5,10,25,45}
CAPS={'B1_CONT':45,'B1_SPLIT':45,'B2_CONT':45,'B2_SPLIT':45}
COUNTERS=('charged_events','verified_successes','attempts','amp_retries','failed_events','uncertain_events','optimizer_events')

class Budget(old.Budget):
    """One active revision ledger carries the immutable previous209, never refunds."""
    def _validate(self,s):
        if s.get('schema')!=4 or s.get('binding_sha256')!=self.binding_sha or s.get('track_caps')!=CAPS or s.get('global_cap')!=400 or s.get('planned_ceiling')!=389:raise GateError('V4 accounting identity mismatch')
        parent=s.get('inherited_ledger',{})
        if s.get('parent_sha256')!=PARENT_SHA or canonical_sha(parent)!=s.get('parent_canonical_sha256'):raise GateError('inherited accounting drift')
        if parent!=check_parent():raise GateError('inherited ledger differs from immutable original')
        if parent.get('charged_events')!=209 or parent.get('verified_successes')!=209 or parent.get('forward_calls')!=210 or parent.get('gradient_calls')!=216:raise GateError('inherited209 missing')
        totals={k:parent[k] for k in COUNTERS}
        for name,t in s['tracks'].items():
            if name not in CAPS or any(type(t.get(k))is not int or t[k]<0 for k in COUNTERS):raise GateError('invalid V4 track/counter')
            if t['charged_events']>45 or t['charged_events']!=t['verified_successes']+t['failed_events']+t['uncertain_events'] or t['optimizer_events']!=t['verified_successes']+t['failed_events']:raise GateError('V4 event total mismatch')
            for k in COUNTERS:totals[k]+=t[k]
        if any(type(s.get(k))is not int or s[k]!=totals[k] for k in COUNTERS):raise GateError('global inherited counter mismatch')
        if s['charged_events']>389 or s['charged_events']>400:raise GateError('V4 budget exhausted')
        if type(s.get('forward_calls'))is not int or type(s.get('gradient_calls'))is not int or not 210<=s['forward_calls']<=3600 or not 216<=s['gradient_calls']<=3960:raise GateError('old compute consumption lost or ceiling exceeded')
        if sum(t.get('pending') is not None for t in s['tracks'].values())>1:raise GateError('multiple pending optimizer events')
        return s
    def _read(self):
        if not self.path.is_file():raise GateError('V4 ledger missing; no reset')
        return self._validate(read_json(self.path))
    def _mutate(self,fn):
        with lock(self.lock_path):
            s=self._read();fn(s);self._validate(s);atomic_json(self.path,s)
    def initialize_migration(self,parent):
        with lock(self.lock_path):
            if self.path.exists():raise GateError('migration already exists; no reset')
            if any(t.get('pending') for t in parent.get('tracks',{}).values()):raise GateError('unresolved parent optimizer event')
            s={'schema':4,'binding_sha256':self.binding_sha,'global_cap':400,'planned_ceiling':389,'track_caps':CAPS,
               'parent_sha256':PARENT_SHA,'parent_canonical_sha256':canonical_sha(parent),'inherited_ledger':parent,
               'inherited_compute_adjudication':'Old step10 counted forward/gradients, no optimizer invocation; parent bytes retained unchanged.',
               'tracks':{},'status':'READY','forward_calls':parent['forward_calls'],'gradient_calls':parent['gradient_calls'],
               'compute_pending':None,'require_compute':True,**{k:parent[k] for k in COUNTERS}}
            self._validate(s);atomic_json(self.path,s)
    def reserve(self,track,planned_step,attempt):
        def f(s):
            if s['status']!='READY' or track not in CAPS:raise GateError('V4 stopped/unknown track')
            if any(t.get('pending') is not None for t in s['tracks'].values()):raise GateError('unresolved optimizer event')
            t=s['tracks'].setdefault(track,{**{k:0 for k in COUNTERS},'pending':None,'retry_cursor':None})
            if s.get('compute_pending')!={'track':track,'step':planned_step,'attempt':attempt}:raise GateError('update without counted forward')
            if t['charged_events']>=45 or s['charged_events']>=389:raise GateError('V4 budget exhausted')
            cursor=t.get('retry_cursor')
            if planned_step!=t['verified_successes']+1 or not 1<=attempt<=9 or (cursor is None and attempt!=1) or (cursor is not None and cursor!={'planned_step':planned_step,'next_attempt':attempt}):raise GateError('V4 draw/retry mismatch')
            s['attempts']+=1;t['attempts']+=1;t['pending']={'planned_step':planned_step,'attempt':attempt,'reserved_at':time.time(),'outcome':'UNKNOWN'}
        self._mutate(f)
    def before_forward(self,track,step,attempt):
        def f(s):
            if s['status']!='READY' or s.get('compute_pending') is not None or any(t.get('pending') for t in s['tracks'].values()):raise GateError('unresolved compute/update or stopped ledger')
            if track not in CAPS or type(step)is not int or type(attempt)is not int or not 1<=attempt<=9:raise GateError('invalid V4 compute identity')
            t=s['tracks'].get(track,{'verified_successes':0,'charged_events':0,'retry_cursor':None})
            cursor=t.get('retry_cursor')
            if step!=t['verified_successes']+1 or t['charged_events']>=45 or s['charged_events']>=389 or (cursor is None and attempt!=1) or (cursor is not None and cursor!={'planned_step':step,'next_attempt':attempt}):raise GateError('V4 compute exceeds budget or retry drift')
            if s['forward_calls']>=3600:raise GateError('cumulative forward ceiling')
            s['forward_calls']+=1;s['current_gradient_calls']=0;s['compute_pending']={'track':track,'step':step,'attempt':attempt}
        self._mutate(f)
    def before_gradient(self,track,step,attempt):
        def f(s):
            if s['status']!='READY' or s.get('compute_pending')!={'track':track,'step':step,'attempt':attempt}:raise GateError('gradient identity mismatch')
            if s.get('current_gradient_calls',0)>=(3 if step in AUDIT_STEPS else 1) or s['gradient_calls']>=3960:raise GateError('gradient ceiling')
            s['gradient_calls']+=1;s['current_gradient_calls']=s.get('current_gradient_calls',0)+1
        self._mutate(f)

def gradient_audit(model,logits,base_loss,weighted_aux_loss,*,scaler,audit_step,before_gradient=lambda:None):
    import math
    import torch
    if audit_step not in AUDIT_STEPS:raise GateError('unscheduled V4 audit')
    ps=tuple(p for p in model.parameters() if p.requires_grad)
    rng=old._rng_snapshot(logits.device);model_hash=old._tensor_digest(model);scale_state=scaler.state_dict()
    grads=[None if p.grad is None else p.grad.clone() for p in ps];scale=float(scaler.get_scale())
    def get(loss):
        before_gradient()
        return tuple(g.detach().float()/scale for g in torch.autograd.grad(scaler.scale(loss),ps,retain_graph=True,allow_unused=False))
    a,b=get(base_loss),get(weighted_aux_loss)
    n0=math.sqrt(sum(float(g.double().square().sum()) for g in a));n1=math.sqrt(sum(float(g.double().square().sum()) for g in b))
    dot=sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
    ratio=n1/n0 if n0 else None;c=dot/(n0*n1) if n0 and n1 else None
    result={'audit_step':audit_step,'base_grad_l2':n0,'weighted_aux_grad_l2':n1,'aux_to_base_l2':ratio,'cosine':c,
        'pause_ratio_gt_2':bool((ratio is not None and ratio>2) or (n0==0 and n1>0)),
        'zero_base_positive_aux':bool(n0==0 and n1>0),'negative_cosine':bool(c is not None and c<0),'unscaled':True,'autograd_calls':2}
    old._tree_finite(result)
    if old._tensor_digest(model)!=model_hash or scaler.state_dict()!=scale_state or not old._rng_equal(rng,old._rng_snapshot(logits.device)):raise GateError('V4 audit state side effect')
    if any((g is None)!=(p.grad is None) or (g is not None and not torch.equal(g,p.grad)) for g,p in zip(grads,ps)):raise GateError('V4 audit changed p.grad')
    return result

def check_parent():
    if file_sha(previous.G2_LEDGER)!=PARENT_SHA:raise GateError('old209 ledger drift')
    parent=read_json(previous.G2_LEDGER)
    if parent['status']!='FAILED_STOP' or parent['charged_events']!=209 or parent['uncertain_events']!=0 or any(t['pending'] for t in parent['tracks'].values()):raise GateError('parent not reconciled')
    return parent

def build_binding():
    import torch
    if torch.cuda.is_initialized():raise GateError('pre-CUDA binding required')
    policy=read_json(POLICY)
    if policy['revision_id']!=REVISION or policy['lambda']!=LAMBDA or policy['planned_total']!=389 or policy['parent_ledger_sha256']!=PARENT_SHA:raise GateError('V4 policy mismatch')
    check_parent()
    inherited=previous.build_binding()
    if canonical_sha(inherited)!='2d4a7bd0c9ad09a8bbe5dcf02492714d94956aeff19488316dcc80141f7022f6':raise GateError('original accepted A0/source binding changed')
    sources={**inherited['source_hashes']}
    extra=[POLICY,previous.G2_LEDGER,previous.G2OUT/'matrix_failure.json',previous.G2OUT/'runs/accept_segment_00.json',
        previous.G2OUT/'runs/R0_LAMBDA0/step_000200.pt',previous.G2OUT/'runs/R0_LAMBDA0/train.jsonl',
        previous.G2OUT/'runs/R0_LAMBDA0/step_000001.pt',previous.G2OUT/'runs/R0_LAMBDA0/segment_000_200.json',
        CODE/'retinal_m0/maples_g2_v4.py',CODE/'retinal_m0/maples_g2_v4_worker.py',CODE/'tests/test_maples_g2_v4.py']
    for p in extra:
        if not p.is_file() or p.is_symlink():raise GateError('missing/symlink V4 source')
        sources[str(p)]=file_sha(p)
    return {'schema':4,'revision_id':REVISION,'phase':'G2_ONLY','lambda':LAMBDA,'sources':sources,
        'policy_sha256':file_sha(POLICY),'parent_ledger_sha256':PARENT_SHA,'global_cap':400,'planned_total':389,'additional_updates':180,
        'track_caps':CAPS,'gpu_uuid':GPU_UUID,'ledger_path':str(LEDGER),'versions':inherited['versions'],'M0_path':inherited['M0_path'],
        'M0_sha256':inherited['M0_sha256'],'R0_reused':True,'science_authorized':False}

def active():
    state=previous.current_g2_state(STATUS.read_text())
    if state.get('revision_id')!=REVISION or state.get('planned_total_charged')!=389:raise GateError('active V4 revision mismatch')
    return state

def verify_release(*,full=False):
    s=active()
    if s['state']!='RELEASED' or s.get('GPU_released') is not True:raise GateError('V4 GPU not released')
    for p in [READY,RECEIPT,RELEASE]:
        if not p.is_file() or p.is_symlink():raise GateError('V4 release artifact missing')
    b=read_json(READY);r=read_json(RELEASE);c=read_json(RECEIPT);h=canonical_sha(b)
    if b.get('schema')!=4 or b.get('revision_id')!=REVISION or b.get('phase')!='G2_ONLY' or b.get('lambda')!=LAMBDA or b.get('global_cap')!=400 or b.get('planned_total')!=389 or b.get('ledger_path')!=str(LEDGER) or b.get('gpu_uuid')!=GPU_UUID:raise GateError('V4 binding scope mismatch')
    if s.get('release_file_sha256')!=file_sha(RELEASE) or s.get('release_id')!=r.get('release_id') or s.get('binding_sha256')!=h:raise GateError('STATUS/release binding mismatch')
    if r.get('phase')!='G2_ONLY' or r.get('revision_id')!=REVISION or r.get('scientific_authorized') is not False or r.get('binding_sha256')!=h or r.get('binding_file_sha256')!=file_sha(READY) or r.get('cpu_receipt_sha256')!=file_sha(RECEIPT) or r.get('review_owner')!='Sol':raise GateError('release scope/evidence mismatch')
    if r.get('charged_update_ceiling')!=400 or r.get('planned_total')!=389 or r.get('lambda')!=LAMBDA:raise GateError('release budget/lambda mismatch')
    if c.get('all_passed') is not True or c.get('binding_sha256')!=h or c.get('R0_reuse_verified') is not True or not c.get('evidence_hashes'):raise GateError('CPU acceptance absent')
    for p,sha in c['evidence_hashes'].items():
        if not Path(p).resolve().is_relative_to(OUT.resolve()) or file_sha(p)!=sha:raise GateError('CPU evidence drift')
    for p,sha in b['sources'].items():
        if full or Path(p).stat().st_size<1048576:
            if file_sha(p)!=sha:raise GateError('V4 bound source drift')
    check_parent()
    if full and build_binding()!=b:raise GateError('current V4 binding mismatch')
    return b

def cpu_preflight():
    import torch
    if torch.cuda.is_initialized():raise GateError('CPU preflight after CUDA prohibited')
    if OUT.exists():raise GateError('V4 production stage already exists; no reset')
    OUT.mkdir()
    binding=build_binding();atomic_json(READY,binding)
    # Read-only full tensor/state and original first200 log replay comparison.
    verify_R0_read_only()
    budget=Budget(LEDGER,canonical_sha(binding));budget.initialize_migration(check_parent())
    result={'status':'V4_CPU_PREFLIGHT_PASSED_GPU_HELD','binding_sha256':canonical_sha(binding),'R0_reuse_verified':True,
        'inherited_charged':209,'global_cap':400,'planned_total':389,'additional_GPU_updates':0,
        'CUDA_initialized':torch.cuda.is_initialized(),'retinal_model_constructed':False,'resources':previous.observe_resources()}
    atomic_json(OUT/'preflight.json',result);print(json.dumps(result,indent=2))

def verify_R0_read_only():
    import torch
    from retinal_m0.maples_g2_worker import compare_checkpoint
    r0=previous.G2OUT/'runs/R0_LAMBDA0'
    original=torch.load(BASE/'step_000200.pt',map_location='cpu',weights_only=True)
    replay=torch.load(r0/'step_000200.pt',map_location='cpu',weights_only=True)
    compare_checkpoint(replay,original,loader=False,initial=False)
    for k in ('model','optimizer','scaler','rng'):
        if hashlib.sha256(replay['initial'][k].encode()).hexdigest()!=original['initial_'+k]:raise GateError('R0 initial state mismatch')
    old_rows=[json.loads(x) for x in (BASE/'train.jsonl').read_text().splitlines()][:200]
    replay_rows=[json.loads(x) for x in (r0/'train.jsonl').read_text().splitlines()]
    pairs=[json.loads(x) for x in (BASE/'pairing.jsonl').read_text().splitlines()][:200]
    if len(old_rows)!=200 or len(replay_rows)!=200:raise GateError('R0 rows incomplete')
    for a,b,p in zip(replay_rows,old_rows,pairs):
        if a['pair']!=p or any(a[k]!=b[k] for k in ('step','loss','gradient_norm','amp_retries','amp_scale')):raise GateError('R0 complete numerical/input replay mismatch')
    receipt=read_json(previous.G2OUT/'runs/accept_segment_00.json')
    if receipt.get('passed') is not True or receipt['checkpoint_sha256']!=file_sha(r0/'step_000200.pt'):raise GateError('R0 old receipt drift')
    return True

def run():
    import secrets
    from retinal_m0.maples_g2_v4_worker import SEGMENTS,validate_segment
    b=verify_release(full=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise GateError('only GPU0 UUID may be visible')
    with lock(previous.RUN_LOCK),lock(OUT/'V4_RUNTIME.lock'):
        budget=Budget(LEDGER,canonical_sha(b));s=budget.snapshot()
        if s['charged_events']!=209 or s['tracks'] or s['status']!='READY' or (OUT/'matrix_launch.json').exists():raise GateError('prior V4 work exists; no automatic restart')
        started=time.perf_counter()
        try:
            for index,segment in enumerate(SEGMENTS):
                verify_release(full=True);token=secrets.token_hex(32)
                atomic_json(OUT/'matrix_launch.json',{'segment_index':index,'parent_pid':os.getpid(),'token_sha256':hashlib.sha256(token.encode()).hexdigest()})
                env=dict(os.environ,RETINA_G2_V4_TOKEN=token,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
                with (OUT/f'worker_{index:02d}.log').open('x') as f:
                    subprocess.run([sys.executable,'-B','-m','retinal_m0.maples_g2_v4','--segment',str(index)],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                validate_segment(OUT/'runs',index,canonical_sha(b))
            s=budget.snapshot()
            if s['charged_events']!=389 or s['verified_successes']!=389 or s['failed_events'] or s['uncertain_events'] or s.get('compute_pending') or any(t['pending'] for t in s['tracks'].values()):raise GateError('V4 final accounting mismatch')
            atomic_json(OUT/'completion.json',{'status':'G2_V4_ENGINEERING_PASSED_FOR_SOL_REVIEW','ledger':s,'elapsed_seconds':time.perf_counter()-started,'scientific_authorized':False})
        except BaseException as e:
            for track,t in budget.snapshot()['tracks'].items():
                if t['pending']:budget.mark_crash_uncertain(track)
            budget.stop(str(e));atomic_json(OUT/'failure.json',{'error':str(e),'ledger':budget.snapshot()});raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-preflight',action='store_true');mode.add_argument('--run-g2',action='store_true');mode.add_argument('--segment',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.cpu_preflight:cpu_preflight()
    elif args.segment is not None:
        from retinal_m0.maples_g2_v4_worker import gpu_worker
        gpu_worker(args.segment)
    else:run()

if __name__=='__main__':main()
