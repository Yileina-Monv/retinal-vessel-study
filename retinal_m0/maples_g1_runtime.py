"""Sol-repaired, bound G1-only runner. Default and CPU preflight NEVER launch GPU.

No caller-supplied callback, model, ledger, batch list, device, or release path in
the production CLI. Actual GPU work needs the separate active Sol release.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from retinal_m0.maples_g1_safety import (
    CAP, GPU_UUID, DIGEST_ALGORITHM, GateError, Ledger, atomic_json, canonical_sha,
    exclusive_lock, file_sha, finite_tree, ledger_snapshot, read_json, safe_evidence,
    verify_authority)

ROOT = Path('${RETINA_PROJECT_ROOT}')
REVIEW = ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
OUTPUT = REVIEW/'g1_engineering_v3'
REPAIR = OUTPUT/'sol_repair_20261021'
READY = REPAIR/'ready_binding.json'
RECEIPT = REPAIR/'cpu_review_receipt.json'
RELEASE = OUTPUT/'G1_RELEASE.json'
LEDGER = OUTPUT/'diagnostic_ledger.json'
STATUS = CODE/'STATUS.md'
GPU_LOCK = ROOT/'experiments/a14_gpu0.lock'
RUN_LOCK = OUTPUT/'G1_RUNTIME.lock'
DOC = ROOT/'docs/full_fov_thin_skeleton_bce_20261019/branch_review_20261021'
BASE = ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
CPU = REVIEW/'sol_readiness_review_20261021/attempt_01'
BATCHES = REVIEW/'scientific_review_20261021/attempt_01/G1_fixed_batches.json'
SOURCE = REVIEW/'attempt_02'
V3_SHA = '7333078e1ebafcc169905f32f10e03fa7ef5d503747665afc0e3c16eb2594d0c'
MIN_FREE = 1024**3  # Operational log-space guard, not a scientific hyperparameter.


def probe_resources(*, require_idle=False):
    result = subprocess.run(['nvidia-smi','-i','0',
        '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    fields=[v.strip() for v in result.stdout.strip().split(',')]
    if len(fields)!=5 or fields[0]!='0' or fields[1]!=GPU_UUID:
        raise GateError('physical GPU0 UUID/index mismatch')
    snap={'physical_index':0,'uuid':fields[1],'total_MiB':int(fields[2]),
          'used_MiB':int(fields[3]),'util_percent':int(fields[4]),
          'free_disk_bytes':shutil.disk_usage(OUTPUT).free,
          'observed_at':time.time(),'GPU1_queried':False}
    if snap['free_disk_bytes']<MIN_FREE: raise GateError('insufficient diagnostic log space')
    if require_idle:
        if snap['used_MiB']>=512 or snap['util_percent']>=10:
            raise GateError('GPU0 busy; no process stopped or alternative card selected')
        apps=subprocess.run(['nvidia-smi','-i','0','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        if any(v.strip() for v in apps.stdout.splitlines()): raise GateError('GPU0 compute process exists')
    return snap


def build_binding():
    import torch
    import numpy as np
    from retinal_m0.maples_g1_entry import _verify_sources, _verify_batches, fixed_combinations
    if torch.cuda.is_initialized(): raise GateError('binding/preflight must precede CUDA initialization')
    v3=read_json(DOC/'gates_amendment_v3.json')
    if file_sha(DOC/'gates_amendment_v3.json')!=V3_SHA:
        raise GateError('approved V3 drift; do not freeze a changed policy implicitly')
    expected,count=_verify_sources(CPU/'source_binding.json')
    formal=read_json(ROOT/'experiments/t0_tw_core_20261016/execution_binding.json')
    essentials=['retinal_m0/maples_core.py','retinal_m0/model.py','retinal_data/fives.py',
                'retinal_data/maples_native.py','retinal_data/maples_fov.py','configs/development_v1.json']
    for rel in essentials:
        if file_sha(CODE/rel)!=formal['sources'][rel]: raise GateError('original formal source drift: '+rel)
    config=read_json(BASE/'config.json'); run_binding=read_json(BASE/'run_binding.json')
    if file_sha(BASE/'config.json')!=run_binding['config_sha256']:
        raise GateError('original T0 config drift')
    if file_sha(config['source_m0'])!=config['source_m0_sha256'] or config['source_m0_sha256']!=run_binding['source_m0_sha256']:
        raise GateError('M0 file drift; no checkpoint is loaded by preflight')
    fixed=read_json(BATCHES)['batches']; combos=fixed_combinations(fixed)
    import csv
    with (SOURCE/'images_110.csv').open() as stream: names={r['filename'] for r in csv.DictReader(stream)}
    check=_verify_batches(fixed,read_json(CPU/'actual32_draws.json'),names,expected)
    files=[CODE/rel for rel in essentials]
    files += [CODE/'retinal_m0'/n for n in ['maples_g1_safety.py','maples_g1_diagnostics.py',
               'maples_g1_runtime.py','maples_g1_entry.py','maples_branch_bce.py']]
    files += [CODE/'tests'/n for n in ['test_maples_g1_safety.py','test_maples_g1_diagnostics.py','test_maples_g1_runtime.py']]
    files += [DOC/n for n in ['gates_amendment_v3.json','GATES_AMENDMENT_V3.md',
                              'gates_amendment_v2.json','GATES_AMENDMENT_V2.md']]
    files += [BASE/'config.json',BASE/'run_binding.json',BATCHES,CPU/'source_binding.json',
              CPU/'actual32_draws.json',SOURCE/'artifact_sha256.json']
    policy=v3['inherited_GPU_limits_not_current_permissions']
    if policy['G1_max_diagnostic_forwards']!=CAP or policy['G1_parameter_updates']!=0:
        raise GateError('fixed G1 scope drift')
    binding={'schema':2,'phase':'G1_ONLY','digest_algorithm':DIGEST_ALGORITHM,
        'max_forwards':CAP,'parameter_updates':0,'gpu_uuid':GPU_UUID,
        'required_CUDA_VISIBLE_DEVICES':GPU_UUID,'production_ledger':str(LEDGER.resolve()),
        'runtime_entry':str(Path(__file__).resolve()),
        'source_hashes':{str(p):file_sha(p) for p in files},
        'Train_source_count':count,'fixed_plan':combos,'fixed_batches_sha256':canonical_sha(fixed),
        'M0_path':config['source_m0'],'M0_sha256':config['source_m0_sha256'],
        'versions':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':np.__version__},
        'resources':{'GPU0_idle_memory_below_MiB':512,'GPU0_idle_util_below_percent':10,
                     'minimum_free_disk_bytes':MIN_FREE,'wall_clock_cutoff':None}}
    return binding,check


class InputProvider:
    """Actual bound T0 tensors and native-generated masks; no model execution."""
    def __init__(self):
        from retinal_m0.maples_core import CorePatches
        self.dataset=CorePatches(read_json(BASE/'config.json'),'T0')
        self.artifacts=read_json(SOURCE/'artifact_sha256.json')

    def get(self,batch,method):
        import numpy as np
        import torch
        from retinal_m0.maples_branch_bce import transform_native_target
        items=[];targets=[];actual_inputs=[];actual_targets=[]
        for row in batch['draws']:
            item=self.dataset[int(row['draw'])]
            if item['filename']!=row['filename'] or item['geometry'].tolist()!=row['geometry']:
                raise GateError('actual input filename/geometry mismatch')
            h=hashlib.sha256()
            for key in ('image','mask','fov','geometry'):
                arr=item[key].contiguous().numpy()
                h.update(f'{key}:{arr.dtype}:{arr.shape}'.encode());h.update(arr.tobytes())
            actual=h.hexdigest()
            if actual!=row['input_sha256_original_T0']: raise GateError('actual input tensor fingerprint mismatch')
            rel=f"candidate_masks_NOT_FOR_TRAINING/{row['filename']}.npz"
            if file_sha(SOURCE/rel)!=self.artifacts[rel]: raise GateError('actual native target source mismatch')
            g=row['geometry']
            with np.load(SOURCE/rel,allow_pickle=False) as masks:
                mask=transform_native_target(masks['B1_reference' if method=='B1' else 'B2_candidate'],
                    top=g[0],left=g[1],size=512,hflip=bool(g[2]),vflip=bool(g[3]),quarter_turns=g[4])
            target_sha=hashlib.sha256(mask.tobytes()).hexdigest()
            if target_sha!=row['target_crop_uint8_C_order_sha256'][method] or int(mask.sum())!=row['N_'+method]:
                raise GateError('actual target crop fingerprint/count mismatch')
            if np.any(mask.astype(bool)&~(item['mask'][0].numpy().astype(bool)&item['fov'][0].numpy().astype(bool))):
                raise GateError('actual target outside GT/FOV')
            items.append(item);targets.append(torch.from_numpy(mask.copy()[None]).float())
            actual_inputs.append({'draw':row['draw'],'sha256':actual})
            actual_targets.append({'draw':row['draw'],'sha256':target_sha})
        return {'x':torch.stack([v['image'] for v in items]),'y':torch.stack([v['mask'] for v in items]),
                'fov':torch.stack([v['fov'] for v in items]),'selected':torch.stack(targets),
                'actual_input_digest':canonical_sha(actual_inputs),'actual_target_digest':canonical_sha(actual_targets)}


def capture_rng(model):
    import numpy as np
    import torch
    state={'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state().clone(),'cuda':None}
    if next(model.parameters()).device.type=='cuda': state['cuda']=torch.cuda.get_rng_state(0).clone()
    return state


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None: torch.cuda.set_rng_state(state['cuda'],0)


def _session(model,optimizer,provider,ledger,plan,batches,out,authorize,scaler_factory):
    """Private tested loop. Production binds every dependency, no external callback API."""
    import torch
    from retinal_m0.maples_g1_diagnostics import diagnose,state_digest
    initial=state_digest(model);rng=capture_rng(model);reports=[]
    by_step={int(b['step_original_T0_one_based']):b for b in batches}
    for combo in plan:
        cid=combo['combo_id'];authorize();restore_rng(rng);ledger.reserve(cid)
        report={'combo':combo,'status':'STARTED','events':[],'metrics':None}
        target=out/(cid+'.json');atomic_json(target,report)
        def record(event,values):
            if event=='autograd_call': ledger.autograd(cid)
            report['events'].append({'event':event,'values':safe_evidence(values)})
            if event=='metrics_observed':report['metrics']=safe_evidence(values)
            atomic_json(target,report)
        def pre_hook(*_):
            authorize()  # Revocation/source drift checked before any model body.
            ledger.before_forward(cid)
        hook=model.register_forward_pre_hook(pre_hook)
        original_step=optimizer.step
        optimizer.step=lambda *a,**k:ledger.prohibit_optimizer(cid)
        started=time.perf_counter();cuda=next(model.parameters()).device.type=='cuda'
        try:
            if state_digest(model)!=initial: raise GateError('model differs from original M0 before combination')
            if cuda:torch.cuda.reset_peak_memory_stats(0)
            data=provider.get(by_step[combo['step']],combo['method'])
            for role in ('x','y','fov','selected'): data[role]=data[role].to(next(model.parameters()).device)
            report['actual_input_digest']=data['actual_input_digest'];report['actual_target_digest']=data['actual_target_digest']
            expected_input=canonical_sha([{'draw':r['draw'],'sha256':r['input_sha256_original_T0']} for r in by_step[combo['step']]['draws']])
            expected_target=canonical_sha([{'draw':r['draw'],'sha256':r['target_crop_uint8_C_order_sha256'][combo['method']]} for r in by_step[combo['step']]['draws']])
            if data['actual_input_digest']!=expected_input or data['actual_target_digest']!=expected_target:
                raise GateError('observed input/target digest mismatch')
            metrics=diagnose(model,data['x'],data['y'],data['fov'],data['selected'],
                precision=combo['precision'],scaler=scaler_factory(combo['precision']),record=record)
            finite_tree(metrics)
            if state_digest(model)!=initial: raise GateError('parameter/buffer changed in zero-update G1')
            if optimizer.state: raise GateError('optimizer state changed without permitted updates')
            if cuda:
                torch.cuda.synchronize(0)
                report['peak_allocated_bytes']=int(torch.cuda.max_memory_allocated(0))
                report['peak_reserved_bytes']=int(torch.cuda.max_memory_reserved(0))
            report.update(status='COMPLETED',metrics=metrics,parameter_buffer_sha256=initial)
            ledger.finish(cid);reports.append(report)
        except Exception as exc:
            report.update(status='FAILED',error_type=type(exc).__name__,error=str(exc))
            ledger.finish(cid,error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            hook.remove();optimizer.step=original_step;restore_rng(rng)
            if cuda:
                try:
                    report.setdefault('peak_allocated_bytes',int(torch.cuda.max_memory_allocated(0)))
                    report.setdefault('peak_reserved_bytes',int(torch.cuda.max_memory_reserved(0)))
                except Exception as exc:report['resource_collection_error']=str(exc)
            else:report.update(peak_allocated_bytes=None,peak_reserved_bytes=None,resource_mode='CPU_TEST_ONLY')
            report['elapsed_seconds']=time.perf_counter()-started
            atomic_json(target,safe_evidence(report))
    pairs={}
    for report in reports:
        c=report['combo'];pairs.setdefault((c['step'],c['method']),{})[c['precision']]=report['metrics']
    differences=[]
    for (step,method),values in pairs.items():
        if set(values)!= {'FP32','AMP'}: raise GateError('incomplete actual precision pair')
        differences.append({'step':step,'method':method,'absolute_differences':{
            key:abs(values['FP32'][key]-values['AMP'][key]) for key in
            ('base_loss','aux_unweighted','grad_l2_base','grad_l2_aux_weighted','grad_l2_combined')}})
    result={'status':'COMPLETED_FOR_SOL_REVIEW_NOT_AUTOMATIC_PASS','forwards_limit':CAP,
        'combinations':len(reports),'parameter_updates':0,'parameter_buffer_sha256':initial,
        'gradient_review_trigger':any(r['metrics']['gradient_review_trigger'] for r in reports),
        'combinations_would_clip_fraction':sum(r['metrics'].get('would_clip',False) for r in reports)/len(reports),
        'FP32_vs_AMP':differences,'G2_authorized':False,'scientific_authorized':False}
    atomic_json(out/'summary.json',result)
    return result


def _next_output(prefix):
    REPAIR.mkdir(parents=True,exist_ok=True)
    for i in range(1,10000):
        p=REPAIR/f'{prefix}_{i:02d}'
        try:p.mkdir();return p
        except FileExistsError:pass
    raise GateError('output attempt namespace exhausted')


def cpu_preflight():
    import torch
    if torch.cuda.is_initialized():raise GateError('CPU preflight after CUDA initialization is prohibited')
    out=_next_output('cpu_attempt')
    try:
        binding,checks=build_binding();old=file_sha(LEDGER);state=ledger_snapshot(LEDGER)
        provider=InputProvider();fixed=read_json(BATCHES)['batches']
        for batch in fixed:
            for method in ('B1','B2'):provider.get(batch,method)
        if file_sha(LEDGER)!=old:raise GateError('preflight must not change production accounting')
        snapshot=probe_resources()
        atomic_json(out/'execution_binding.json',binding)
        result={'status':'CPU_PREFLIGHT_PASSED_GPU_HELD','binding_sha256':canonical_sha(binding),
            'binding_file_sha256':file_sha(out/'execution_binding.json'),'digest_algorithm':DIGEST_ALGORITHM,
            'data_checks':checks,'actual_input_batches_methods_rechecked':16,
            'resources':snapshot,'ledger_schema':state['schema'],'ledger_entries':len(state['combinations']),
            'ledger_unchanged':True,'cuda_initialized':torch.cuda.is_initialized(),
            'retinal_model_constructed':0,'model_forward_calls':0,'checkpoint_loads':0,
            'parameter_updates':0,'GPU_launch_authorized':False,'G1_release_issued':False,
            'human_reviewed':0,'human_total':26,'Test_accessed':False}
        atomic_json(out/'verification.json',result)
        return out,result
    except Exception as exc:
        atomic_json(out/'failure.json',{'status':'CPU_PREFLIGHT_FAILED','error':str(exc),'GPU_launch_authorized':False})
        raise


def _authorized_gpu_body(binding,ledger,plan,out,authorize):
    """Called only inside authorization, GPU0 lease, and failure-journal scope."""
    import torch
    ledger.initialize_empty_legacy(out/'legacy_ledger_before_migration.json')
    if torch.cuda.device_count()!=1:raise GateError('exactly one authorized visible CUDA device required')
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import numpy as np
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed(2026)
    from retinal_m0.model import UNet
    from retinal_m0.maples_g1_diagnostics import state_digest
    config=read_json(BASE/'config.json');architecture=read_json(CODE/'configs/development_v1.json')
    model=UNet(architecture['widths'],architecture['group_norm_groups']).cuda().train()
    if file_sha(config['source_m0'])!=binding['M0_sha256']:raise GateError('M0 changed before load')
    checkpoint=torch.load(config['source_m0'],map_location='cpu',weights_only=True)
    if checkpoint['step']!=10000:raise GateError('wrong M0 training step')
    model.load_state_dict(checkpoint['model'],strict=True)
    class SourceState:
        def state_dict(self):return checkpoint['model']
    if state_digest(model)!=state_digest(SourceState()):raise GateError('loaded model differs from M0 tensors')
    del checkpoint
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'],
        betas=tuple(config['betas']),eps=config['eps'],foreach=False)
    def scalers(precision):return torch.amp.GradScaler('cuda',enabled=precision=='AMP',init_scale=config['amp_initial_scale'])
    return _session(model,optimizer,InputProvider(),ledger,plan,read_json(BATCHES)['batches'],out,authorize,scalers)


def run_gpu():
    import torch
    if torch.cuda.is_initialized():raise GateError('authorization must precede CUDA/model creation')
    binding,_=build_binding()
    def authorize():
        # Full raw-data/M0 checks at entry; small bound source/policy checks at every forward.
        for p,v in binding['source_hashes'].items():
            if file_sha(p)!=v:raise GateError('bound implementation/policy/input manifest drift: '+p)
        return verify_authority(status_text=STATUS.read_text(encoding='utf-8'),release_path=RELEASE,
            binding_path=READY,current_binding=binding,ledger_path=LEDGER,review_receipt_path=RECEIPT)
    authorize()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise GateError('mask exactly the physical GPU0 UUID BEFORE process startup; other cards must not be visible')
    with exclusive_lock(RUN_LOCK),exclusive_lock(GPU_LOCK):
        authorize();snapshot=probe_resources(require_idle=True)
        state=ledger_snapshot(LEDGER)
        if state.get('combinations'):raise GateError('prior diagnostic work exists; no automatic restart or reset')
        plan=binding['fixed_plan'];ledger=Ledger(LEDGER,canonical_sha(binding),[c['combo_id'] for c in plan])
        out=_next_output('gpu_attempt');job_started=time.perf_counter()
        atomic_json(out/'launch_binding.json',binding);atomic_json(out/'resources_before.json',snapshot)
        try:
            _authorized_gpu_body(binding,ledger,plan,out,authorize)
        except Exception as exc:
            atomic_json(out/'job_failure.json',{'status':'FAILED_NO_AUTOMATIC_RESTART','error':str(exc),
                'elapsed_seconds':time.perf_counter()-job_started,'G2_authorized':False})
            raise
        finally:
            atomic_json(out/'job_consumption.json',{'elapsed_seconds':time.perf_counter()-job_started,
                'ledger':ledger_snapshot(LEDGER),'GPU1_used':False,'G2_authorized':False})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--cpu-preflight',action='store_true')
    modes.add_argument('--run-g1',action='store_true')
    args=parser.parse_args(argv)
    if args.cpu_preflight:
        out,result=cpu_preflight();print(json.dumps({'out':str(out),**result},ensure_ascii=False))
    else:run_gpu()


if __name__=='__main__':main()
