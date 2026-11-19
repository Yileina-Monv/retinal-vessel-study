"""CPU-only fixed eight-batch candidate audit. No optimizer, inference, or GPU CLI."""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import argparse
import contextlib
import hashlib
import json
import math
import random
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from retinal_m0.maples_core import CorePatches, weighted_masked_loss
from retinal_m0.maples_focal_dice import focal_dice_loss, focal_dice_terms
from retinal_m0.model import UNet
from retinal_m0.state import state_digest

ROOT = Path('${RETINA_PROJECT_ROOT}')
CODE = ROOT / 'code/retinal-vessel-study'
REVIEW = ROOT / 'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1'
OUT = REVIEW / 'focal_dice_cpu_v1_20261024'
FIXED = REVIEW / 'scientific_review_20261021/attempt_01/G1_fixed_batches.json'
BASE = ROOT / 'experiments/t0_tw_core_20261016/runs/seed_2026/T0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''): h.update(data)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.pending')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def rng():
    n = np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),*n[2:]],'torch':torch.get_rng_state()}


def norm(values):
    return math.sqrt(sum(float(x.detach().double().square().sum()) for x in values))


def dot(a, b):
    return sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(a,b))


@contextlib.contextmanager
def forbid_updates():
    patched=[]
    def reject(*args, **kwargs): raise RuntimeError('CPU_ONLY_PARAMETER_UPDATES_FORBIDDEN')
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer) and 'step' in cls.__dict__:
            patched.append((cls,cls.__dict__['step'])); cls.step=reject
    try: yield
    finally:
        for cls,step in patched: cls.step=step


def guard_data(dataset, config):
    prefix = Path(config['data_root']).resolve()
    allowed = {str(prefix/name) for name in ['data_manifest.csv','split_record.json']}
    for row in dataset.native.rows:
        for key in ['image_path','label_vessels','label_hemorrhages',
                    'label_brightuncertains','label_reduncertains']:
            allowed.add(str((prefix/row[key]).resolve()))
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if str(path).startswith(str(prefix)+'/') and str(path) not in allowed:
                raise RuntimeError('NON_TRAIN_CONTENT_FORBIDDEN:'+str(path))
    sys.addaudithook(audit)


def run():
    auth=read_json(OUT/'CPU_AUTHORIZATION.json')
    expected={'phase':'FOCAL_DICE_CPU_ONLY_V1','user_approved':True,'GPU_allowed':False,
              'parameter_updates':0,'max_M0_forwards':8,'max_parameter_autograd_calls':32}
    if any(auth.get(k)!=v for k,v in expected.items()): raise RuntimeError('CPU scope binding mismatch')
    binding=read_json(OUT/'CPU_BINDING.json')
    for path,digest in binding['source_sha256'].items():
        if sha(path)!=digest: raise RuntimeError('CPU source binding drift:'+path)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise RuntimeError('GPU must remain invisible')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    # Same established reference backend as C; do not retry with changed tolerances.
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026)
    attempt=OUT/'attempt_01';attempt.mkdir(exist_ok=False)
    ledger={'status':'RUNNING_CPU_ONLY','max_forwards':8,'max_gradients':32,
            'forwards':0,'gradients':0,'completed_batches':0,'failed_batches':0,
            'pending':None,'parameter_updates':0,'GPU_calls':0}
    started=time.monotonic();records=[];sources=dict(binding['source_sha256'])
    def check_resource():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>auth['max_RSS_KiB']:
            raise RuntimeError('CPU memory cap exceeded')
    def timeout(*_): raise TimeoutError('CPU review wall-time limit reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(auth['max_wall_seconds'])
    atomic_json(attempt/'cpu_ledger.json',ledger)
    try:
        config=read_json(BASE/'config.json'); architecture=read_json(CODE/'configs/development_v1.json')
        if config['seed']!=2026 or config['batch_size']!=4 or config['patch']!=512:
            raise RuntimeError('original T0 setting mismatch')
        if sha(config['source_m0'])!=config['source_m0_sha256']:
            raise RuntimeError('M0 provenance mismatch')
        dataset=CorePatches(config,'T0');guard_data(dataset,config)
        train=read_json(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json')
        assert train['count']==110 and [r['filename'] for r in train['rows']]==dataset.names
        fixed=read_json(FIXED)['batches'];assert len(fixed)==8
        references={r['step']:r for r in map(json.loads,(BASE/'pairing.jsonl').read_text().splitlines())}
        batches=[];used=set()
        for spec in fixed:
            items=[dataset[int(r['draw'])] for r in spec['draws']]
            original=references[spec['step_original_T0_one_based']]
            for slot,(item,ref) in enumerate(zip(items,spec['draws'])):
                assert item['filename']==ref['filename'] and item['geometry'].tolist()==ref['geometry']
                assert item['input_sha256']==ref['input_sha256_original_T0']==original['input_sha256'][slot]
                used.add(item['filename'])
            batches.append((spec,default_collate(items)))
        for row in dataset.native.rows:
            if row['filename'] not in used: continue
            for key in ['image_path','label_vessels','label_hemorrhages','label_brightuncertains','label_reduncertains']:
                path=Path(config['data_root'])/row[key];sources[str(path)]=sha(path)
        for name in ['data_manifest.csv','split_record.json']:
            path=Path(config['data_root'])/name;sources[str(path)]=sha(path)
        dataset.cache.clear()
        model=UNet(architecture['widths'],architecture['group_norm_groups']).cpu().train()
        saved=torch.load(config['source_m0'],map_location='cpu',weights_only=True)
        assert saved['step']==10000;model.load_state_dict(saved['model'],strict=True);del saved
        parameters=tuple(model.parameters());names=[name for name,_ in model.named_parameters()]
        original_model=state_digest(model.state_dict());original_rng=state_digest(rng())
        for j,(spec,batch) in enumerate(batches):
            ledger['pending']={'batch':j,'original_T0_step':spec['step_original_T0_one_based']}
            if ledger['forwards']>=8: raise RuntimeError('M0 forward cap exhausted')
            ledger['forwards']+=1;atomic_json(attempt/'cpu_ledger.json',ledger)
            stamp=time.monotonic();z=model(batch['image']);forward_seconds=time.monotonic()-stamp
            base=weighted_masked_loss(z,batch['mask'],batch['fov'],batch['fov'])
            zero=focal_dice_loss(z,batch['mask'],batch['fov'],gamma=0)
            t0=time.monotonic();terms=focal_dice_terms(z,batch['mask'],batch['fov'],gamma=1)
            loss_seconds=time.monotonic()-t0;new=terms['loss']
            def gradient(loss, retain=True):
                if ledger['gradients']>=32:raise RuntimeError('parameter gradient cap exhausted')
                ledger['gradients']+=1;atomic_json(attempt/'cpu_ledger.json',ledger)
                result=torch.autograd.grad(loss,parameters,retain_graph=retain,allow_unused=False)
                if any(not torch.isfinite(v).all() for v in result):raise RuntimeError('nonfinite true parameter gradient')
                return result
            g0=gradient(base);gz=gradient(zero)
            if not torch.equal(base,zero) or not all(torch.equal(a,b) for a,b in zip(g0,gz)):
                raise RuntimeError('gamma-zero A0 value or parameter-gradient identity failed')
            del gz
            g1=gradient(new);gd=gradient(new-base,retain=False)
            n0,n1,nd=norm(g0),norm(g1),norm(gd)
            if n0<=0 or n1<=0:raise RuntimeError('missing nonzero A0/candidate gradient')
            residual=norm(tuple(d-(b-a) for a,b,d in zip(g0,g1,gd)))
            relative=residual/max(n1,n0,1e-30)
            if relative>1e-4:raise RuntimeError('gradient difference/additivity mismatch')
            ratio=nd/n0
            layers={name:{'A0_L2':norm((a,)),'candidate_L2':norm((n,)),
                          'delta_L2':norm((d,)),'A0_dot_delta':dot((a,),(d,))}
                    for name,a,n,d in zip(names,g0,g1,gd)}
            row={'batch_index':j,'original_T0_step':spec['step_original_T0_one_based'],
                 'gamma':1,'draws':batch['draw'].tolist(),'filenames':list(batch['filename']),
                 'geometry':batch['geometry'].tolist(),'input_sha256':list(batch['input_sha256']),
                 'loss_A0':float(base.detach()),'loss_candidate':float(new.detach()),
                 'per_crop':{k:v.detach().tolist() for k,v in terms.items() if k!='loss'},
                 'A0_parameter_L2':n0,'candidate_parameter_L2':n1,'delta_parameter_L2':nd,
                 'candidate_to_A0':n1/n0,'delta_to_A0':ratio,
                 'cosine_A0_candidate':dot(g0,g1)/(n0*n1),
                 'cosine_A0_delta':dot(g0,gd)/(n0*nd) if nd else None,
                 'gamma_zero_exact_value_and_parameter_gradient':True,
                 'parameter_gradients_unscaled_preclip':True,'difference_additivity_relative_error':relative,
                 'per_layer':layers,'forward_seconds':forward_seconds,'loss_seconds':loss_seconds,
                 'batch_seconds':time.monotonic()-stamp,
                 'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            write_new(attempt/f'batch_{j:02d}.json',row);records.append(row)
            if ratio>2:raise RuntimeError('delta/A0 parameter-gradient ratio > 2; review required')
            if any(v.grad is not None for v in parameters):raise RuntimeError('unexpected accumulated parameter gradients')
            if state_digest(model.state_dict())!=original_model or state_digest(rng())!=original_rng:
                raise RuntimeError('model or CPU RNG changed during gradient-only audit')
            ledger['completed_batches']+=1;ledger['pending']=None;atomic_json(attempt/'cpu_ledger.json',ledger)
            check_resource()
            print('BATCH_COMPLETE',j+1,'/8','candidate/A0',n1/n0,'delta/A0',ratio,flush=True)
            del z,base,zero,terms,new,g0,g1,gd
        for path,digest in sources.items():
            if sha(path)!=digest:raise RuntimeError('input/source changed:'+path)
        write_new(attempt/'source_manifest.json',sources)
        summary={'status':'PASSED_CPU_ONLY_GPU_NOT_RELEASED','gamma':1,
                 'verified_batches':8,'actual_M0_forwards':ledger['forwards'],
                 'actual_parameter_autograd_calls':ledger['gradients'],
                 'parameter_updates':0,'GPU_calls':0,'Dev_or_Test_content_read':False,
                 'Train_only_data_open_guard_active':True,'M0_and_RNG_unchanged':True,
                 'sources_unchanged':len(sources),'max_delta_to_A0':max(r['delta_to_A0'] for r in records),
                 'candidate_to_A0_range':[min(r['candidate_to_A0'] for r in records),max(r['candidate_to_A0'] for r in records)],
                 'cosine_A0_candidate_range':[min(r['cosine_A0_candidate'] for r in records),max(r['cosine_A0_candidate'] for r in records)],
                 'max_difference_additivity_relative_error':max(r['difference_additivity_relative_error'] for r in records),
                 'backend':'ATen CPU FP32, MKLDNN disabled; not a change to future GPU settings',
                 'CPU_threads':2,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'elapsed_seconds':time.monotonic()-started,'CUDA_AMP_verified':False,
                 'updated_trajectory_resume_verified':False,'GPU_training_authorized':False,
                 'interpretation':'Train-M0 engineering evidence only; no method-efficacy conclusion'}
        write_new(attempt/'completion.json',summary)
        ledger['status']='COMPLETE_CPU_ONLY';atomic_json(attempt/'cpu_ledger.json',ledger)
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        ledger['status']='FAILED_STOP';ledger['failed_batches']+=int(ledger['pending'] is not None)
        ledger['exception']=repr(exc);atomic_json(attempt/'cpu_ledger.json',ledger)
        write_new(attempt/'failure.json',{'exception':repr(exc),'ledger':ledger,'elapsed_seconds':time.monotonic()-started})
        raise
    finally: signal.alarm(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-cpu-review', action='store_true')
    args=parser.parse_args()
    if not args.run_cpu_review:parser.error('explicit --run-cpu-review required; no GPU modes exist')
    with forbid_updates():run()


if __name__=='__main__': main()
