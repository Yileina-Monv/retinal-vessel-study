"""CPU-only audit of the capped nnU-Net engineering artifacts."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

ROOT=Path('${RETINA_PROJECT_ROOT}')
EXP=ROOT/'experiments/nnunet2d_branch_20261020'
ENG=EXP/'engineering'
ATT=ENG/'attempts'

def equal(a,b,path='root',diffs=None):
    diffs=diffs if diffs is not None else []
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)):
            diffs.append(path); return False
        return True
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        if not (isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and np.array_equal(a,b)):
            diffs.append(path); return False
        return True
    if isinstance(a,dict) or isinstance(b,dict):
        if not (isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b)):
            diffs.append(path+' keys'); return False
        return all(equal(a[k],b[k],f'{path}.{k}',diffs) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        if not (isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b)):
            diffs.append(path+' shape'); return False
        return all(equal(x,y,f'{path}[{i}]',diffs) for i,(x,y) in enumerate(zip(a,b)))
    if a!=b: diffs.append(path); return False
    return True

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ledger=json.loads((ENG/'budget_ledger.json').read_text(encoding='utf-8'))
assert ledger['status']=='GPU_ENGINEERING_PHASES_COMPLETE_NEEDS_STATE_COMPARISON'
assert ledger['attempts_used']==100 and ledger['gpu_window_elapsed_seconds']<1800
assert all(p['return_code']==0 and not p['timed_out'] for p in ledger['phases'])
assert ledger['official_test_accessed'] is False and ledger['dev_model_calls']==0 and ledger['run_training_called'] is False

phase_names=('train50','fixture25','split12','resume13')
phase_reports={}
all_events=[]
for name in phase_names:
    d=ATT/name
    summary=json.loads((d/'worker_summary.json').read_text(encoding='utf-8'))
    events=[json.loads(x) for x in (d/'attempts.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    starts=[e for e in events if e['event']=='optimizer_attempt_start']
    successes=[e for e in events if e['event']=='optimizer_attempt_success']
    failures=[e for e in events if e['event'] in ('optimizer_attempt_failed','optimizer_attempt_exception')]
    assert len(starts)==len(successes)==summary['optimizer_attempts'] and not failures
    assert summary['successful_updates']==len(successes) and summary['amp_skips']==0
    assert all(s['loss_finite'] and s['gradient_finite'] and not s['amp_skipped'] for s in successes)
    assert all(s['input_shape']==[2,3,1024,1536] and s['network_output_shapes']==s['target_shapes'] if False else True for s in [])
    for s in successes:
        assert s['input_shape']==[2,3,1024,1536]
        assert len(s['network_output_shapes'])==8 and len(s['target_shapes'])==8
        assert [x[2:] for x in s['network_output_shapes']]==[x[2:] for x in s['target_shapes']]
    phase_reports[name]={'attempts':len(starts),'successful_updates':len(successes),'amp_skips':summary['amp_skips'],
      'elapsed_seconds':ledger['phases'][phase_names.index(name)]['elapsed_seconds'],
      'checkpoint_sha256':sha(summary['checkpoint_out']),
      'loss_min':min(s['loss'] for s in successes),'loss_max':max(s['loss'] for s in successes),
      'gradient_norm_preclip_min':min(s['gradient_norm_before_clip'] for s in successes),
      'gradient_norm_preclip_max':max(s['gradient_norm_before_clip'] for s in successes)}
    all_events.extend(successes)

base_summary=json.loads((ATT/'train50/worker_summary.json').read_text(encoding='utf-8'))
cov=base_summary['native_size_coverage']
assert len(cov)==3 and min(cov.values())>0, cov
manifest=json.loads((ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json').read_text(encoding='utf-8'))
sizes={r['filename']:tuple(r['native_shape']) for r in manifest['rows']}
observed={k for s in base_summary['steps'] for k in s['case_ids']}
observed_sizes={sizes[k] for k in observed}
assert observed_sizes==set(sizes.values())
step_times=np.array([s['elapsed_seconds'] for s in base_summary['steps']],dtype=float)
step_intervals=np.diff(np.r_[0.0,step_times])
median_update_seconds=float(np.median(step_intervals))
projected_25000_updates_hours=25000*median_update_seconds/3600

# Compare continuous 25 fixed-input updates against independent 12+13 restored updates.
cont=torch.load(ATT/'fixture25/checkpoint_continuous25.pth',map_location='cpu',weights_only=False)
split=torch.load(ATT/'resume13/checkpoint_resumed25.pth',map_location='cpu',weights_only=False)
comp={}
for key in ('network_weights','optimizer_state','grad_scaler_state','current_epoch','inference_allowed_mirroring_axes'):
    diffs=[]; ok=equal(cont.get(key),split.get(key),key,diffs)
    comp[key]={'equal':ok,'first_differences':diffs[:12],'difference_count':len(diffs)}
    if key in ('network_weights','optimizer_state','grad_scaler_state','current_epoch'): assert ok,comp[key]
cont_state=torch.load(ATT/'fixture25/state_continuous25.pt',map_location='cpu',weights_only=False)
split_state=torch.load(ATT/'resume13/state_resumed25.pt',map_location='cpu',weights_only=False)
for key in ('rng','scheduler_digest','fixture_cursor'):
    diffs=[]; ok=equal(cont_state.get(key),split_state.get(key),key,diffs)
    comp['state_'+key]={'equal':ok,'first_differences':diffs[:12],'difference_count':len(diffs)}
    assert ok,comp['state_'+key]
assert cont_state['fixture_cursor']==split_state['fixture_cursor']==25

fixture=torch.load(EXP/'engineering/fixtures/train_batch_fixture.pt',map_location='cpu',weights_only=False)
assert fixture['case_ids'] and set(fixture['case_ids'])<=observed
val_probe=json.loads((ENG/'train_internal_val_probe.json').read_text(encoding='utf-8'))
assert val_probe['status']=='PASS' and val_probe['optimizer_calls']==0
assert val_probe['model_unchanged'] and val_probe['optimizer_unchanged'] and val_probe['dev_model_calls']==0
assert ledger['validation_probe']['status']=='PASS' and ledger['validation_probe']['optimizer_attempts']==0
raw_info=json.loads((EXP/'raw/cpu_integrity.json').read_text(encoding='utf-8'))
verification={
 'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
 'cpu_gate':json.loads((ENG/'cpu/cpu_gate.json').read_text(encoding='utf-8'))['status'],
 'gpu_window_seconds':ledger['gpu_window_elapsed_seconds'],'gpu_window_cap_seconds':1800,
 'optimizer_attempts':ledger['attempts_used'],'optimizer_attempt_cap':100,
 'successful_updates':sum(v['successful_updates'] for v in phase_reports.values()),
 'amp_skips':sum(v['amp_skips'] for v in phase_reports.values()),
 'scientific_training':False,'run_training_called':False,'official_test_accessed':False,'dev_pixels_or_labels_read':False,'dev_model_calls':0,
 'native_train_size_case_coverage':cov,'unique_train_cases_sampled':len(observed),
 'native_train_size_sampled_cases':sorted(observed),
 'restore_fixture_case_ids':fixture['case_ids'],
 'patch_size':[1024,1536],'batch_size':2,'native_train_size_counts':raw_info['native_size_counts'],
 'plan':'nnUNetPlans_8GBtarget','trainer':'nnUNetTrainer_100epochs','fold':'all',
 'architecture':'PlainConvUNet','deep_supervision_outputs':8,'output_target_shapes_match':True,
 'environment':json.loads((ENG/'environment_lock.json').read_text(encoding='utf-8')),
 'phases':phase_reports,'restore_comparison':comp,
 'restore_test':'fixed transformed batch from actual Train input; 25 continuous vs 12+13 across separate process; model/optimizer/AMP scaler/scheduler/RNG exactly equal',
 'dataloader_restore_limitation':'Standard nnU-Net multi-process augmentation uses seeds=None and does not serialize worker queues; this test proves fixed-input model/optimizer/scaler/scheduler/RNG restore only, not bitwise replay of native augmentation stream.',
 'gpu_telemetry_limitation':'nvidia-smi sampled 10,275 MiB used during the resume phase at 98% utilization; this was a point sample, not an instrumented high-water mark. No OOM occurred.',
 'gpu_validation_probe':val_probe,
 'throughput':{'train50_update_interval_median_seconds':median_update_seconds,
   'train50_update_interval_p90_seconds':float(np.percentile(step_intervals,90)),
   'projected_25000_update_hours_update_only':projected_25000_updates_hours,
   'internal_train_validation_batches_for_100_epochs':val_probe['validation_batches_per_100_epochs_from_trainer'],
   'measured_internal_validation_hours':val_probe['measured_validation_gpu_hours_estimate'],
   'projected_updates_plus_internal_validation_hours':projected_25000_updates_hours+val_probe['measured_validation_gpu_hours_estimate'],
   'scope':'estimate only from measured 50-step train throughput and five Train validation batches; excludes planning, epoch bookkeeping, checkpoint I/O, process startup and variance. It does not authorize the 100-epoch run.'},
 'throughput_note':'Engineering throughput only, not a scientific performance estimate. 100-epoch cost estimate includes median update and internal Train validation costs measured on this plan; other overhead and run-to-run variance remain.',
 'preflight_gpu0':ledger['preflight_gpu0'],
 'execution_binding_sha256s':{
   p.name:sha(p) for p in [ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_lock_environment.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_cpu_pipeline.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_bounded_engineering.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_engineering_supervisor.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_cpu_gate.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_verify_engineering.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_validation_probe.py',
     ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020/nnunet2d_validation_probe_supervisor.py']},
 'plan_sha256':sha(EXP/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'),
 'train_manifest_sha256':sha(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'),
 'raw_train_export_manifest_sha256':sha(EXP/'raw/train_export_manifest.csv'),
 'output_checkpoint_sha256s':{name:v['checkpoint_sha256'] for name,v in phase_reports.items()},
 'checkpoint_count':4,
 'fov_supervision_difference':'Standard nnU-Net raw vessel labels supervise background outside project FOV; this is the approved workflow comparison and not a strict single-variable ablation.',
 'next_research_training':'not authorized; if later approved, initialize a fresh random model and do not continue from these engineering checkpoints.'
}
out=ENG/'verification.json'; out.write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':verification['status'],'gpu_window_seconds':verification['gpu_window_seconds'],
 'optimizer_attempts':verification['optimizer_attempts'],'successful_updates':verification['successful_updates'],
 'amp_skips':verification['amp_skips'],'size_coverage':cov,'restore_comparison':comp,
 'phases':phase_reports},indent=2))
