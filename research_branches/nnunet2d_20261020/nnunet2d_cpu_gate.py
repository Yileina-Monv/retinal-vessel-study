"""Recheck Train-only exports, fingerprints, plans, preprocessing and bounded entry."""
from __future__ import annotations
import ast, csv, hashlib, json, subprocess, sys
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path('${RETINA_PROJECT_ROOT}')
EXP=ROOT/'experiments/nnunet2d_branch_20261020'
BRANCH=ROOT/'code/retinal-vessel-study/research_branches/nnunet2d_20261020'
RAW=EXP/'raw/Dataset501_MAPLES'
PRE=EXP/'preprocessed/Dataset501_MAPLES'
PLANS=PRE/'nnUNetPlans_8GBtarget.json'
MANIFEST=ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'
DEV=ROOT/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
train=json.loads(MANIFEST.read_text(encoding='utf-8')); dev=json.loads(DEV.read_text(encoding='utf-8'))
ids={r['filename'] for r in train['rows']}
devrows=dev.get('rows',dev.get('images',[]))
dev_ids=set(devrows) if isinstance(devrows,dict) else {r.get('filename') or r.get('id') for r in devrows if isinstance(r,dict)} if devrows and isinstance(devrows[0],dict) else set(devrows)
assert len(ids)==110 and len(dev_ids)==28 and not ids&dev_ids
imgs={p.name[:-9] for p in (RAW/'imagesTr').glob('*_0000.png')}
labs={p.stem for p in (RAW/'labelsTr').glob('*.png')}
assert imgs==ids and labs==ids
dataset=json.loads((RAW/'dataset.json').read_text(encoding='utf-8'))
assert dataset['numTraining']==110 and dataset['file_ending']=='.png'
assert dataset['channel_names']=={'0':'R','1':'G','2':'B'}
assert dataset['labels']=={'background':0,'vessel':1}
fp=json.loads((PRE/'dataset_fingerprint.json').read_text(encoding='utf-8'))
plan=json.loads(PLANS.read_text(encoding='utf-8'))
cfg=plan['configurations']['2d']
assert plan['dataset_name']=='Dataset501_MAPLES'
assert cfg['data_identifier']=='nnUNetPlans_8GBtarget_2d'
assert cfg['batch_size']==2 and list(map(int,cfg['patch_size']))==[1024,1536]
assert cfg['normalization_schemes']==['ZScoreNormalization']*3
assert plan['plans_name']=='nnUNetPlans_8GBtarget'
folder=PRE/cfg['data_identifier']
data_files={p.name[:-5] for p in folder.glob('*.b2nd') if not p.stem.endswith('_seg')}
seg_files={p.stem[:-4] for p in folder.glob('*_seg.b2nd')}
pkl_files={p.stem for p in folder.glob('*.pkl')}
assert data_files==ids and seg_files==ids and pkl_files==ids
gt_files={p.stem for p in (PRE/'gt_segmentations').glob('*.png')}
assert gt_files==ids
assert not any((EXP/'dev_inputs').iterdir())
assert not list((EXP/'raw').glob('imagesTs/*'))

env_lock=json.loads((EXP/'engineering/environment_lock.json').read_text(encoding='utf-8'))
freeze=(EXP/'engineering/environment_freeze.txt').read_text(encoding='utf-8')
import torch
assert env_lock['versions']['torch'].startswith('2.4.0') and 'torch==2.4.0' in freeze
assert env_lock['torch_runtime_version']==torch.__version__=='2.4.0+cu121'
assert env_lock['torch_runtime_cuda_build']==torch.version.cuda=='12.1' and env_lock['pip_check_exit_code']==0
worker=BRANCH/'nnunet2d_bounded_engineering.py'
supervisor=BRANCH/'nnunet2d_engineering_supervisor.py'
cpu_pipeline=BRANCH/'nnunet2d_cpu_pipeline.py'
verify_script=BRANCH/'nnunet2d_verify_engineering.py'
probe=BRANCH/'nnunet2d_validation_probe.py'
probe_supervisor=BRANCH/'nnunet2d_validation_probe_supervisor.py'
lockenv=BRANCH/'nnunet2d_lock_environment.py'
finalizer=BRANCH/'nnunet2d_finalize_evidence.py'
for p in (worker,supervisor,cpu_pipeline,verify_script,probe,probe_supervisor,lockenv,finalizer):
    ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
worker_ast=ast.parse(worker.read_text(encoding='utf-8'))
calls=[n for n in ast.walk(worker_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run_training']
assert not calls, 'bounded entry must not call nnUNet run_training'
assert 'MAX_SECONDS=1800' in supervisor.read_text(encoding='utf-8') and 'MAX_ATTEMPTS=100' in supervisor.read_text(encoding='utf-8')
probe_ast=ast.parse(probe.read_text(encoding='utf-8'))
assert not [n for n in ast.walk(probe_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('train_step','run_training','step')]

# Exercise the selected upstream Trainer constructor on CPU only; never initialize a model here.
os.environ.update({'nnUNet_raw':str(EXP/'raw'),'nnUNet_preprocessed':str(EXP/'preprocessed'),
                   'nnUNet_results':str(EXP/'results'),'nnUNet_compile':'false','CUDA_VISIBLE_DEVICES':''})
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
trainer_plan=dict(plan); trainer_plan['continue_training']=False
trainer=nnUNetTrainer_100epochs(trainer_plan,'2d','all',dataset,torch.device('cpu'))
assert trainer.num_epochs==100 and trainer.num_iterations_per_epoch==250
assert trainer.initial_lr==0.01 and trainer.weight_decay==3e-5
assert not torch.cuda.is_initialized(), 'CPU gate unexpectedly initialized CUDA'
del trainer

report={
 'status':'CPU_GATE_PASS','created_utc':datetime.now(timezone.utc).isoformat(),
 'train_count':len(ids),'dev_name_count':len(dev_ids),'train_dev_name_overlap':0,
 'test_read':False,'dev_pixels_or_labels_read':False,'dev_model_calls':0,
 'raw_rgb_count':len(imgs),'raw_label_count':len(labs),'preprocessed_data_count':len(data_files),
 'preprocessed_seg_count':len(seg_files),'ground_truth_copy_count':len(gt_files),
 'native_sizes':json.loads((EXP/'raw/cpu_integrity.json').read_text())['native_size_counts'],
 'planner':'ExperimentPlanner','plans_name':plan['plans_name'],'configuration':'2d',
 'patch_size':[1024,1536],'batch_size':2,'trainer':'nnUNetTrainer_100epochs',
 'architecture':cfg['architecture']['network_class_name'],'normalization':cfg['normalization_schemes'],
 'cpu_pipeline':json.loads((EXP/'engineering/cpu/cpu_pipeline.json').read_text()),
 'environment_lock_sha256':sha(EXP/'engineering/environment_lock.json'),
 'torch_runtime_version':torch.__version__,'torch_cuda_build':torch.version.cuda,
 'source_sha256s':{p.name:sha(p) for p in (worker,supervisor,cpu_pipeline,verify_script,probe,probe_supervisor,lockenv,finalizer)},
 'dataset_sha256':sha(RAW/'dataset.json'),'plan_sha256':sha(PLANS),
 'fingerprint_sha256':sha(PRE/'dataset_fingerprint.json'),
 'run_training_called':False,'budget_caps':{'gpu_window_seconds':1800,'optimizer_attempts':100},
 'cpu_only_trainer_preflight':{'class':'nnUNetTrainer_100epochs','configured_epochs':100,
   'default_iterations_per_epoch':250,'cuda_initialized':False,'model_initialized':False},
 'cpu_gate_checks':['Train-only membership and exact counts','3-channel RGB and binary 0/1 GT roundtrip verified earlier for all 110','nnU-Net integrity/fingerprint','8GB ExperimentPlanner 2D plans','all 110 preprocessed data/segmentation/properties','isolated environment pip check','bounded code parses and no run_training call','Dev inputs empty and no Test copy']
}
out=EXP/'engineering/cpu/cpu_gate.json'
out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256s','cpu_pipeline')},indent=2))
