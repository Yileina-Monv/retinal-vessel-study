"""CPU-only implementation gate for the frozen MAPLES 2/1 exploration."""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retinal_m0.maples_class_split import ClassSplitPatches
from retinal_m0.maples_core import CorePatches, draw_geometry
from retinal_m0.maples_core_experiment import config, statehash
from retinal_m0.model import UNet
from retinal_m0.state import state_digest

ROOT=Path('${RETINA_PROJECT_ROOT}')
CODE=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/t0_tw_class_split_20261018'
PROTO=ROOT/'docs/t0_tw_class_split_20261018/protocol.json'
MANIFEST=ROOT/'docs/t0_tw_class_split_20261018/source_manifest.json'
PROTO_SHA='b486512cd3c3d9605fdb8f498b4e62c46c0666adcdf7544d0c7e901c8344775a'
SEEDS=(2026,2027,2028)
PAIR=ROOT/'experiments/t0_tw_core_20261016'
PYTHON=Path('${RETINA_PROJECT_ROOT}/envs/retina_a11/bin/python')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def writej(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def protected_check():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    allowed=m.get('authorized_derived_output_exception',{})
    exception_path=allowed.get('path') or allowed.get('file') or '${RETINA_PROJECT_ROOT}/experiments/t0_tw_weight_ablation_20261017/analysis/fixed_0p5_per_image.csv'
    exception_sha=allowed.get('before_sha256') or allowed.get('sha256') or allowed.get('original_sha256')
    protected=m.get('protected_sha256') or m.get('protected_files')
    if not isinstance(protected,dict): raise RuntimeError('source_manifest protected file map missing')
    failures=[];checked=0;official_paths=[]
    for path,want in protected.items():
        if 'official_test' in path.lower() or '/test/' in path.lower(): official_paths.append(path)
        if path==exception_path: continue
        p=Path(path)
        if not p.is_file() or sha(p)!=want: failures.append(path)
        checked+=1
    if failures: raise RuntimeError('Protected source mismatch: '+', '.join(failures[:10]))
    metadata=m.get('current_metadata_hashes_checked',{})
    metadata_failures=[p for p,want in metadata.items() if not Path(p).is_file() or sha(p)!=want]
    if metadata_failures: raise RuntimeError('Current metadata binding mismatch: '+', '.join(metadata_failures[:10]))
    # Confirm the separately authorized derived table still matches its frozen original.
    if exception_sha:
        if not Path(exception_path).is_file() or sha(exception_path)!=exception_sha:
            raise RuntimeError('Authorized derived table differs before byte-backup repair')
    if official_paths: raise RuntimeError('Manifest unexpectedly names Official Test paths')
    return {'status':'passed','source_manifest_sha256':sha(MANIFEST),
            'protected_file_count':checked,'authorized_csv_original_sha256':sha(exception_path),
            'official_test_paths_in_manifest':0}

def read_pairing(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if len(rows)!=2000 or [int(r['step']) for r in rows]!=list(range(1,2001)):
        raise RuntimeError(f'Pair log step coverage invalid: {path}')
    flat=[]
    for r in rows:
        if not all(len(r[k])==4 for k in ('draws','filenames','geometry','input_sha256')):
            raise RuntimeError(f'Pair row width invalid: {path}')
        for j,d in enumerate(r['draws']):
            flat.append((int(d),r['filenames'][j],r['geometry'][j],r['input_sha256'][j]))
    if [x[0] for x in flat]!=list(range(8000)):
        raise RuntimeError(f'Pair draw coverage invalid: {path}')
    return rows,flat

def check_seed(seed):
    c=config(seed,'T0',False);base=CorePatches(c,'T0')
    t0=PAIR/f'runs/seed_{seed}/T0/pairing.jsonl'
    tw=PAIR/f'runs/seed_{seed}/TW/pairing.jsonl'
    if t0.read_bytes()!=tw.read_bytes(): raise RuntimeError(f'Old T0/TW pairing logs differ seed={seed}')
    rows,flat=read_pairing(t0)
    for draw,name,geometry,_ in flat:
        idx=draw%len(base.names);row=base.native.rows[idx]
        expected=(draw,base.names[idx],list(draw_geometry(seed,draw,int(row['height']),int(row['width']),patch=512)))
        if (draw,name,geometry)!=expected:
            raise RuntimeError(f'Geometry/name mismatch seed={seed} draw={draw}')
    # Check actual full tensors at initial, round boundary, and final draws.
    dataset=ClassSplitPatches(c);checked=[]
    for draw in (0,1,2,3,7199,7200,7996,7997,7998,7999):
        old=base[draw];new=dataset[draw]
        if any(old[k]!=new[k] for k in ('filename','input_sha256')):
            raise RuntimeError(f'Real input identity mismatch seed={seed} draw={draw}')
        for k in ('image','mask','fov','geometry'):
            if not torch.equal(old[k],new[k]): raise RuntimeError(f'Tensor mismatch {k} seed={seed} draw={draw}')
        if not torch.equal(new['weight'],new['fov']*(1+new['area']*new['mask'])):
            raise RuntimeError(f'2/1 mapping mismatch seed={seed} draw={draw}')
        target=flat[draw]
        got=(draw,new['filename'],new['geometry'].tolist(),new['input_sha256'])
        if got!=target: raise RuntimeError(f'New sample differs from old paired log seed={seed} draw={draw}')
        checked.append(draw)
    # Verify the two inherited trajectories really began from paired state summaries.
    summaries={m:json.loads((PAIR/f'runs/seed_{seed}/{m}/summary.json').read_text()) for m in ('T0','TW')}
    keys=('initial_model','initial_optimizer','initial_scaler','initial_rng')
    if any(summaries['T0'].get(k)!=summaries['TW'].get(k) for k in keys):
        raise RuntimeError(f'Old initial-state pairing mismatch seed={seed}')
    if any(summaries[m].get('status')!='passed' or summaries[m].get('completed_steps')!=2000 or summaries[m].get('next_global_draw')!=8000 for m in ('T0','TW')):
        raise RuntimeError(f'Old reference incomplete seed={seed}')
    return {'seed':seed,'all_8000_draws_metadata_geometry':'passed',
            'old_t0_tw_logs_identical':True,'actual_input_draws_checked':checked,
            'new_inputs_equal_old_T0_TW':True,'old_initial_state_summaries_equal':True}

def current_binding():
    p=json.loads(PROTO.read_text(encoding='utf-8'))
    if sha(PROTO)!=PROTO_SHA or p.get('status')!='FROZEN_USER_APPROVED_EXPLORATORY':
        raise RuntimeError('Frozen protocol hash/status mismatch')
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    files=[CODE/'retinal_m0/maples_class_split.py',CODE/'retinal_m0/maples_class_split_experiment.py',
           CODE/'retinal_m0/maples_class_split_gate.py',CODE/'tests/test_maples_class_split.py',
           CODE/'retinal_m0/maples_core.py',CODE/'retinal_m0/maples_core_experiment.py',
           CODE/'retinal_m0/maples_t0.py',CODE/'retinal_m0/objectives.py',
           CODE/'retinal_m0/model.py',CODE/'retinal_m0/state.py',CODE/'retinal_m0/inference.py',
           CODE/'retinal_m0/maples_evaluate.py',CODE/'retinal_data/maples_native.py',
           CODE/'retinal_data/maples_fov.py',CODE/'retinal_data/fives.py',
           CODE/'retinal_m1/maps.py',CODE/'configs/development_v1.json',CODE/'releases/development_v1/normalization.json']
    if any(not f.is_file() for f in files): raise RuntimeError('Missing implementation source for binding')
    parent=ROOT/'experiments/t0_tw_core_20261016/execution_binding.json'
    m0=ROOT/'checkpoints/m0_10000_20261008/last.pt'
    parent_json=json.loads(parent.read_text())
    expected_parent=m.get('current_metadata_hashes_checked',{}).get(str(parent))
    if expected_parent and sha(parent)!=expected_parent:
        raise RuntimeError('Formal parent execution binding differs from frozen manifest')
    if parent_json.get('protocol_sha256')!=p['parents']['formal_sha256']:
        raise RuntimeError('Formal parent execution binding is for a different protocol')
    if sha(m0)!=p['training']['source_M0_sha256']:
        raise RuntimeError('Original M0 checksum changed')
    data=ROOT/'data/maples_handoff_20261008'
    return {'protocol_sha256':sha(PROTO),'source_manifest_sha256':sha(MANIFEST),
            'parent_binding_sha256':sha(parent),'m0_sha256':sha(m0),
            'data_manifest_sha256':sha(data/'data_manifest.csv'),
            'split_sha256':sha(data/'split_record.json'),
            'normalization_sha256':sha(CODE/'releases/development_v1/normalization.json'),
            'sources':{str(f.relative_to(CODE)):sha(f) for f in files},
            'scope':{'seeds':list(SEEDS),'method':'v2_b1','updates_each':2000,'draws_each':8000,
                     'training_images':110,'dev_images':28,'official_test_accessed':False}}

def run_gate():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'runs').exists() or (OUT/'execution_binding.json').exists(): raise RuntimeError('A binding or run already exists; refusing to overwrite')
    gate=OUT/'implementation_gate';gate.mkdir(parents=True,exist_ok=True)
    nums=[int(p.name.split('_')[-1]) for p in gate.glob('attempt_[0-9][0-9]') if p.is_dir()]
    attempt=gate/f'attempt_{max(nums,default=0)+1:02d}';attempt.mkdir(parents=True,exist_ok=False)
    p=json.loads(PROTO.read_text(encoding='utf-8'))
    if sha(PROTO)!=PROTO_SHA: raise RuntimeError('Protocol SHA mismatch')
    protected=protected_check();writej(attempt/'protected_manifest_check.json',protected)
    test=subprocess.run([str(PYTHON),'-B','-m','unittest','-v','tests.test_maples_class_split'],
                        cwd=CODE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (attempt/'targeted_tests.log').write_text(test.stdout,encoding='utf-8')
    if test.returncode: raise RuntimeError('Targeted CPU tests failed')
    seeds=[check_seed(seed) for seed in SEEDS]
    binding=current_binding()
    bind_path=OUT/'execution_binding.json';writej(bind_path,binding)
    evidence={'status':'passed','all_passed':True,'protocol_sha256':PROTO_SHA,
      'source_manifest_sha256':sha(MANIFEST),'execution_binding_sha256':sha(bind_path),
      'training_updates':0,'gpu_training_started':False,'official_test_accessed':False,
      'checks':{'protocol_and_manifest':True,'protected_files':protected,
        'mapping_and_loss_tests':{'passed':7,'failed':0,'log':str(attempt/'targeted_tests.log')},
        'seed_pairing':seeds,'binding_matches_current_sources':True},
      'authorization_scope':p['authorization']['gpu_training_permission']}
    writej(gate/'verification.json',evidence)
    # One last read-back after writing is the actual startup gate.
    reread=json.loads((gate/'verification.json').read_text())
    if not reread.get('all_passed') or reread.get('execution_binding_sha256')!=sha(bind_path):
        raise RuntimeError('Gate read-back failed')
    return reread

if __name__=='__main__':
    print(json.dumps(run_gate(),ensure_ascii=False,indent=2))
