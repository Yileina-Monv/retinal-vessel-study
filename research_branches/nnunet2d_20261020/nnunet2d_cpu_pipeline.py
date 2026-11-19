"""Run the approved Train-only nnU-Net CPU fingerprint, plan and 2D preprocessing."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('${RETINA_PROJECT_ROOT}')
EXP = ROOT / 'experiments/nnunet2d_branch_20261020'
ENV = ROOT / 'envs/nnunet2d_v2'
BIN = ROOT / 'envs/retina_a11/bin'
OUT = EXP / 'engineering/cpu'
DATASET_ID = '501'
PLAN = 'nnUNetPlans_8GBtarget'

env = os.environ.copy()
env.update({
    'nnUNet_raw':str(EXP/'raw'),
    'nnUNet_preprocessed':str(EXP/'preprocessed'),
    'nnUNet_results':str(EXP/'results'),
    'CUDA_VISIBLE_DEVICES':'',
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'NUMEXPR_NUM_THREADS':'1','nnUNet_n_proc_DA':'1',
})
python = str(ENV/'bin/python')
commands = [
    ('fingerprint', [python, str(BIN/'nnUNetv2_extract_fingerprint'), '-d', DATASET_ID, '-np', '1', '--verify_dataset_integrity', '--no_pbar']),
    ('planning', [python, str(BIN/'nnUNetv2_plan_experiment'), '-d', DATASET_ID, '-pl', 'ExperimentPlanner', '-gpu_memory_target', '8', '-overwrite_plans_name', PLAN]),
    ('preprocess_2d', [python, str(BIN/'nnUNetv2_preprocess'), '-d', DATASET_ID, '-plans_name', PLAN, '-c', '2d', '-np', '1', '--no_pbar']),
]
record={'started_utc':datetime.now(timezone.utc).isoformat(),'dataset_id':int(DATASET_ID),'plan_name':PLAN,'cuda_visible_devices':'','worker_limit':1,'stages':[]}
for name, cmd in commands:
    start=time.monotonic()
    log=OUT/f'{name}.log'
    print(f'[{name}] start: {cmd}',flush=True)
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); f.write(line); f.flush()
        rc=p.wait()
    elapsed=time.monotonic()-start
    record['stages'].append({'name':name,'command':cmd,'return_code':rc,'elapsed_seconds':elapsed,'log':str(log)})
    print(f'[{name}] exit={rc} elapsed_seconds={elapsed:.1f}',flush=True)
    (OUT/'cpu_pipeline.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if rc:
        record['status']='FAILED'
        (OUT/'cpu_pipeline.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        raise SystemExit(rc)
record['status']='PASS'
record['finished_utc']=datetime.now(timezone.utc).isoformat()
(OUT/'cpu_pipeline.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('CPU_PIPELINE_PASS',flush=True)
