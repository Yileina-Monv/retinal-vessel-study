"""Protocol-bound original-A0-loss runner with the reviewed stratified origins."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retinal_m0.maples_g1_safety import GateError, read_json, canonical_sha, exclusive_lock, atomic_json
from retinal_m0.maples_stratified_science_core import validate_scope

ROOT=Path('${RETINA_PROJECT_ROOT}'); CODE=Path(__file__).resolve().parents[1]
STAGE=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/spatial_stratified_scientific_2000_v1_20261025'
PROTOCOL=STAGE/'protocol.json'; STATUS=CODE/'STATUS.md'; OUT=STAGE/'runs/seed_2026/SPATIAL'
PROTO_SHA='7c3680884ef79426e62f53272b88727805cd1a8ae36a8a79ae28a0fa1c3f75e7'
GPU_UUID='${GPU_UUID}'
NORMATIVE_SHA='45311163b551550b734af4603a845293e2f4a8ee5bce3cd387561a5e49ea5c33'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def check_protocol():
    q=read_json(PROTOCOL)
    if sha(PROTOCOL)!=PROTO_SHA or q.get('normative_text_sha256')!=NORMATIVE_SHA: raise GateError('frozen D protocol digest mismatch')
    validate_scope({'method':q['experiment']['method'],'seed':q['experiment']['seed'],'updates':q['experiment']['successful_update_target']})
    if q['experiment']['loss']!='original BCE+Dice' or q['experiment']['sampling']!='fixed_6x12_origins_v1': raise GateError('frozen sampler/loss drift')
    if q['experiment']['evaluation_steps'] != list(range(200,2001,200)): raise GateError('evaluation schedule drift')
    if q['runtime']['training_forward_cap']!=2000 or q['runtime']['training_gradient_cap']!=2000: raise GateError('training budget drift')
    if q['primary']['required_metrics']!=['dice'] or q['runtime']['evaluation_forward_cap']!=1900 or q['runtime']['evaluation_tile_cap']!=7450: raise GateError('Dice-only primary/evaluation budget drift')
    if q['prohibited'] and ('official Test' not in q['prohibited']): raise GateError('Test prohibition missing')
    return q


def _source_receipts(full_sources=True):
    q=check_protocol()
    previous=STAGE.parent/'ufl_asymmetric_scientific_2000_v1_20261025'
    inherited=dict(read_json(previous/'cpu_preparation/source_binding.json')['inherited_receipts']['original_expected_sha256'])
    for path,want in inherited.items():
        if full_sources or path.startswith(str(CODE)) or path.endswith(('.json','.py','.csv','.jsonl')):
            if not Path(path).is_file() or sha(path)!=want:raise GateError('original immutable source drift:'+path)
    a0run=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0'
    cfg=read_json(a0run/'config.json');pairing=Path(q['reference']['expected_pairs_path'])
    fixed={
      'protocol.json':(PROTOCOL,PROTO_SHA),
      'SCIENTIFIC_PROTOCOL.md':(STAGE/'SCIENTIFIC_PROTOCOL.md',NORMATIVE_SHA),
      'reference_manifest':(Path(q['reference']['science_reference_manifest']),q['reference']['science_reference_manifest_sha256']),
      'A0_reuse_binding':(Path(q['reference']['A0_reuse_binding']),q['reference']['A0_reuse_binding_sha256']),
      'original_A0_pairs':(a0run/'pairing.jsonl',q['reference']['original_A0_pairs_sha256']),
      'candidate_pairs':(pairing,q['reference']['expected_pairs_sha256']),
      'candidate_trajectory':(Path(q['reference']['trajectory_path']),q['reference']['trajectory_sha256']),
      'candidate_CPU_review':(Path(q['reference']['candidate_CPU_review_path']),q['reference']['candidate_CPU_review_sha256']),
      'candidate_input_review':(STAGE/'cpu_preparation/input_preparation.json',q['evidence']['input_preparation_sha256']),
      'A0_config':(a0run/'config.json',inherited[str(a0run/'config.json')]),
      'original_M0':(Path(cfg['source_m0']),cfg['source_m0_sha256'])}
    result={'original_expected_sha256':inherited}
    for name,(path,want) in fixed.items():
        if not path.is_file() or ((full_sources or name!='original_M0') and sha(path)!=want):raise GateError('source/evidence mismatch:'+name)
        result[name]={'path':str(path),'sha256':want}
    rows=[json.loads(x) for x in pairing.read_text().splitlines() if x]
    original=[json.loads(x) for x in (a0run/'pairing.jsonl').read_text().splitlines() if x]
    if len(rows)!=2000 or len(original)!=2000 or [x for r in rows for x in r['draws']]!=list(range(8000)):raise GateError('exact2000/8000 candidate identity required')
    for a,b in zip(rows,original):
        if any(a[k]!=b[k] for k in ('step','draws','filenames')) or any(ga[2:]!=gb[2:] for ga,gb in zip(a['geometry'],b['geometry'])):raise GateError('original image order/augmentation drift')
    result.update(pairing_row_count=2000,paired_draw_count=8000,pairing_rows_sha256=sha(pairing),pairing_meaning='same image visits/augmentations/budget; changed origins and input hashes by design')
    return q,result,rows


def cpu_session_probe():
    import secrets
    nonce=secrets.token_hex(24); env=dict(os.environ,SPATIAL_STRATIFIED_SESSION_NONCE=nonce)
    proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--session-worker','--parent-pid',str(os.getpid()),'--nonce',nonce],
                        env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    if proc.returncode or proc.stdout.strip()!='SPATIAL_SESSION_CHILD_OK': raise GateError('real CLI parent-child session probe failed: '+proc.stdout[-2000:])
    return {'passed':True,'parent_pid':os.getpid(),'child_output':proc.stdout.strip()}


def session_worker(parent_pid,nonce):
    if os.getppid()!=parent_pid or os.environ.get('SPATIAL_STRATIFIED_SESSION_NONCE')!=nonce or len(nonce)<32:
        raise GateError('CLI worker is not the authenticated direct child')
    print('SPATIAL_SESSION_CHILD_OK')
    return 0


def legacy_cldice_crosscheck():
    import ast,numpy as np
    from skimage.morphology import skeletonize
    from retinal_m0.maples_stratified_science_analysis import hard_cldice
    q=check_protocol(); diagnostic=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/literature_mechanism_review_20261023/diagnose_cldice.py'
    spec=read_json(diagnostic.with_name('diagnostic_spec.json'))
    if spec.get('status')!='EXPLORATORY_DIAGNOSTIC_NOT_NEW_PRIMARY_OR_PROTECTION' or 'Full frozen FOV, including U' not in spec.get('support','') or 'mask before skeletonization' not in spec.get('support',''): raise GateError('frozen hard-clDice diagnostic semantics changed')
    tree=ast.parse(diagnostic.read_text(encoding='utf-8')); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('divide','measure')]
    if {n.name for n in nodes}!={'divide','measure'}: raise GateError('pure legacy clDice functions unavailable')
    env={'np':np,'skeletonize':skeletonize}; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(diagnostic),'exec'),env)
    refs=read_json(Path(q['reference']['science_reference_manifest']))['images']; name=refs[0]['filename']; arrays=_load_arrays(refs[0]['reference_npz']); fov=arrays['FOV'].astype(bool); gt=arrays['GT'].astype(bool)&fov
    a0=read_json(Path(q['reference']['A0_reuse_binding']))['probability_maps']; a0p={r['filename']:r for r in a0}[name]
    b1root=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022'
    outcomes={}
    maps={'A0':(Path(a0p['path']),a0p['sha256'])}
    for method in ('B1','B2'):
        metrics=read_json(b1root/f'runs/{method}/evaluation/step_002000/metrics.json')
        row=next(r for r in metrics['images'] if r['filename']==name)
        maps[method]=(b1root/f'runs/{method}/evaluation/step_002000/probabilities/{name}.npy',row['probability_sha256'])
    for method,(path,want) in maps.items():
        if sha(path)!=want: raise GateError('legacy crosscheck probability digest mismatch: '+method)
        prob=np.load(path,allow_pickle=False); pred=(prob>=.5)&fov
        old,_,_=env['measure'](pred,gt); new=hard_cldice(pred,arrays['GT'],arrays['FOV'])
        pairs={'topology_precision':'topology_precision','topology_sensitivity':'topology_sensitivity'}
        for a,b in pairs.items():
            if old[a]!=new[b]: raise GateError(f'legacy hard-clDice mismatch {method}/{a}')
        if old['clDice']!=new['cldice']: raise GateError('legacy hard-clDice score mismatch: '+method)
        outcomes[method]={'filename':name,'probability_sha256':want,'cldice':new['cldice'],'topology_precision':new['topology_precision'],'topology_sensitivity':new['topology_sensitivity']}
    return {'diagnostic_source':str(diagnostic),'diagnostic_source_sha256':sha(diagnostic),'diagnostic_spec_sha256':sha(diagnostic.with_name('diagnostic_spec.json')),'comparisons':outcomes,'top_level_module_not_imported':True}


def a0_summary_path():
    return ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0/summary.json'


def install_dataset_guard():
    """Whitelisted Train/Dev content only, before any full-source checks."""
    from retinal_m0.maples_ufl_engineering import data_guard
    cfg=read_json(ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0/config.json')
    dev=read_json(Path(check_protocol()['reference']['science_reference_manifest']))
    names=set(cfg['train_sample_filenames'])|{r['filename'] for r in dev['images']}
    old=read_json(STAGE/'cpu_preparation/source_binding.json') if (STAGE/'cpu_preparation/source_binding.json').is_file() else None
    inherited=old['inherited_receipts']['original_expected_sha256'] if old else read_json(ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/ready_binding.json')['sources']
    root=Path(cfg['data_root']).resolve(); allowed=[]
    for path in inherited:
        pp=Path(path).resolve()
        if root in pp.parents and pp.suffix.lower() in ('.png','.tif','.tiff','.jpg','.npy','.npz'):
            if pp.stem not in names: raise GateError('non-Train/Dev content in inherited evidence; no read')
            allowed.append(pp)
    # Train references may be absent from an older scientific manifest.
    eng=read_json(ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/ufl_asymmetric_v1_20261024/engineering100/execution_binding.json')
    allowed.extend(eng['train_content_paths'])
    data_guard(root,allowed)


def cpu_preflight():
    import platform,torch,scipy,skimage,PIL,numpy as np
    evidence=STAGE/'cpu_preparation'; evidence.mkdir(parents=True,exist_ok=True)
    test_env=dict(os.environ,PYTHONPATH=str(CODE),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    test=subprocess.run([sys.executable,'-m','unittest','tests.test_maples_stratified_science','-v'],cwd=CODE,env=test_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (evidence/'cpu_tests.log').write_text(test.stdout,encoding='utf-8')
    if test.returncode: raise GateError('targeted C CPU tests failed; see '+str(evidence/'cpu_tests.log'))
    import re
    match=re.search(r'Ran (\d+) tests?',test.stdout)
    if not match: raise GateError('unittest summary missing from CPU log')
    test_count=int(match.group(1))
    session=cpu_session_probe()
    legacy=legacy_cldice_crosscheck()
    install_dataset_guard()
    q,receipts,rows=_source_receipts()
    sources={}
    for name,path in {
      'science_core':CODE/'retinal_m0/maples_stratified_science_core.py',
      'science_entry':CODE/'retinal_m0/maples_stratified_science.py',
      'science_analysis':CODE/'retinal_m0/maples_stratified_science_analysis.py',
      'science_tests':CODE/'tests/test_maples_stratified_science.py',
      'base_loss':CODE/'retinal_m0/maples_core.py',
      'training_data':CODE/'retinal_m0/maples_core_experiment.py',
      'model':CODE/'retinal_m0/model.py',
      'state':CODE/'retinal_m0/state.py',
      'inference':CODE/'retinal_m0/inference.py',
      'frozen_analysis':CODE/'retinal_m0/maples_branch_science_analysis.py',
      'diagnostic_spec':ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/literature_mechanism_review_20261023/diagnostic_spec.json',
      'frozen_reference_code':CODE/'retinal_m0/maples_branch_science_reference.py',
      'original_T0_summary':a0_summary_path(),
      'old_science_endpoint_analysis':ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/endpoint_analysis.json',
      'legacy_diagnostic_functions_source':ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/literature_mechanism_review_20261023/diagnose_cldice.py',
      'candidate_data':CODE/'retinal_m0/maples_stratified_data.py',
      'automatic_CPU_finalizer':STAGE/'finalize_spatial_after_exit.py',
      'SPATIAL_engineering_core':CODE/'retinal_m0/maples_ufl_engineering_core.py',
      'SPATIAL_engineering_entry':CODE/'retinal_m0/maples_ufl_engineering.py',
    }.items():
        if not path.is_file(): raise GateError('required source missing: '+str(path))
        sources[name]={'path':str(path),'sha256':sha(path)}
    binding={'phase':'SPATIAL_STRATIFIED_SCIENTIFIC_2000_V1','protocol_sha256':PROTO_SHA,'normative_sha256':NORMATIVE_SHA,
             'sources':sources,'inherited_receipts':receipts}
    binding_sha=canonical_sha(binding)
    (evidence/'source_binding.json').write_text(json.dumps({**binding,'binding_sha256':binding_sha},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fixture=STAGE/'cpu_preparation/A0_endpoint_fixture/analysis/endpoint_2000.json'
    fixture_obj=read_json(fixture)
    if fixture_obj.get('status')!='COMPLETE_FIXTURE_ONLY' or len(fixture_obj.get('per_image',[]))!=28 or len(fixture_obj['fixed_ids_by_metric']['near_thin_recall_d1'])!=9 or any(v['SPATIAL-A0']['estimate']!=0 for v in fixture_obj['metrics'].values() if 'SPATIAL-A0' in v): raise GateError('complete existing-probability CPU endpoint fixture missing or failed')
    envs={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,'skimage':skimage.__version__,'pillow':PIL.__version__}
    receipt={'status':'CPU_VALIDATED_NOT_A_GPU_RELEASE','all_passed':True,'phase':'SPATIAL_STRATIFIED_SCIENTIFIC_2000_V1',
             'protocol_sha256':PROTO_SHA,'binding_sha256':binding_sha,'binding_file_sha256':sha(evidence/'source_binding.json'),
             'cpu_test_log_sha256':sha(evidence/'cpu_tests.log'),'tests_passed':test_count,'cli_parent_child_probe':session,
             'checks':{'budget_failure_and_unknown_stop':'passed','shared_loop_checkpoint_resume':'passed',
                       'full_fov_hard_cldice_empty_cases':'passed','legacy_A0_B1_B2_real_prediction_crosscheck':legacy,'fixed_pairing_2000_rows_8000_draws':'passed',
                       'protocol_and_inherited_receipts':'passed','a0_inference':'not_run_reuse_only','full_endpoint_existing_A0_fixture':{'status':'COMPLETE_FIXTURE_ONLY','path':str(fixture),'sha256':sha(fixture),'old_near_valid_images':9,'old_near_positions':77}},
             'environment':envs,'cuda_initialized':False,'gpu_queries':False,'training_started':False,'official_Test_read':False,
             'evidence_paths':[str(evidence/'source_binding.json'),str(evidence/'cpu_tests.log')]}
    (evidence/'cpu_review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return receipt


def current_spatial_authority():
    text=STATUS.read_text(encoding='utf-8'); start='<!-- SPATIAL_SCIENCE2000_CURRENT -->'; end='<!-- /SPATIAL_SCIENCE2000_CURRENT -->'
    if text.count(start)!=1 or text.count(end)!=1: raise GateError('unique current C authorization block missing')
    record=json.loads(text.split(start,1)[1].split(end,1)[0].strip())
    if record.get('phase')!='SPATIAL_STRATIFIED_SCIENTIFIC_2000_V1' or record.get('protocol_sha256')!=PROTO_SHA: raise GateError('current STATUS authorization is for another phase')
    if record.get('user_approved_science2000') is not True: raise GateError('explicit scientific2000 user approval missing')
    return record


def validate_release(full_sources=True):
    status=current_spatial_authority(); release=STAGE/'SCIENCE_RELEASE.json'; ev=STAGE/'cpu_preparation'
    binding=read_json(ev/'source_binding.json'); receipt=read_json(ev/'cpu_review_receipt.json')
    binding_obj={k:v for k,v in binding.items() if k!='binding_sha256'}
    bsha=canonical_sha(binding_obj)
    _,current_receipts,_=_source_receipts(full_sources=full_sources)
    if binding.get('inherited_receipts')!=current_receipts: raise GateError('inherited A0/reference evidence changed after CPU review')
    if binding.get('binding_sha256')!=bsha or receipt.get('binding_sha256')!=bsha or not receipt.get('all_passed') or receipt.get('status')!='CPU_VALIDATED_NOT_A_GPU_RELEASE':
        raise GateError('C CPU review binding/receipt missing or stale')
    for item in binding['sources'].values():
        if sha(item['path'])!=item['sha256']: raise GateError('source drift after CPU review: '+item['path'])
    if status.get('GPU_released') is not True or status.get('state')!='RELEASED' or not release.is_file(): raise GateError('Sol scientific release is not active; GPU held')
    r=read_json(release)
    if r.get('phase')!='SPATIAL_STRATIFIED_SCIENTIFIC_2000_V1' or r.get('protocol_sha256')!=PROTO_SHA or r.get('binding_sha256')!=bsha or r.get('release_id')!=status.get('release_id'):
        raise GateError('release is not bound to current C CPU review')
    if status.get('binding_sha256')!=bsha or status.get('cpu_receipt_sha256')!=sha(ev/'cpu_review_receipt.json') or status.get('release_file_sha256')!=sha(release):
        raise GateError('STATUS is not bound to this release/source/CPU receipt')
    if r.get('cpu_receipt_sha256')!=sha(ev/'cpu_review_receipt.json'):
        raise GateError('release/receipt digest mismatch')
    if r.get('GPU_released') is not True or r.get('method')!='SPATIAL' or r.get('seed')!=2026 or r.get('updates')!=2000 or r.get('charged_update_cap')!=2000 or r.get('GPU0_uuid')!=GPU_UUID or r.get('official_Test_allowed') is not False:
        raise GateError('release scope mismatch')
    if r.get('loss')!='original BCE+Dice' or r.get('trajectory_sha256')!=check_protocol()['reference']['trajectory_sha256']: raise GateError('released sampler/loss drift')
    return status,r,bsha



def process_token(pid):
    try:
        return str(pid)+':'+Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError) as exc:raise GateError('required live process missing') from exc


def verify_live_session(session,nonce):
    parent=os.getppid();worker=os.getpid()
    if session.get('controller_pid')!=parent or session.get('worker_pid')!=worker or session.get('nonce')!=nonce or session.get('controller_token')!=process_token(parent) or session.get('worker_token')!=process_token(worker):raise GateError('controller/worker live session identity lost')
    return True


def runtime_authorizer(binding_sha):
    binding=read_json(STAGE/'cpu_preparation/source_binding.json')
    pins=dict(binding['inherited_receipts']['original_expected_sha256'])
    pins.update({v['path']:v['sha256'] for v in binding['sources'].values()})
    pins.update({v['path']:v['sha256'] for v in binding['inherited_receipts'].values() if isinstance(v,dict) and 'path' in v and 'sha256' in v})
    # Immutable small code/config pins per step; raw input fingerprints are compared in the actual batch.
    small={p:h for p,h in pins.items() if p.startswith(str(CODE)) or p.endswith(('.json','.py','.csv','.jsonl'))}
    session_path=STAGE/'RUN_SESSION.json';session=read_json(session_path);session_sha=sha(session_path);nonce=os.environ.get('SPATIAL_STRATIFIED_SESSION_NONCE')
    verify_live_session(session,nonce)
    release_sha=sha(STAGE/'SCIENCE_RELEASE.json'); cpu_sha=sha(STAGE/'cpu_preparation/cpu_review_receipt.json')
    def authorize():
        if sha(session_path)!=session_sha: raise GateError('runtime session changed')
        verify_live_session(session,nonce)
        status=current_spatial_authority()
        if status.get('GPU_released') is not True or status.get('state')!='RELEASED' or status.get('binding_sha256')!=binding_sha or status.get('release_file_sha256')!=release_sha or status.get('cpu_receipt_sha256')!=cpu_sha or sha(STAGE/'SCIENCE_RELEASE.json')!=release_sha:
            raise GateError('current scientific authority withdrawn or changed')
        for path,want in small.items():
            if sha(path)!=want: raise GateError('runtime code/config source drift: '+path)
        return True
    return authorize


def _update_current_status(**changes):
    text=STATUS.read_text(encoding='utf-8'); start='<!-- SPATIAL_SCIENCE2000_CURRENT -->'; end='<!-- /SPATIAL_SCIENCE2000_CURRENT -->'
    a=text.index(start)+len(start); b=text.index(end); q=json.loads(text[a:b].strip()); q.update(changes)
    updated=text[:a]+'\n'+json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n'+text[b:]
    if STATUS.read_text(encoding='utf-8')!=text: raise GateError('STATUS changed concurrently')
    tmp=STATUS.with_name('STATUS.md.SPATIAL_science.pending')
    with tmp.open('x',encoding='utf8') as stream: stream.write(updated);stream.flush();os.fsync(stream.fileno())
    if STATUS.read_text(encoding='utf-8')!=text: raise GateError('STATUS changed concurrently')
    os.replace(tmp,STATUS)


def _gpu0_resources():
    import shutil
    raw=subprocess.check_output(['nvidia-smi','--id=${GPU_UUID}','--query-gpu=uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',')
    uuid,total,used,util=[x.strip() for x in raw]
    if uuid!='${GPU_UUID}' or int(used)>=512 or int(util)>=10: raise GateError('GPU0 UUID/busy resource gate failed; no process stopped')
    procs=subprocess.check_output(['nvidia-smi','--id=${GPU_UUID}','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    if procs.strip(): raise GateError('GPU0 has a compute process; no process stopped')
    free=shutil.disk_usage(STAGE).free
    if free<21474836480: raise GateError('less than 20 GiB free disk')
    return {'index':0,'uuid':uuid,'total_mib':int(total),'used_mib':int(used),'util_percent':int(util),'free_disk_bytes':free}


def worker_authority(parent_pid,nonce):
    status,release,binding_sha=validate_release()
    import time
    session_path=STAGE/'RUN_SESSION.json'; session=None
    for _ in range(150):
        if session_path.is_file():
            candidate=read_json(session_path)
            if candidate.get('worker_pid')==os.getpid(): session=candidate; break
        time.sleep(.1)
    if session is None: raise GateError('controller session record was not published')
    if os.getppid()!=parent_pid or session.get('controller_pid')!=parent_pid or session.get('worker_pid')!=os.getpid() or session.get('nonce')!=nonce or os.environ.get('SPATIAL_STRATIFIED_SESSION_NONCE')!=nonce:
        raise GateError('worker is not the authenticated direct child of current controller')
    if session.get('release_sha256')!=sha(STAGE/'SCIENCE_RELEASE.json') or session.get('binding_sha256')!=binding_sha:
        raise GateError('worker session release/binding drift')
    verify_live_session(session,nonce)
    return status,release,binding_sha


def _setup(seed):
    import random,numpy as np,torch
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _build_training_objects(device):
    import torch
    from retinal_m0.model import UNet
    from retinal_m0.maples_stratified_data import StratifiedPatches
    cfg_path=ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0/config.json'
    cfg=read_json(cfg_path)
    if cfg.get('seed')!=2026 or cfg.get('max_steps')!=2000 or cfg.get('source_m0_sha256')!='8bce5b8f47c7fe43d553d4467adc46f5387eade014265b7a6a6f4b5211e67d79':
        raise GateError('original T0 training source configuration drift')
    saved=torch.load(cfg['source_m0'],map_location='cpu',weights_only=True)
    if saved.get('step')!=10000 or sha(cfg['source_m0'])!=cfg['source_m0_sha256']: raise GateError('original M0 checkpoint mismatch')
    _setup(2026)
    model_cfg=read_json(CODE/'configs/development_v1.json')
    model=UNet(model_cfg['widths'],model_cfg['group_norm_groups']).to(device)
    model.load_state_dict(saved['model'],strict=True)
    del saved
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'],foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'])
    q=check_protocol()
    dataset=StratifiedPatches(cfg,q['reference']['trajectory_path'],q['reference']['trajectory_sha256'])
    pairs=Path(q['reference']['expected_pairs_path'])
    expected=[json.loads(x) for x in pairs.read_text().splitlines() if x]
    if len(expected)!=2000: raise GateError('A0 source pairing row count changed')
    return model,optimizer,scaler,dataset,expected,cfg


def _load_arrays(npz):
    import numpy as np
    with np.load(npz,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}


def hard_scalar_details(hard):
    result={k:hard[k] for k in ('pred_skeleton_count','gt_skeleton_count','pred_hit_count','gt_hit_count')}
    result.update({k+'_NA_reason':v for k,v in hard['NA_reason'].items()})
    return result


def validate_probability_records(rows,refs,probs):
    import numpy as np
    expected=[r['filename'] for r in refs]
    if len(expected)!=28 or len(set(expected))!=28 or len(rows)!=28 or [r.get('filename') for r in rows]!=expected: raise GateError('evaluation must contain exactly frozen Dev28 in order')
    for row,ref in zip(rows,refs):
        path=Path(probs)/(row['filename']+'.npy')
        if not path.is_file() or sha(path)!=row.get('probability_sha256'): raise GateError('probability digest mismatch')
        a=np.load(path,allow_pickle=False)
        if a.dtype!=np.float32 or list(a.shape)!=ref['native_shape'] or not np.isfinite(a).all() or np.any((a<0)|(a>1)): raise GateError('probability dtype/shape/range mismatch')
    return True


def reserve_evaluation_forward(budget,slot_key,name,batch_tiles,protocol):
    """Charge before a real inference forward; no retries/refunds or extra slots."""
    if slot_key not in {'SPATIAL:'+str(x) for x in range(200,2001,200)}:
        raise GateError('unapproved evaluation slot')
    profile=protocol['runtime']['evaluation_image_profile'][name]
    state=budget['slots'][slot_key]['images'][name]
    if state.get('status')!='INCOMPLETE' or state.get('attempts')!=1 or type(batch_tiles) is not int or batch_tiles<1:
        raise GateError('unapproved evaluation image/attempt')
    calls=sum(v.get('forward_calls',0) for s in budget['slots'].values() for v in s['images'].values())
    tiles=sum(v.get('tiles',0) for s in budget['slots'].values() for v in s['images'].values())
    if (calls+1>protocol['runtime']['evaluation_forward_cap'] or tiles+batch_tiles>protocol['runtime']['evaluation_tile_cap']
        or state['forward_calls']+1>profile['forward_calls'] or state['tiles']+batch_tiles>profile['tiles']):
        raise GateError('fixed evaluation forward/tile budget exceeded')
    state['forward_calls']+=1;state['tiles']+=batch_tiles


def evaluate_checkpoint(model,step,binding_sha):
    import csv, numpy as np, torch
    from PIL import Image
    from retinal_m0.inference import tiled_probability
    from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
    from retinal_m0.maples_stratified_science_analysis import hard_cldice
    from retinal_m0.maples_stratified_science_core import rng_state,restore_rng
    if step not in range(200,2001,200): raise GateError('unplanned C evaluation step')
    ref_manifest=read_json(Path(check_protocol()['reference']['science_reference_manifest']))
    source_manifest=read_json(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json')
    sources={r['filename']:r for r in source_manifest['images']}
    dest=OUT/'evaluation'/f'step_{step:06d}'; probs=dest/'probabilities'; dest.mkdir(parents=True,exist_ok=True); probs.mkdir(exist_ok=True)
    metrics_path=dest/'metrics.json'; ckpt=OUT/f'step_{step:04d}.pt'; checkpoint_sha=sha(ckpt); reference_sha=sha(Path(check_protocol()['reference']['science_reference_manifest']))
    if len(ref_manifest['images'])!=28 or len(sources)!=28 or set(sources)!={r['filename'] for r in ref_manifest['images']}: raise GateError('Dev input/reference membership differs')
    if metrics_path.exists():
        saved=read_json(metrics_path)
        if saved.get('method')!='SPATIAL' or saved.get('step')!=step or saved.get('threshold')!=.5 or saved.get('binding_sha256')!=binding_sha or saved.get('checkpoint_sha256')!=checkpoint_sha or saved.get('reference_manifest_sha256')!=reference_sha: raise GateError('existing C evaluation binding drift')
        validate_probability_records(saved.get('images',[]),ref_manifest['images'],probs)
        return saved
    evaluation_authorize=runtime_authorizer(binding_sha)
    budget_path=STAGE/'evaluation_budget.json'
    if not budget_path.exists(): atomic_json(budget_path,{'schema':1,'slots':{}})
    budget=read_json(budget_path); slot_key='SPATIAL:'+str(step); slot=budget['slots'].setdefault(slot_key,{'checkpoint_sha256':checkpoint_sha,'images':{}})
    if slot['checkpoint_sha256']!=checkpoint_sha: raise GateError('evaluation checkpoint changed')
    model_snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    before=rng_state('cuda'); before_digest=__import__('retinal_m0.state',fromlist=['state_digest']).state_digest(before); was=model.training; model_before=__import__('retinal_m0.state',fromlist=['state_digest']).state_digest(model.state_dict()); rows=[]
    try:
        from retinal_m0.maples_stratified_science_core import verify_checkpoint
        validate_release(full_sources=True)
        saved_ckpt=verify_checkpoint(ckpt,expected_step=step,binding_sha=binding_sha,run_dir=OUT)
        model.load_state_dict(saved_ckpt['model'],strict=True); del saved_ckpt; model.eval()
        norm=read_json(CODE/'releases/development_v1/normalization.json'); mu=torch.tensor(norm['mean'])[:,None,None]; sd=torch.tensor(norm['std'])[:,None,None]
        for row in ref_manifest['images']:
            name=row['filename']; destp=probs/(name+'.npy'); state=slot['images'].get(name)
            if state and state.get('status')=='COMPLETE':
                if sha(destp)!=state['probability_sha256']: raise GateError('completed probability file digest changed')
                prob=np.load(destp,allow_pickle=False)
            else:
                if state is not None: raise GateError('incomplete evaluation image cannot be retried automatically')
                inp=sources[name]['inputs']['RGB']
                if sha(inp['path'])!=inp['sha256']: raise GateError('frozen Dev RGB source changed')
                with Image.open(inp['path']) as im: raw=np.asarray(im.convert('RGB')).copy()
                arrays=_load_arrays(row['reference_npz']); fov=arrays['FOV'].astype(bool)
                image=(torch.from_numpy(raw.transpose(2,0,1).copy()).float()/255-mu)/sd
                image*=torch.from_numpy(fov[None].astype(np.float32))
                state={'status':'INCOMPLETE','attempts':1,'forward_calls':0,'tiles':0}; slot['images'][name]=state; atomic_json(budget_path,budget)
                def count_forward(module,inputs):
                    evaluation_authorize()
                    reserve_evaluation_forward(budget,slot_key,name,int(inputs[0].shape[0]),check_protocol());atomic_json(budget_path,budget)
                hook=model.register_forward_pre_hook(count_forward)
                try: prediction,tiles=tiled_probability(model,image,patch=512,stride=256,batch_size=4,device='cuda',amp=False)
                finally: hook.remove()
                prob=prediction.detach().cpu().numpy().astype(np.float32)
                with destp.with_name(destp.name+'.pending').open('wb') as f: np.save(f,prob,allow_pickle=False); f.flush(); os.fsync(f.fileno())
                os.replace(destp.with_name(destp.name+'.pending'),destp)
                profile=check_protocol()['runtime']['evaluation_image_profile'][name]
                if state['tiles']!=profile['tiles'] or state['forward_calls']!=profile['forward_calls'] or int(tiles)!=profile['tiles']:raise GateError('inference geometry differs from frozen evaluation profile')
                state.update(status='COMPLETE',probability_sha256=sha(destp),tiled_count=int(tiles)); atomic_json(budget_path,budget)
            if sha(row['reference_npz'])!=row['reference_npz_sha256']: raise GateError('sealed GT reference drift')
            arrays=_load_arrays(row['reference_npz'])
            if prob.dtype!=np.float32 or list(prob.shape)!=row['native_shape'] or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)): raise GateError('new/saved prediction dtype/shape/range mismatch')
            record=evaluate_image(prob,arrays,'B2',threshold=.5); record['method']='SPATIAL'
            hard=hard_cldice((prob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV'])
            record.update(hard_scalar_details(hard))
            record['hard_cldice']=hard['cldice']; record['topology_precision']=hard['topology_precision']; record['topology_sensitivity']=hard['topology_sensitivity']
            rows.append({'filename':name,'probability_sha256':sha(destp),'RGB_sha256':sources[name]['inputs']['RGB']['sha256'],'reference_npz_sha256':row['reference_npz_sha256'],**record})
            atomic_json(STAGE/'heartbeat.json',{'status':'EVALUATING','step':step,'image':name,'completed_images':len(rows),'pid':os.getpid(),'unix_time':__import__('time').time()})
            print('EVALUATION',step,len(rows),name,flush=True)
        result={'method':'SPATIAL','step':step,'threshold':.5,'checkpoint_sha256':checkpoint_sha,'reference_manifest_sha256':reference_sha,'binding_sha256':binding_sha,'images':rows}
        validate_probability_records(rows,ref_manifest['images'],probs)
        atomic_json(metrics_path,result)
        return result
    finally:
        model.load_state_dict(model_snapshot,strict=True); del model_snapshot
        model.train(was); restore_rng(before,'cuda')
        after=__import__('retinal_m0.state',fromlist=['state_digest']).state_digest(rng_state('cuda'))
        if after!=before_digest: raise GateError('evaluation changed RNG after restoration')
        if __import__('retinal_m0.state',fromlist=['state_digest']).state_digest(model.state_dict())!=model_before: raise GateError('evaluation changed model parameters')


def analyze_endpoint(binding_sha, *, result_root=OUT, fixture_d_from_a0=False):
    import csv,numpy as np
    from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
    from retinal_m0.maples_stratified_science_analysis import hard_cldice,paired_bootstrap,bootstrap_indices,fixed_gt_metric_validity,descriptive_strata
    root=Path(result_root); ref=read_json(Path(check_protocol()['reference']['science_reference_manifest']))['images']
    a0_binding=read_json(Path(check_protocol()['reference']['A0_reuse_binding']))
    a0={r['filename']:r for r in a0_binding['probability_maps']}
    expected_names=[r['filename'] for r in ref]
    if len(ref)!=28 or list(a0)!=expected_names: raise GateError('sealed Dev28/A0 image membership or order mismatch')
    cpath=root/'evaluation/step_002000/metrics.json'
    if fixture_d_from_a0:
        c_records=[]
        for refrow in ref:
            name=refrow['filename']; path=Path(a0[name]['path'])
            if sha(path)!=a0[name]['sha256']: raise GateError('A0 fixture probability digest mismatch')
            prob=np.load(path,allow_pickle=False); arrays=_load_arrays(refrow['reference_npz'])
            rr=evaluate_image(prob,arrays,'B2',threshold=.5); rr['method']='SPATIAL'
            hard=hard_cldice((prob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV'])
            rr.update(hard_scalar_details(hard))
            rr.update(hard_cldice=hard['cldice'],topology_precision=hard['topology_precision'],topology_sensitivity=hard['topology_sensitivity'])
            c_records.append({'filename':name,'probability_sha256':a0[name]['sha256'],**rr})
        c={'method':'SPATIAL','step':2000,'threshold':.5,'binding_sha256':'A0_FIXTURE_ONLY','images':c_records}
    else:
        c=read_json(cpath)
        if c.get('method')!='SPATIAL' or c.get('step')!=2000 or c.get('threshold')!=.5 or c.get('binding_sha256')!=binding_sha: raise GateError('C endpoint metric binding mismatch')
        if len(c.get('images',[]))!=28 or [r.get('filename') for r in c['images']]!=expected_names: raise GateError('C endpoint must contain exactly the sealed Dev28 in order')
        for refrow,row in zip(ref,c['images']):
            probpath=root/'evaluation/step_002000/probabilities'/(row['filename']+'.npy')
            if sha(probpath)!=row['probability_sha256']: raise GateError('C endpoint probability digest mismatch')
            prob=np.load(probpath,allow_pickle=False)
            if prob.dtype!=np.float32 or list(prob.shape)!=refrow['native_shape'] or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)): raise GateError('C endpoint probability dtype/shape/range mismatch')
    b1b2_root=ROOT/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022'
    secondary={}
    for method in ('B1','B2'):
        q=read_json(b1b2_root/f'runs/{method}/evaluation/step_002000/metrics.json')
        if q.get('method')!=method or q.get('step')!=2000 or q.get('threshold')!=.5 or len(q.get('images',[]))!=28 or [r.get('filename') for r in q['images']]!=expected_names: raise GateError(method+' fixed endpoint reference membership/binding mismatch')
        secondary[method]={r['filename']:r for r in q['images']}
    if len(c.get('images',[]))!=28 or [r.get('filename') for r in c['images']]!=expected_names: raise GateError('endpoint image count mismatch; zip truncation prohibited')
    rows=[]; fixed={}; all_metrics=set()
    names_all=['SPATIAL','A0','B1','B2']
    for i,refrow in enumerate(ref):
        name=expected_names[i]; arrays=_load_arrays(refrow['reference_npz']); crow=c['images'][i]
        if crow['filename']!=name: raise GateError('C endpoint image order mismatch')
        ap=a0[name]
        if sha(ap['path'])!=ap['sha256']: raise GateError('A0 endpoint probability digest mismatch')
        aprob=np.load(ap['path'],allow_pickle=False)
        if aprob.dtype!=np.float32 or list(aprob.shape)!=refrow['native_shape'] or not np.isfinite(aprob).all() or np.any((aprob<0)|(aprob>1)): raise GateError('A0 probability dtype/shape/range mismatch')
        ar=evaluate_image(aprob,arrays,'B2',threshold=.5); ar['method']='A0'
        ah=hard_cldice((aprob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV'])
        ar.update(hard_scalar_details(ah))
        ar.update(hard_cldice=ah['cldice'],topology_precision=ah['topology_precision'],topology_sensitivity=ah['topology_sensitivity'])
        methods={'SPATIAL':crow,'A0':ar}
        for method in ('B1','B2'):
            old=secondary[method][name]
            probpath=b1b2_root/f'runs/{method}/evaluation/step_002000/probabilities/{name}.npy'
            if sha(probpath)!=old['probability_sha256']: raise GateError(method+' endpoint probability digest mismatch')
            prob=np.load(probpath,allow_pickle=False)
            if prob.dtype!=np.float32 or list(prob.shape)!=refrow['native_shape'] or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)): raise GateError(method+' probability dtype/shape/range mismatch')
            full=dict(old); hard=hard_cldice((prob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV'])
            full.update(hard_scalar_details(hard))
            full.update(hard_cldice=hard['cldice'],topology_precision=hard['topology_precision'],topology_sensitivity=hard['topology_sensitivity'])
            methods[method]=full
        scalar={method:metric_values(value) for method,value in methods.items()}
        for value in scalar.values(): all_metrics.update(value)
        rows.append({'filename':name,'methods':scalar,'full_records':methods,
                     'probability_sha256':{m:(crow['probability_sha256'] if m=='SPATIAL' else ap['sha256'] if m=='A0' else secondary[m][name]['probability_sha256']) for m in names_all}})
        valid=fixed_gt_metric_validity(arrays,refrow)
        for metric,is_valid in valid.items():
            if is_valid: fixed.setdefault(metric,[]).append(name)
    metric_names=sorted(valid)
    by={r['filename']:r for r in rows}; analysis={}; cache={}; contrasts=(('SPATIAL','A0'),('SPATIAL','B1'),('SPATIAL','B2')); all_indices={}
    for metric in metric_names:
        ids=fixed.get(metric,[])
        if not ids:
            analysis[metric]={'n_images':0,'NA_reason':'GT_fixed_zero_valid_images'}; continue
        n=len(ids); ix=cache.setdefault(n,bootstrap_indices(n,seed=20261016,replicates=1000)); analysis[metric]={}
        for left,right in contrasts:
            try:
                result=paired_bootstrap([by[x]['methods'][left].get(metric) for x in ids],[by[x]['methods'][right].get(metric) for x in ids],indices=ix,seed=20261016,replicates=1000)
                result.pop('indices',None)  # retained once in bootstrap_indices_by_n
            except ValueError as exc:
                stopped={'status':'STOPPED_ON_FIXED_SET_NA','comparison':'SPATIAL-A0 primary; SPATIAL-B1/SPATIAL-B2 supplemental','endpoint':2000,
                         'binding_sha256':binding_sha,'fixed_ids_by_metric':fixed,'bootstrap_indices_by_n':{str(n):ix.tolist()},
                         'per_image':rows,'stop_reason':str(exc),'seed_SD':None}
                atomic_json(root/'analysis/endpoint_2000.json',stopped)
                raise GateError('fixed-image analysis stopped on NA; full per-image records retained: '+str(exc)) from exc
            analysis[metric][left+'-'+right]={'fixed_ids':ids,**result}
        all_indices[str(n)]=ix.tolist()
    from retinal_m0.maples_stratified_science_analysis import dice_readout
    interpretation=dice_readout(analysis)
    out={'status':'COMPLETE_FIXTURE_ONLY' if fixture_d_from_a0 else 'COMPLETE','comparison':'SPATIAL-A0 primary; SPATIAL-B1/SPATIAL-B2 supplemental','endpoint':2000,'threshold':.5,
         'binding_sha256':binding_sha if not fixture_d_from_a0 else 'A0_FIXTURE_ONLY','fixed_ids_by_metric':fixed,'bootstrap_indices_by_n':all_indices,
         'metrics':analysis,'per_image':rows,'seed_SD':None,
         'readout':interpretation,
         'old_neighborhood_per_image_numerators_denominators_retained':True,'Tprec_Tsens_NA_retained':True,
         'full_path_records_retained':True,'official_Test_read':False}
    out['path_strata'],out['native_size_strata']=descriptive_strata(rows,ref)
    out['topology_diagnostics']={}
    for method in names_all:
        out['topology_diagnostics'][method]={}
        for metric in ('topology_precision','topology_sensitivity'):
            vals=[r['full_records'][method][metric] for r in rows];defined=[v for v in vals if v is not None]
            out['topology_diagnostics'][method][metric]={'image_macro':float(np.mean(defined)) if defined else None,'n_defined':len(defined),'n_NA':28-len(defined),'per_image_NA':[{ 'filename':r['filename'],'reason':r['full_records'][method].get(metric+'_NA_reason')} for r in rows if r['full_records'][method][metric] is None],'scope':'original diagnostic empty rule; not a new primary/guard or selected validity bootstrap'}
    out['absolute_means']={m:{metric:float(np.mean([by[name]['methods'][m][metric] for name in ids])) for metric,ids in fixed.items()} for m in names_all}
    out['per_image_changes']={}
    from retinal_m0.maples_branch_science_analysis import change_summary
    for metric,ids in fixed.items():
        out['per_image_changes'][metric]={}
        for right in ('A0','B1','B2'):
            deltas=[{'filename':name,'delta':by[name]['methods']['SPATIAL'][metric]-by[name]['methods'][right][metric]} for name in ids]
            out['per_image_changes'][metric]['SPATIAL-'+right]=change_summary(deltas,metric)
    report=root/'analysis/endpoint_2000.json'; report.parent.mkdir(parents=True,exist_ok=True); atomic_json(report,out)
    # Flatten all scalar and ratio records; preserve NA, numerators, denominators, and reasons.
    methods=['SPATIAL','A0','B1','B2']; field_values=set()
    def flatten(record,prefix):
        result={}
        for key,value in record.items():
            if key in ('path_records','excluded_path_records'): continue
            if isinstance(value,dict) and {'value','numerator','denominator'}<=set(value):
                for suffix,k in [('_value','value'),('_numerator','numerator'),('_denominator','denominator'),('_NA_reason','NA_reason')]: result[prefix+key+suffix]=value.get(k)
            elif isinstance(value,(str,int,float,bool)) or value is None: result[prefix+key]=value
        return result
    csvrows=[]
    for row in rows:
        qrow={'filename':row['filename']}
        for method in methods:qrow.update(flatten(row['full_records'][method],method+'_'))
        field_values.update(qrow);csvrows.append(qrow)
    with (report.parent/'per_image_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['filename']+sorted(field_values-{'filename'}); writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(csvrows)
    paths=[]
    for row in rows:
        for method in methods:
            record=row['full_records'][method]
            for role,key in [('candidate','path_records'),('excluded','excluded_path_records')]:
                for item in record.get(key,[]): paths.append({'filename':row['filename'],'method':method,'path_role':role,**item})
    if paths:
        fields=sorted({k for r in paths for k in r})
        with (report.parent/'per_path_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(paths)
    lines=['# 空间分层采样固定2000终点结果'+('（A0替身CPU接线检查，非训练结果）' if fixture_d_from_a0 else ''),'','仅单seed2026、既有Dev28，反复探索，不是独立验证。阈值0.5。旧邻域9有效图/77位置及19NA完整保留。','', '|指标|SPATIAL|A0|SPATIAL-A0|95%区间|有效图|','|---|---:|---:|---:|---|---:|']
    for metric in metric_names:
        x=analysis[metric].get('SPATIAL-A0',{})
        lines.append('|'+metric+'|'+str(out['absolute_means']['SPATIAL'].get(metric))+'|'+str(out['absolute_means']['A0'].get(metric))+'|'+str(x.get('estimate'))+'|['+str(x.get('lower'))+','+str(x.get('upper'))+']|'+str(x.get('n_images',0))+'|')
    lines += ['', '## 预定判读',json.dumps(out['readout'],ensure_ascii=False,indent=2),'','全部分母/NA/路径/逐图变化、SPATIAL-B1/SPATIAL-B2补充比较、bootstrap索引与估计见同目录JSON/CSV。不以正面效果作为工程通过标准。单seed波动SD=NA。']
    (report.parent/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return out


def _worker(parent_pid,nonce):
    import torch,shutil,time
    from retinal_m0.maples_stratified_science_core import Budget,run_updates,verify_checkpoint,rng_state,restore_rng,state_digest
    status,release,binding_sha=worker_authority(parent_pid,nonce)
    import ctypes,signal
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent_pid: raise GateError('parent-death guard failed')
    install_dataset_guard()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.is_initialized(): raise GateError('worker must initialize only the released GPU UUID')
    gpu=_gpu0_resources()
    if torch.cuda.device_count()!=1: raise GateError('exactly GPU0 must be visible')
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(min(.8,8589934592/(gpu['total_mib']*1048576)),0)
    model,opt,scaler,dataset,pairs,cfg=_build_training_objects('cuda')
    summary=read_json(ROOT/'experiments/t0_tw_core_20261016/runs/seed_2026/T0/summary.json')
    origin={key:state_digest(value) for key,value in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state('cuda'))]}
    for key in ('model','optimizer','scaler'):
        if hashlib.sha256(origin[key].encode()).hexdigest()!=summary['initial_'+key]: raise GateError('fresh initialization differs from original T0: '+key)
    legacy_rng=rng_state('cuda');legacy_rng['cuda']=[legacy_rng.pop('cuda0')]
    if hashlib.sha256(state_digest(legacy_rng).encode()).hexdigest()!=summary['initial_rng']: raise GateError('fresh original T0 RNG differs')
    atomic_json(STAGE/'initialization_receipt.json',{'status':'FRESH_ORIGINAL_M0_AND_OPTIMIZER_SCALER','initial_state':origin,'M0_sha256':cfg['source_m0_sha256'],'binding_sha256':binding_sha,'engineering_weights_used':False,'pid':os.getpid()})
    ledger_path=STAGE/'science_budget.json';ledger=Budget(ledger_path,binding_sha);run_dir=OUT
    if not ledger_path.exists(): raise GateError('missing prereleased scientific ledger; no implicit reset')
    if not run_dir.exists():
        if ledger.snapshot()['charged_events']!=0: raise GateError('budget exists without run directory; no reset')
        start=0
    else:
        if not (run_dir/'last.pt').is_file(): raise GateError('existing run directory has no recoverable last checkpoint')
        raw=torch.load(run_dir/'last.pt',map_location='cpu',weights_only=True);start=int(raw['step']);del raw
        ck=verify_checkpoint(run_dir/'last.pt',expected_step=start,binding_sha=binding_sha,run_dir=run_dir,ledger=ledger)
        if ck['initial_state']!=origin: raise GateError('restored trajectory has different fresh origin')
        if start%200==0:
            milestone=verify_checkpoint(run_dir/f'step_{start:04d}.pt',expected_step=start,binding_sha=binding_sha,run_dir=run_dir)
            if milestone['digests']!=ck['digests'] or milestone['budget']!=ck['budget']: raise GateError('last vs matching milestone inconsistent')
            del milestone
        fresh_rng=rng_state('cuda');fresh_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};fresh_opt=opt.state_dict();fresh_scaler=scaler.state_dict()
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'],'cuda')
        loader=torch.Generator();loader.set_state(ck['loader_generator'])
        for key,value in [('model',model.state_dict()),('optimizer',opt.state_dict()),('scaler',scaler.state_dict()),('rng',rng_state('cuda')),('loader_generator',loader.get_state())]:
            if state_digest(value)!=ck['digests'][key]: raise GateError('full restored runtime state differs: '+key)
        atomic_json(STAGE/f'restore_admission_{start:04d}.json',{'passed':True,'step':start,'last_sha256':sha(run_dir/'last.pt'),'binding_sha256':binding_sha,'before_any_resumed_evaluation':True})
        del ck
        for step in range(200,start+1,200):
            ck=verify_checkpoint(run_dir/f'step_{step:04d}.pt',expected_step=step,binding_sha=binding_sha,run_dir=run_dir);del ck
            evaluate_checkpoint(model,step,binding_sha)
        if start<2000:
            model.load_state_dict(fresh_model,strict=True);opt.load_state_dict(fresh_opt);scaler.load_state_dict(fresh_scaler);restore_rng(fresh_rng,'cuda')
        del fresh_model,fresh_opt,fresh_scaler,fresh_rng
    authorize=runtime_authorizer(binding_sha);wall_start=time.monotonic()
    def heartbeat(step,payload):
        snap=payload['budget'];free=shutil.disk_usage(STAGE).free
        hb={'status':'RUNNING','step':step,'target':2000,'pid':os.getpid(),'unix_time':time.time(),'elapsed_seconds':payload['elapsed_seconds'],
            'last_sha256':sha(run_dir/'last.pt'),'charged_events':snap['charged_events'],'free_disk_bytes':free,
            'GPU_peak_allocated_MiB':torch.cuda.max_memory_allocated(0)/1048576,'GPU_peak_reserved_MiB':torch.cuda.max_memory_reserved(0)/1048576}
        atomic_json(STAGE/'heartbeat.json',hb)
        if step<=32:
            row=json.loads((OUT/'train.jsonl').read_text().splitlines()[step-1])
            if row['pair']!=pairs[step-1] or row['loss']!=row['A0_loss'] or snap['gradient_calls']!=step:raise GateError('first32 actual sampler/loss/budget mismatch')
            if payload['trajectory_sha256']!=check_protocol()['reference']['trajectory_sha256']:raise GateError('checkpoint sampler identity mismatch')
            atomic_json(STAGE/'first32_input_and_loss_match.json',{'matched_steps':step,'new_GPU_calculations':0,'same_as_A0_loss_primitive':True,'input_comparison':'frozen candidate hashes, not original A0 crops','binding_sha256':binding_sha})
        if step==1 or step%200==0: _update_current_status(execution_started=True,state='RELEASED',active_worker_pid=os.getpid(),completed_steps=step,charged_updates=snap['charged_events'])
        print('UPDATE_COMMITTED',step,2000,'elapsed',round(payload['elapsed_seconds'],2),flush=True)
        if free<21474836480: raise GateError('free disk below20GiB after committed update')
        if step%200==0:
            validate_release(full_sources=True);evaluate_checkpoint(model,step,binding_sha)
            atomic_json(STAGE/'heartbeat.json',{**hb,'status':'RUNNING','evaluation_complete':step,'unix_time':time.time()})
    if start<2000:
        run_updates(model=model,optimizer=opt,scaler=scaler,dataset=dataset,track='SPATIAL',start=start,end=2000,out=run_dir,budget=ledger,
                    binding_sha=binding_sha,expected_pairs=pairs,device='cuda',authorize=authorize,on_update=heartbeat)
    validate_release(full_sources=True)
    final=verify_checkpoint(run_dir/'last.pt',expected_step=2000,binding_sha=binding_sha,run_dir=run_dir,ledger=ledger);del final
    for step in range(200,2001,200):
        ck=verify_checkpoint(run_dir/f'step_{step:04d}.pt',expected_step=step,binding_sha=binding_sha,run_dir=run_dir);del ck
        evaluate_checkpoint(model,step,binding_sha)
    report=analyze_endpoint(binding_sha);snap=ledger.snapshot()
    evaluation=read_json(STAGE/'evaluation_budget.json');slots=evaluation['slots']
    if set(slots)!={'SPATIAL:'+str(x) for x in range(200,2001,200)} or any(len(x['images'])!=28 for x in slots.values()): raise GateError('missing fixed Dev evaluation slots')
    images=[x for s in slots.values() for x in s['images'].values()]
    if any(x['status']!='COMPLETE' or x['attempts']!=1 for x in images) or sum(x['forward_calls'] for x in images)!=1900 or sum(x['tiles'] for x in images)!=7450: raise GateError('evaluation budget incomplete or extra inference occurred')
    if snap['verified_successes']!=2000 or snap['charged_events']!=2000 or snap['forward_calls']!=2000 or snap['gradient_calls']!=2000 or snap['failed_events'] or snap['uncertain_events'] or snap['failed_compute']: raise GateError('C scientific budget endpoint mismatch')
    result=run_dir/'analysis/endpoint_2000.json'
    atomic_json(STAGE/'completion.json',{'status':'SPATIAL2000_TRAINED_AND_EVALUATED','updates':2000,'seed':2026,'binding_sha256':binding_sha,'GPU0':gpu,'endpoint_report':str(result),'result_paths':[str(result),str(result.with_name('report.md'))],'budget':snap,'worker_wall_seconds':time.monotonic()-wall_start})
    atomic_json(STAGE/'heartbeat.json',{'status':'COMPLETE','step':2000,'target':2000,'unix_time':time.time(),'pid':os.getpid()})
    return 0


def launch_authorized():
    import secrets
    status,release,binding_sha=validate_release()
    # The lock spans controller and worker; a busy shared GPU lock is a hard stop.
    with exclusive_lock(STAGE/'SCIENCE_RUNTIME.lock'),exclusive_lock(ROOT/'experiments/a14_gpu0.lock'):
        if (STAGE/'RUN_SESSION.json').exists() or OUT.exists(): raise GateError('science already attempted; no implicit restart/resume')
        from retinal_m0.maples_stratified_science_core import Budget
        initial=Budget(STAGE/'science_budget.json',binding_sha).snapshot()
        if initial['status']!='READY' or initial['tracks'] or any(initial[x]!=0 for x in ['charged_events','forward_calls','gradient_calls','verified_successes','failed_compute']): raise GateError('fresh prereleased science ledger required')
        gpu=_gpu0_resources()
        nonce=secrets.token_hex(24); child_env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONPATH=str(CODE),SPATIAL_STRATIFIED_SESSION_NONCE=nonce,PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                                                    PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--parent-pid',str(os.getpid()),'--nonce',nonce]
        proc=subprocess.Popen(cmd,cwd=CODE,env=child_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        session={'controller_pid':os.getpid(),'worker_pid':proc.pid,'controller_token':process_token(os.getpid()),'worker_token':process_token(proc.pid),'nonce':nonce,'release_sha256':sha(STAGE/'SCIENCE_RELEASE.json'),'binding_sha256':binding_sha,'GPU0':gpu}
        atomic_json(STAGE/'RUN_SESSION.json',session)
        log=STAGE/'controller.log'
        with log.open('a',encoding='utf-8') as f:
            for line in proc.stdout: f.write(line); f.flush()
        code=proc.wait()
        if code:
            ledger_path=STAGE/'science_budget.json'; snap=None
            if ledger_path.is_file():
                ledger=__import__('retinal_m0.maples_stratified_science_core',fromlist=['Budget']).Budget(ledger_path,binding_sha); snap=ledger.snapshot()
                pending=snap['tracks'].get('SPATIAL',{}).get('pending')
                if pending is not None: ledger.resolve('SPATIAL','uncertain' if pending.get('started') else 'failed'); snap=ledger.snapshot()
                ledger.stop('controller observed worker exit '+str(code)); snap=ledger.snapshot()
            failure={'status':'FAILED_STOP','worker_exit_code':code,'execution_started':bool(snap and snap.get('verified_successes',0)>0),'budget':snap,'log':str(log)}
            atomic_json(STAGE/'runtime_failure.json',failure)
            _update_current_status(state='FAILED_STOP',GPU_released=False,execution_started=failure['execution_started'],active_worker_pid=None,stop_reason='worker_exit_'+str(code),failure_path=str(STAGE/'runtime_failure.json'))
            raise GateError('C worker stopped with exit '+str(code)+'; failure evidence saved at '+str(STAGE/'runtime_failure.json'))
        completion=read_json(STAGE/'completion.json')
        _update_current_status(state='CONSUMED',GPU_released=False,execution_started=True,training_complete=True,active_worker_pid=None,completed_steps=2000,charged_updates=2000,result_paths=completion.get('result_paths',[]))
    return 0


def main():
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cpu-preflight',action='store_true'); g.add_argument('--run-authorized',action='store_true')
    g.add_argument('--session-worker',action='store_true',help=argparse.SUPPRESS); g.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--parent-pid',type=int); p.add_argument('--nonce',default='')
    a=p.parse_args()
    if a.session_worker: return session_worker(a.parent_pid,a.nonce)
    if a.worker:
        _worker(a.parent_pid,a.nonce); return 0
    if a.cpu_preflight:
        result=cpu_preflight(); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
    return launch_authorized()


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}',file=sys.stderr); raise SystemExit(2)
