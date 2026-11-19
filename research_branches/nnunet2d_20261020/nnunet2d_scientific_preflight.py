from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, pickle, subprocess, sys, time, traceback
from importlib.metadata import version as package_version
from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np
import torch
from nnunet2d_scientific_runtime import HardBudget, StopScientificRun, finite_update, lr_poly, install_seeded_worker_adapter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

BASE=pathlib.Path('${RETINA_PROJECT_ROOT}')
FREEZE=BASE/'docs/nnunet2d_seed2026_100epochs_20261020'
PROTO=json.loads((FREEZE/'protocol.json').read_text())
MAN=json.loads((FREEZE/'source_manifest.json').read_text())
DEV=BASE/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json'
ROOT=pathlib.Path(PROTO['paths']['scientific_root'])
PREFLIGHT=ROOT/'preflight'
BRANCH=pathlib.Path(PROTO['paths']['branch_code'])
PYTHON=pathlib.Path(PROTO['paths']['python'])
EVIDENCE=PREFLIGHT/'verification.json'

class SeedProbe:
    def __init__(self): self.thread_id=0; self.n=0
    def set_thread_id(self,i): self.thread_id=i
    def __iter__(self): return self
    def __next__(self):
        import random
        self.n+=1
        return {'python':random.random(),'numpy':float(np.random.random()),'torch':float(torch.rand(()).item()),'n':self.n}

def run_seed_probe(seed):
    loader=SeedProbe()
    augmenter=NonDetMultiThreadedAugmenter(data_loader=loader,transform=None,num_processes=1,num_cached=2,seeds=[seed],pin_memory=False,wait_time=0.005)
    values=[]
    try:
        with patch('torch.cuda.is_available',return_value=False):
            for _ in range(4): values.append(next(augmenter))
    finally:
        augmenter._finish()
    return values

def main():
    global PREFLIGHT, EVIDENCE
    parser=argparse.ArgumentParser()
    parser.add_argument('--repair-audit-dir', help='New CPU-only repair evidence directory under the scientific root; never authorizes launch')
    args=parser.parse_args()
    repair_only=args.repair_audit_dir is not None
    if repair_only:
        audit=pathlib.Path(args.repair_audit_dir).resolve()
        if not audit.is_relative_to(ROOT.resolve()) or audit == ROOT.resolve():
            raise SystemExit('Repair evidence must be a new subdirectory of the existing scientific root')
        audit.mkdir(parents=True,exist_ok=False)
        PREFLIGHT=audit
        EVIDENCE=PREFLIGHT/'verification.json'
    PREFLIGHT.mkdir(parents=True,exist_ok=True)
    checks={}; details={}; errors=[]
    def check(name,ok,detail=None):
        checks[name]=bool(ok); details[name]=detail
        if not ok: errors.append(name)
    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        return h.hexdigest()
    # Frozen package receipts and source references.
    receipt=json.loads((FREEZE/'freeze_receipt.json').read_text())
    ph=sha(FREEZE/'protocol.json')
    check('freeze_receipt',ph==receipt['protocol_sha256']==PROTO['freeze']['protocol_sha256'] if 'freeze' in PROTO else ph==receipt['protocol_sha256'],{'protocol_sha256':ph,'receipt':receipt['protocol_sha256']})
    for name, expected in receipt['files'].items():
        actual=sha(FREEZE/name)
        check('freeze_'+name,actual==expected,{'expected':expected,'actual':actual})
    manifest_results=[]
    for item in MAN['files']:
        p=pathlib.Path(item['path']); actual=sha(p) if p.is_file() else 'MISSING'
        manifest_results.append({'path':str(p),'expected':item['sha256'],'actual':actual,'match':actual==item['sha256']})
    check('source_manifest_files',all(x['match'] for x in manifest_results),{'count':len(manifest_results),'mismatch_count':sum(not x['match'] for x in manifest_results)})
    t0_results=[]
    for item in MAN['T0_seed2026_step2000_predictions']:
        p=pathlib.Path(item['path']); actual=sha(p) if p.is_file() else 'MISSING'
        t0_results.append({'case':item['filename'],'path':str(p),'expected':item['sha256'],'actual':actual,'match':actual==item['sha256']})
    check('T0_terminal_probabilities',len(t0_results)==28 and all(x['match'] for x in t0_results),{'count':len(t0_results),'mismatch_count':sum(not x['match'] for x in t0_results)})
    # Train/Dev membership only, no Dev RGB/GT opened.
    train=json.loads((BASE/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json').read_text())
    dev=json.loads((BASE/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json').read_text())
    train_ids=[x['filename'] for x in train['rows']]
    dev_ids=[x['filename'] for x in dev['images']]
    raw_rows=list(csv.DictReader(open(BASE/'experiments/nnunet2d_branch_20261020/raw/train_export_manifest.csv',newline='')))
    raw_ids=[x['id'] for x in raw_rows]
    check('train_dev_membership',len(train_ids)==110 and len(set(train_ids))==110 and len(dev_ids)==28 and not (set(train_ids)&set(dev_ids)) and raw_ids==train_ids,{'train_count':len(train_ids),'dev_name_count':len(dev_ids),'overlap':len(set(train_ids)&set(dev_ids)),'raw_train_count':len(raw_ids),'dev_pixels_or_labels_read':False,'test_access':False})
    # Existing Train-only dataset and plans are immutable, counts and native sizes match freeze.
    exp=pathlib.Path(PROTO['paths']['data_root'])
    pre=pathlib.Path(PROTO['paths']['preprocessed'])/'Dataset501_MAPLES/nnUNetPlans_8GBtarget_2d'
    pkl=sorted(pre.glob('*.pkl')); all_b2nd=sorted(pre.glob('*.b2nd')); data=[x for x in all_b2nd if not x.name.endswith('_seg.b2nd')]; seg=sorted(pre.glob('*_seg.b2nd'))
    plan_path=pathlib.Path(PROTO['paths']['preprocessed'])/'Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'
    plan_hash=sha(plan_path)
    check('plans_and_preprocessed',plan_hash==PROTO['recipe']['plans_sha256'] and len(pkl)==110 and len(data)==110 and len(seg)==110,{'plans_sha256':plan_hash,'properties':len(pkl),'data':len(data),'segmentation':len(seg),'all_b2nd_including_seg':len(all_b2nd)})
    # Reconcile immutable original engineering ledger using independent phase logs and state verification.
    eng=exp/'engineering'; ledger=json.loads((eng/'budget_ledger.json').read_text()); ev=json.loads((eng/'verification.json').read_text())
    binding=json.loads((eng/'execution_binding.json').read_text())
    actual_ledger_sha=sha(eng/'budget_ledger.json')
    declared_ledger_sha=binding.get('budget_ledger_sha256') or binding.get('inputs',{}).get('budget_ledger_sha256') or MAN.get('engineering_budget_binding_discrepancy',{}).get('binding_declared_sha256')
    phase_seconds=sum(float(x.get('elapsed_seconds',0)) for x in ledger['phases'])
    probe_seconds=float(ledger['validation_probe']['elapsed_seconds'])
    summed=phase_seconds+probe_seconds
    declared_seconds=float(ledger['gpu_window_elapsed_seconds'])
    phase_attempts=sum(int(x.get('optimizer_attempts',0)) for x in ledger['phases'])
    log_counts={}
    for phase in ledger['phases']:
        lp=pathlib.Path(phase['attempts_log'])
        records=[json.loads(line) for line in open(lp,encoding='utf-8')]
        successes=sum(r.get('event')=='optimizer_attempt_success' and r.get('amp_skipped') is False and r.get('loss_finite') is True and r.get('gradient_finite') is True for r in records)
        attempts=len(set(r['attempt'] for r in records if r.get('event')=='optimizer_attempt_start'))
        log_counts[phase['name']]={'declared':phase['optimizer_attempts'],'logged_records':len(records),'logged_attempt_starts':attempts,'logged_successful_updates':successes}
    restore=all(v.get('equal') is True for v in ev.get('restore_comparison',{}).values())
    reconciliation={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'read-only reconciliation; original engineering files are not modified by this preflight','binding_declared_ledger_sha256':declared_ledger_sha,'current_ledger_sha256':actual_ledger_sha,'source_manifest_expected_current_ledger_sha256':next((x['sha256'] for x in MAN['files'] if x['path']==str(eng/'budget_ledger.json')),None),'phase_seconds_sum':phase_seconds,'train_internal_probe_seconds':probe_seconds,'recomputed_total_seconds':summed,'conservative_ledger_total_seconds':declared_seconds,'attempts':phase_attempts,'attempt_log_counts':log_counts,'restore_state_comparison_pass':restore,'resolution':'The original execution binding predates the appended Train-only validation probe. Preserve that binding. Phase logs sum 222.736955810s; the probe accounts for 24.021517600s; the ledger conservatively declares 246.771468868s versus recomputed 246.758473410s.'}
    (PREFLIGHT/'budget_binding_reconciliation.json').write_text(json.dumps(reconciliation,indent=2,ensure_ascii=False)+'\n')
    check('engineering_budget_reconciliation',actual_ledger_sha==reconciliation['source_manifest_expected_current_ledger_sha256'] and phase_attempts==100 and all(v['declared']==v['logged_attempt_starts']==v['logged_successful_updates'] for v in log_counts.values()) and restore and summed<=declared_seconds,{'current_ledger_sha256':actual_ledger_sha,'binding_declared_sha256':declared_ledger_sha,'attempts':phase_attempts,'phase_logs':log_counts,'phase_seconds_sum':phase_seconds,'probe_seconds':probe_seconds,'ledger_seconds':declared_seconds,'state_comparison_pass':restore})
    # Runtime versions and source hashes of installed packages are already included above; confirm import identity.
    import nnunetv2, sys as _sys
    check('runtime_versions',sys.version_info[:2]==(3,10) and torch.__version__=='2.4.0+cu121' and torch.version.cuda=='12.1' and package_version('nnunetv2')=='2.8.1' and torch.cuda.is_initialized() is False,{'python':sys.version,'torch':torch.__version__,'cuda_build':torch.version.cuda,'nnunetv2':package_version('nnunetv2'),'cuda_initialized':torch.cuda.is_initialized()})
    # Worker-seed plumbing test uses the frozen one-worker-per-pool CPU policy.
    os.environ['nnUNet_n_proc_DA']='1'
    os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
    used=install_seeded_worker_adapter()
    with patch('torch.cuda.is_available',return_value=False):
        a=run_seed_probe(2026); b=run_seed_probe(2026); c=run_seed_probe(2027)
    seq=lambda xs:[(x['python'],x['numpy'],x['torch']) for x in xs]
    seed_pass=seq(a)==seq(b) and seq(a)!=seq(c) and all(x['n']==i+1 for xs in (a,b,c) for i,x in enumerate(xs))
    check('worker_seed_plumbing',seed_pass,{'effective_train_worker_seed':[2026],'effective_internal_monitor_worker_seed':[2027],'repeat_same_seed_identical':seq(a)==seq(b),'different_seed_differs':seq(a)!=seq(c),'sample_count':4,'cuda_initialized':torch.cuda.is_initialized()})
    # Optimizer attempt/time caps and failure handling are CPU-only unit probes.
    hb=HardBudget(24999,0.0); hb.before_attempt(); hb.record(finite=True,amp_skipped=False,grad_finite=True)
    over_attempt=False
    try: HardBudget(25000,0).before_attempt()
    except StopScientificRun: over_attempt=True
    time_guard=False
    try: HardBudget(0,34200).before_attempt()
    except StopScientificRun: time_guard=True
    amp_stop=finite_update(1.0,1.0,8.0,8.0)['amp_skipped'] is False and finite_update(1.0,1.0,8.0,4.0)['amp_skipped'] is True
    nonfinite_stop=not finite_update(float('nan'),1.0,8.0,8.0)['finite_loss'] and not finite_update(1.0,float('inf'),8.0,8.0)['finite_gradient_norm']
    check('hard_budget_and_failure_guards',over_attempt and time_guard and amp_stop and nonfinite_stop,{'attempt_cap_exact':True,'stop_at_25000':over_attempt,'training_reserve_stop_at_34200s':time_guard,'amp_skip_detected':amp_stop,'nonfinite_loss_or_gradient_detected':nonfinite_stop})
    lr=[lr_poly(0.01,s) for s in range(100)]
    expected=[0.01*(1-s/100)**0.9 for s in range(100)]
    check('poly_lr_100_epoch_horizon',np.allclose(lr,expected,rtol=0,atol=1e-15) and lr[0]==0.01 and lr[-1]>0 and len(lr)==100,{'steps':100,'first_lr':lr[0],'last_lr':lr[-1],'epoch_100_defined':False})
    # CPU-only official inverse-export smoke tests with synthetic logits and serialized Train geometry.
    plans=json.loads(plan_path.read_text()); dataset=json.loads((pathlib.Path(PROTO['paths']['preprocessed'])/'Dataset501_MAPLES/dataset.json').read_text())
    # Exercise the exact shared constructor used by the scientific runner, without
    # initialize(), a model, optimizer, CUDA context, or dataset pixels. A separate
    # process redirects constructor logger metadata into this audit only.
    probe_env=os.environ.copy()
    probe_env.update({'CUDA_VISIBLE_DEVICES':'','nnUNet_compile':'false','PYTHONDONTWRITEBYTECODE':'1'})
    probe=subprocess.run([str(PYTHON),'-B',str(BRANCH/'nnunet2d_scientific_constructor.py'),
                          '--cpu-probe','--output-dir',str(PREFLIGHT/'constructor_probe')],
                         env=probe_env,text=True,capture_output=True,timeout=120)
    (PREFLIGHT/'constructor_probe.log').write_text(probe.stdout+probe.stderr)
    probe_result_path=PREFLIGHT/'constructor_probe/result.json'
    probe_result=json.loads(probe_result_path.read_text()) if probe_result_path.is_file() else {'status':'STOP','return_code':probe.returncode}
    check('shared_scientific_trainer_constructor',probe.returncode==0 and probe_result.get('status')=='PASS',probe_result)
    pm=PlansManager(plans); cm=pm.get_configuration('2d'); lm=pm.get_label_manager(dataset)
    size_results=[]
    torch.set_num_threads(1)
    for h,w in [(960,1440),(1488,2240),(1536,2304)]:
        props_path=next(p for p in pkl if pickle.load(open(p,'rb'))['shape_before_cropping']==(1,h,w))
        props=pickle.load(open(props_path,'rb'))
        logits=np.empty((2,1,h,w),dtype=np.float32); logits[0].fill(8.0); logits[1].fill(-8.0)
        coords=[(0,0),(h-1,w-1),(h//3,w//4),(2*h//3,3*w//4)]
        for y,x in coords: logits[0,0,y,x]=-8.0; logits[1,0,y,x]=8.0
        seg_out,prob=convert_predicted_logits_to_segmentation_with_correct_shape(logits,pm,cm,lm,props,return_probabilities=True,num_threads_torch=1)
        shape_ok=tuple(prob.shape)==(2,1,h,w) and tuple(seg_out.shape)==(1,h,w)
        marker_ok=all(int(seg_out[0,y,x])==1 and float(prob[1,0,y,x])>0.99 and float(prob[0,0,y,x])<0.01 for y,x in coords)
        bg_ok=all(int(seg_out[0, min(h-1,y+1),x])==0 for y,x in coords if y+1<h and (y+1,x) not in coords)
        finite_ok=bool(np.isfinite(prob).all() and prob.dtype==np.float32)
        size_results.append({'native_hw':[h,w],'returned_probability_shape':list(prob.shape),'returned_segmentation_shape':list(seg_out.shape),'float32_finite':finite_ok,'class1_marker_orientation_channel_order':marker_ok,'background_class_argmax':bg_ok,'pass':shape_ok and marker_ok and bg_ok and finite_ok})
    # Deliberately cropped synthetic case checks background reinsertion and channel/orientation.
    h,w=64,80; props_crop=dict(props); props_crop['shape_before_cropping']=(1,h,w); props_crop['shape_after_cropping_and_before_resampling']=(1,h-4,w-6); props_crop['bbox_used_for_cropping']=[[0,1],[2,h-2],[3,w-3]]
    logits=np.empty((2,1,h-4,w-6),dtype=np.float32); logits[0].fill(8); logits[1].fill(-8)
    cy,cx=17,29; logits[0,0,cy,cx]=-8; logits[1,0,cy,cx]=8
    seg_crop,prob_crop=convert_predicted_logits_to_segmentation_with_correct_shape(logits,pm,cm,lm,props_crop,return_probabilities=True,num_threads_torch=1)
    crop_pass=tuple(seg_crop.shape)==(1,h,w) and int(seg_crop[0,cy+2,cx+3])==1 and int(seg_crop[0,0,0])==0 and int(seg_crop[0,h-1,w-1])==0 and float(prob_crop[1,0,0,0])==0.0 and float(prob_crop[1,0,cy+2,cx+3])>0.99
    size_pass=all(x['pass'] for x in size_results)
    check('official_probability_inverse_export',size_pass and crop_pass,{'three_native_sizes':size_results,'cropped_background_reinsertion':crop_pass,'test_data_used':False,'model_or_gpu_used':False})
    # Fixed artifact isolation and machine capacity; initial directory must have no prior run/checkpoint.
    preexisting=ROOT.exists()
    prior_contents=sorted(str(p.relative_to(ROOT)) for p in ROOT.iterdir()) if preexisting else []
    allowed_prior={'preflight','execution_binding.json'}
    fresh=not preexisting or set(prior_contents).issubset(allowed_prior)
    if repair_only:
        failure=json.loads((ROOT/'startup_failure.json').read_text())
        spent=json.loads((ROOT/'budget_ledger.json').read_text())
        no_checkpoints=not any(ROOT.rglob('*.pth')) and not any(ROOT.rglob('*.pt'))
        zero_update_failure=(failure.get('model_initialized') is False and failure.get('optimizer_attempts')==0
                             and failure.get('successful_updates')==0 and spent.get('optimizer_attempts')==0
                             and spent.get('successful_updates')==0 and no_checkpoints)
        check('zero_update_startup_failure_preserved',zero_update_failure,
              {'failure_path':str(ROOT/'startup_failure.json'),'spent_gpu_seconds':spent.get('cumulative_gpu_process_seconds'),
               'optimizer_attempts':spent.get('optimizer_attempts'),'no_checkpoints':no_checkpoints,'launch_authorized':False})
    else:
        check('new_scientific_directory_isolated',fresh,{'existed_before_preflight':preexisting,'prior_top_level_contents':prior_contents})
    free_gib=int(subprocess.check_output(['df','-BG',str(BASE)],text=True).splitlines()[1].split()[3].rstrip('G'))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    gpu0=gpu[0].split(','); gpu_free=len(gpu0)>=4 and int(gpu0[2].strip().split()[0])<=512 and float(gpu0[3].strip().split()[0])==0 and (not apps or 'No running' in apps)
    check('gpu0_idle_and_disk_margin',gpu_free and free_gib>=50,{'gpu0_row':gpu[0],'compute_apps':apps,'free_disk_gib':free_gib,'gpu1_unused_for_task':True})
    # No alternate train path or wrong endpoint selection in new wrapper source.
    runner=BRANCH/'nnunet2d_scientific_train.py'
    infer=BRANCH/'nnunet2d_scientific_infer_eval.py'
    supervisor=BRANCH/'nnunet2d_scientific_supervisor.py'
    runtime=BRANCH/'nnunet2d_scientific_runtime.py'
    check('explicit_train_mode_and_fixed_checkpoint',runner.is_file() and infer.is_file() and supervisor.is_file() and runtime.is_file() and '--execute-scientific-run' in runner.read_text() and 'checkpoint_final.pth' in infer.read_text() and 'checkpoint_best.pth' not in infer.read_text(),{'runner_exists':runner.is_file(),'inference_exists':infer.is_file(),'supervisor_exists':supervisor.is_file(),'runtime_exists':runtime.is_file()})
    # Fresh source binding for this scientific branch; the frozen package remains read-only.
    srcfiles=[BRANCH/n for n in ('nnunet2d_scientific_runtime.py','nnunet2d_scientific_preflight.py','nnunet2d_scientific_train.py','nnunet2d_scientific_infer_eval.py','nnunet2d_scientific_supervisor.py','nnunet2d_scientific_constructor.py','nnunet2d_scientific_restart_job.py','test_nnunet2d_restart_guards.py')]
    source_hashes={p.name:sha(p) for p in srcfiles}
    current_status=(BASE/'code/retinal-vessel-study/STATUS.md').read_text()
    authority_ok='seed2026' in current_status and '100epochs' in current_status and '唯一单种子探索已批准' in current_status and 'P1' in current_status
    binding={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_id':PROTO['protocol_id'],'protocol_sha256':ph,'source_manifest_sha256':sha(FREEZE/'source_manifest.json'),'frozen_recipe_sha256':ph,'implementation_sha256s':source_hashes,'management_binding':{'current_STATUS_sha256':sha(BASE/'code/retinal-vessel-study/STATUS.md'),'source_manifest_STATUS_sha256':MAN.get('current_management_hashes',{}).get('STATUS.md'),'current_authority_read_from_A7':authority_ok},'runtime':{'python':sys.version,'torch':torch.__version__,'cuda_build':torch.version.cuda,'nnunetv2':package_version('nnunetv2')},'data_binding':{'train_manifest_sha256':sha(BASE/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json'),'dev_reference_manifest_sha256':sha(DEV),'raw_train_export_manifest_sha256':sha(exp/'raw/train_export_manifest.csv'),'plans_sha256':plan_hash,'T0_terminal_probability_sha256s':{x['case']:x['actual'] for x in t0_results}},'execution_config':{'seed':2026,'fresh_initialization':True,'trainer':'nnUNetTrainer_100epochs','epochs':100,'iterations_per_epoch':250,'attempts':25000,'fold':'all','patch_size':[1024,1536],'batch_size':2,'optimizer':'SGD lr=.01 momentum=.99 nesterov weight_decay=.00003','scheduler':'Poly100 exponent=.9','amp':True,'clip_norm':12,'train_worker_seeds':[2026],'internal_train_monitor_worker_seeds':[2027],'workers_per_pool':1,'cudnn_deterministic':True,'cudnn_benchmark':False,'compile':False,'final_checkpoint':'checkpoint_final.pth','dev_threshold':0.5,'dev_mirroring':False,'official_test_accessed':False},'engineering_budget_reconciliation_sha256':sha(PREFLIGHT/'budget_binding_reconciliation.json')}
    binding['repair_audit_only']=repair_only
    if repair_only:
        binding['launch_authorized']=False
        binding['supersedes_implementation_for_future_review_only']=str(ROOT/'execution_binding.json')
    binding_path=(PREFLIGHT/'execution_binding.json') if repair_only else ROOT/'execution_binding.json'
    binding_path.write_text(json.dumps(binding,indent=2,ensure_ascii=False)+'\n')
    binding_hash=sha(binding_path)
    check('current_STATUS_authority',authority_ok,{'current_sha256':sha(BASE/'code/retinal-vessel-study/STATUS.md'),'source_manifest_STATUS_sha256':MAN.get('current_management_hashes',{}).get('STATUS.md'),'source_STATUS_is_pre_handoff_snapshot':sha(BASE/'code/retinal-vessel-study/STATUS.md')!=MAN.get('current_management_hashes',{}).get('STATUS.md'),'latest_STATUS_confirms_authorized_scope':authority_ok})
    result={'status':'PASS' if all(checks.values()) else 'STOP','created_utc':datetime.now(timezone.utc).isoformat(),'host':subprocess.check_output(['hostname'],text=True).strip(),'protocol_id':PROTO['protocol_id'],'protocol_sha256':ph,'execution_binding_sha256':binding_hash,'authorization':'frozen conditional single seed2026 100epochs; preflight only; no model, optimizer, training update or actual Dev/Test read performed','checks':checks,'details':details,'source_manifest_rehash':manifest_results,'T0_probability_rehash':t0_results,'budget_binding_reconciliation':reconciliation,'worker_seed_samples':{'2026_repeat_a':a,'2026_repeat_b':b,'2027':c},'cuda_initialized':bool(torch.cuda.is_initialized()),'actual_dev_pixels_or_labels_read':False,'official_test_accessed':False,'gpu_optimizer_updates':0}
    result['repair_audit_only']=repair_only
    if repair_only:
        result['launch_authorized']=False
        result['authorization']='User-authorized Sol CPU-only constructor repair and validation; no GPU restart authorization'
    EVIDENCE.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':result['status'],'failed_checks':errors,'evidence':str(EVIDENCE),'passed':sum(checks.values()),'total':len(checks),'gpu_initialized':result['cuda_initialized'],'optimizer_updates':0},ensure_ascii=False,indent=2))
    if errors: raise SystemExit(2)

if __name__=='__main__': main()
