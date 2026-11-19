"""Actual per-seed A0 provenance/metrics, class-cleanup CPU regression and bounded admission."""
import os,sys,json,subprocess,ast,hashlib,copy
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',nnUNet_n_proc_DA='1')
from pathlib import Path
import replication_gate as g
from retinal_m0.maples_g1_safety import atomic_json,read_json,canonical_sha
from fast_observer import Journal,AmpLimits
import numpy as np,torch
from retinal_m0.state import state_digest
from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
from retinal_m0.maples_stratified_science_analysis import hard_cldice
S=g.GROUP;B=g.BRANCH
if (S/'CPU_REVIEW.json').exists():raise RuntimeError('CPU phase already frozen')
ev=S/'cpu_preparation';ev.mkdir(exist_ok=True)
os.environ['PYTHONPATH']=os.pathsep.join([str(B),str(g.CODE),str(g.OLD),str(g.ENG),str(g.OBS),str(g.BRANCH.parent/'amp_recovery_revision_20261021')])
for p in B.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
tests=[(B/'test_replication.py',8),(g.OBS/'test_fast_observer.py',10)]
for p,count in tests:
    result=subprocess.run([sys.executable,'-B',str(p)],env=os.environ,cwd=g.CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240);log=ev/(p.stem+'.log')
    if log.exists():log.rename(log.with_name(log.name+'.previous_'+str(len(list(ev.glob(log.name+'.previous*'))))))
    log.write_text(result.stdout);assert result.returncode==0 and ('Ran '+str(count)+' tests') in result.stdout,result.stdout
original=read_json(g.group_protocol()['old_source_binding'])
for kind in ['all_code_sources','Train_sources']:
    for path,h in original[kind].items():assert g.sha(path)==h,path
all_sources=dict(original['all_code_sources']);runtime={p:h for p,h in original['runtime_code_sources'].items() if 'scientific_' not in Path(p).parent.name}
for p in [g.OBS/'fast_observer.py',g.OBS/'test_fast_observer.py']:all_sources[str(p)]=g.sha(p);runtime[str(p)]=g.sha(p)
valid=read_json(g.group_protocol()['original_A0_GT_validity_fixture'])['fixed_ids_by_metric'];baseline_receipts={}
for seed in [2027,2028]:
    g.set_context(seed);q=g.protocol();a0=g.ROOT/f'experiments/t0_tw_core_20261016/runs/seed_{seed}/T0';cfg=read_json(a0/'config.json');summary=read_json(a0/'summary.json');binding=read_json(a0/'run_binding.json');metrics_path=a0/'evaluation/step_002000/metrics.json';metrics=read_json(metrics_path)
    assert cfg['seed']==seed and summary['seed']==seed and summary['status']=='passed' and summary['completed_steps']==2000 and summary['next_global_draw']==8000
    assert g.sha(a0/'config.json')==binding['config_sha256'] and summary['run_binding']==binding==metrics['run_binding']
    assert g.sha(a0/'last.pt')==summary['checkpoint_sha256'] and g.sha(a0/'step_002000.pt')==metrics['checkpoint_sha256']
    last=torch.load(a0/'last.pt',map_location='cpu',weights_only=True);mile=torch.load(a0/'step_002000.pt',map_location='cpu',weights_only=True)
    for k in ['model','optimizer','scaler','rng','step','next_global_draw']:assert state_digest(last[k])==state_digest(mile[k]),k
    assert mile['step']==2000 and mile['next_global_draw']==8000 and mile['binding']==binding
    del last,mile
    refs=read_json(q['references']['science_reference_manifest'])['images'];old={r['filename']:r for r in metrics['images']};prob_dir=a0/'evaluation/step_002000/probabilities';assert len(list(prob_dir.glob('*.npy')))==28
    scalar={};probrows=[]
    for ref in refs:
        name=ref['filename'];p=prob_dir/(name+'.npy');prob=np.load(p,allow_pickle=False);assert prob.dtype==np.float32 and list(prob.shape)==ref['native_shape'] and np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
        assert g.sha(ref['reference_npz'])==ref['reference_npz_sha256']
        with np.load(ref['reference_npz'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        rr=evaluate_image(prob,arrays,'B2',threshold=.5);h=hard_cldice((prob>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV']);rr['hard_cldice']=h['cldice'];vals=metric_values(rr);vals['hard_cldice']=h['cldice'];scalar[name]={m:vals[m] for m in valid}
        for metric in ['dice','precision','recall','fp_density','near_pixel_recall','near_thin_recall_d1','far_recall']:
            a=vals[metric];b=old[name][metric]['value'];assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-12),(seed,name,metric,a,b)
        probrows.append({'filename':name,'path':str(p),'sha256':g.sha(p)})
    means={m:float(np.mean([scalar[name][m] for name in names])) for m,names in valid.items()}
    fixture={'status':'CPU_VERIFIED_EXISTING_A0_BASELINE','seed':seed,'endpoint_step':2000,'threshold':.5,'fixed_ids_by_metric':valid,'absolute_means':{'A0':means},'per_image_scalar':scalar,'source_metrics_sha256':g.sha(metrics_path),'checkpoint_sha256':metrics['checkpoint_sha256'],'same_GT_validity_definition_as_2026':True}
    fixturepath=Path(q['references']['accepted_A0_fixture']);reusepath=Path(q['references']['A0_reuse_binding']);atomic_json(fixturepath,fixture);atomic_json(reusepath,{'seed':seed,'completed_steps':2000,'probability_maps':probrows,'checkpoint_sha256':metrics['checkpoint_sha256'],'config_sha256':g.sha(a0/'config.json'),'summary_sha256':g.sha(a0/'summary.json'),'old_last_milestone_available_states_identical':True,'new_inference_calls':0})
    for p in [a0/'config.json',a0/'summary.json',a0/'run_binding.json',a0/'last.pt',a0/'step_002000.pt',metrics_path,fixturepath,reusepath]:all_sources[str(p)]=g.sha(p)
    for row in probrows:all_sources[row['path']]=row['sha256']
    runtime[str(fixturepath)]=g.sha(fixturepath);runtime[str(reusepath)]=g.sha(reusepath)
    command=[sys.executable,'-B',str(B/'analysis_runner.py'),'--seed',str(seed),'--fixture'];result=subprocess.run(command,env=os.environ,cwd=g.CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300);(ev/('fixture_seed'+str(seed)+'.log')).write_text(result.stdout);assert result.returncode==0,result.stdout
    checked=read_json(g.STAGE/'cpu_reference/full_endpoint_fixture/endpoint_100epochs.json');assert checked['status']=='CPU_FIXTURE_ONLY' and len(checked['per_image'])==28 and all(x['NNUNET-A0']['estimate']==0 for x in checked['metrics'].values())
    baseline_receipts[str(seed)]={'raw_28_probability_files_verified':True,'last_and_milestone_states_equal':True,'original7metrics_all28_images_recomputed_exact':True,'full31metric_fixture_passed':True,'A0_Dice':means['dice'],'A0_checkpoint_sha256':metrics['checkpoint_sha256'],'full_fixture_sha256':g.sha(g.STAGE/'cpu_reference/full_endpoint_fixture/endpoint_100epochs.json')}
for p in B.glob('*.py'):all_sources[str(p)]=g.sha(p);runtime[str(p)]=g.sha(p)
for p in [S/'protocol.json',S/'SCIENTIFIC_PROTOCOL.md',S/'runs/seed_2027/protocol.json',S/'runs/seed_2028/protocol.json']:all_sources[str(p)]=g.sha(p);runtime[str(p)]=g.sha(p)
b={'phase':g.PHASE,'group_protocol_sha256':g.GROUP_PROTOCOL_SHA,'all_sources':all_sources,'runtime_sources':runtime,'Train_sources':original['Train_sources'],'versions':original['versions'],'baseline_receipts':baseline_receipts,'official_loop_fast_observer_GPU32_evidence_reused':True,'no_new_GPU_engineering':True};key=canonical_sha(b);b['binding_sha256']=key;atomic_json(S/'source_binding.json',b)
denials=[]
for seed in [2027,2028]:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',RETINA_NN_SEED=str(seed))
    for filename,args in [('replication_controller.py',['--run-authorized-serial']),('replication_train.py',['--authorized-train','--parent-pid','1','--nonce','bad']),('replication_infer.py',['--authorized-final-inference','--parent-pid','1','--nonce','bad'])]:
        p=subprocess.run([sys.executable,'-B',str(B/filename),*args],env=env,cwd=g.CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60);assert p.returncode!=0;denials.append({'seed':seed,'script':filename,'exit_code':p.returncode,'tail':p.stdout[-1200:]})
    g.set_context(seed);Journal(g.STAGE/'budget.json',g.binding_key(seed),100,250,50,AmpLimits(5,25)).initialize()
atomic_json(ev/'CLI_denials.json',denials);assert not torch.cuda.is_initialized()
atomic_json(S/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','binding_sha256':key,'new_seed_cleanup_tests_passed':8,'reused_official_fast_observer_tests_passed':10,'original_Train_sources_rehashed':len(original['Train_sources']),'baseline_receipts':baseline_receipts,'actual_CLI_denials':6,'no_GPU_initialized':True,'Test_accessed':False,'no_new_GPU_engineering':True})
print('REPLICATION_CPU_ACCEPTED',key,flush=True)
