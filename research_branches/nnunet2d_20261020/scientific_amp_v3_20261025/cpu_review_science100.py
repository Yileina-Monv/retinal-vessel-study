"""CPU admission for the new scientific entry, no retinal forward/CUDA."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent;BR=HERE.parent;sys.path.extend([str(BR/'amp_engineering32_20261025'),str(BR/'amp_recovery_revision_20261021'),str(BR/'no_wall_limit_restart_02')])
import json,subprocess,importlib.util,importlib.metadata as md,pickle,ast,copy
import numpy as np,torch
torch.set_num_threads(1)
from science100_gate import STAGE,CODE,PHASE,PROTOCOL_SHA,protocol,sha,canonical
from fast_observer import atomic
q=protocol();ev=STAGE/'cpu_preparation';ev.mkdir(exist_ok=False);base=STAGE.parent
os.environ.update(nnUNet_raw=str(base/'raw'),nnUNet_preprocessed=str(base/'preprocessed'),nnUNet_results=str(ev/'constructor_only'),nnUNet_compile='false',nnUNet_n_proc_DA='1')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(HERE),str(BR/'amp_engineering32_20261025'),str(BR/'amp_recovery_revision_20261021'),str(BR/'no_wall_limit_restart_02'),str(CODE)]))
t=subprocess.run([sys.executable,'-u','-B',str(HERE/'test_fast_observer.py')],env=env,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
(ev/'fast_observer_tests.log').write_text(t.stdout)
if t.returncode:raise RuntimeError('fast observer CPU tests failed; see saved log')
assert 'Ran 10 tests' in t.stdout and '\nOK' in t.stdout;print('FAST_OBSERVER10_TESTS_PASSED',flush=True)
from nnunet2d_scientific_constructor import build_fresh_trainer
from amp32_runner import bind_train_datasets
pre=base/'preprocessed/Dataset501_MAPLES';plans=json.loads((pre/'nnUNetPlans_8GBtarget.json').read_text());dataset=json.loads((pre/'dataset.json').read_text());assert sha(pre/'nnUNetPlans_8GBtarget.json')==q['recipe']['plans_sha256']
trainer=build_fresh_trainer(plans,dataset,torch.device('cpu'));tr,va=bind_train_datasets(trainer,q['train_ids']);assert len(tr.identifiers)==110 and trainer.network is None and trainer.optimizer is None
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
pm=PlansManager(plans);cm=pm.get_configuration('2d');lm=pm.get_label_manager(dataset);assert list(cm.spacing)==[1.,1.] and plans['transpose_forward']==plans['transpose_backward']==[0,1,2]
props_by_shape={}
for p in (pre/'nnUNetPlans_8GBtarget_2d').glob('*.pkl'):
 props=pickle.loads(p.read_bytes());props_by_shape.setdefault(tuple(props['shape_before_cropping']),props)
export_checks=[]
for h,w in [(960,1440),(1488,2240),(1536,2304)]:
 props=props_by_shape[(1,h,w)];logits=np.empty((2,1,h,w),dtype=np.float32);logits[0].fill(8);logits[1].fill(-8);coords=[(0,0),(h-1,w-1),(h//3,w//4),(2*h//3,3*w//4)]
 for y,x in coords:logits[0,0,y,x]=-8;logits[1,0,y,x]=8
 seg,prob=convert_predicted_logits_to_segmentation_with_correct_shape(logits,pm,cm,lm,props,return_probabilities=True,num_threads_torch=1)
 assert prob.shape==(2,1,h,w) and seg.shape==(1,h,w) and prob.dtype==np.float32 and np.isfinite(prob).all()
 assert all(seg[0,y,x]==1 and prob[1,0,y,x]>.99 and prob[0,0,y,x]<.01 for y,x in coords)
 export_checks.append({'shape':[h,w],'marker_channel_orientation_pass':True})
h,w=64,80;pc=copy.deepcopy(props);pc.update(shape_before_cropping=(1,h,w),shape_after_cropping_and_before_resampling=(1,h-4,w-6),bbox_used_for_cropping=[[0,1],[2,h-2],[3,w-3]])
lg=np.empty((2,1,h-4,w-6),dtype=np.float32);lg[0].fill(8);lg[1].fill(-8);lg[0,0,17,29]=-8;lg[1,0,17,29]=8
sg,pr=convert_predicted_logits_to_segmentation_with_correct_shape(lg,pm,cm,lm,pc,return_probabilities=True,num_threads_torch=1);assert sg.shape==(1,h,w) and sg[0,19,32]==1 and pr[1,0,0,0]==0
dev=json.loads(Path(q['references']['Dev_RGB_manifest']).read_text())['images']
for r in dev:
 shape=[max(n,p) for n,p in zip(r['native_shape'],[1024,1536])];steps=compute_steps_for_sliding_window(tuple(shape),(1024,1536),.5);count=int(np.prod([len(x) for x in steps]));assert count==q['inference']['expected_tiles_by_image'][r['filename']]
from science100_infer import prevent_native_retry
class Fake:
 def __init__(self):self.calls=0
 def _internal_predict_sliding_window_return_logits(self):self.calls+=1;raise RuntimeError('synthetic OOM')
fake=Fake();state={'running':True};prevent_native_retry(fake,lambda e:state.update(running=False),lambda:state['running'])
for i in range(2):
 try:fake._internal_predict_sliding_window_return_logits()
 except RuntimeError:pass
assert fake.calls==1
atomic(ev/'inverse_export_review.json',{'status':'PASS','three_sizes':export_checks,'cropped_background_reinsertion':True,'tile_forward_cap':sum(q['inference']['expected_tiles_by_image'].values()),'native_retry_blocked_before_model_call':True,'GPU_calls':0})
print('CPU_CONSTRUCTOR_GEOMETRY_AND_NO_RETRY_PASSED',flush=True)
from science100_analysis import analyze
fixture=analyze(ev/'A0_endpoint_fixture',fixture=True);assert fixture['status']=='CPU_FIXTURE_ONLY' and len(fixture['per_image'])==28 and len(fixture['fixed_ids_by_metric']['near_thin_recall_d1'])==9
assert all(v['NNUNET-A0']['estimate']==0 for v in fixture['metrics'].values());print('A0_FULL_ENDPOINT_FIXTURE31_METRICS_PASSED',flush=True)
engineer=json.loads((base/'amp_engineering32_20261025/cpu_binding.json').read_text());engreview=Path(q['references']['accepted_engineering_review']);assert sha(engreview)==q['references']['accepted_engineering_review_sha256']
for group in ['code_sources','data_sources']:
 for p,h in engineer[group].items():assert sha(p)==h,p
allcode=dict(engineer['code_sources']);allcode.update({str(p):sha(p) for p in HERE.glob('*.py')});runtime={str(p):sha(p) for p in HERE.glob('*.py')}
for name in ['amp_observer_v2.py']:
 p=BR/'amp_engineering32_20261025'/name;runtime[str(p)]=sha(p)
for folder,names in [(BR/'amp_recovery_revision_20261021',['revised_policy.py','revised_runtime.py']),(BR/'no_wall_limit_restart_02',['nnunet2d_scientific_constructor.py','nnunet2d_scientific_runtime.py']),(CODE/'retinal_m0',['maples_branch_science_analysis.py','maples_branch_science_reference.py','maples_stratified_science_analysis.py'])]:
 for n in names:p=folder/n;allcode[str(p)]=sha(p);runtime[str(p)]=sha(p)
for p in [STAGE/'protocol.json',STAGE/'SCIENTIFIC_PROTOCOL.md']:allcode[str(p)]=sha(p);runtime[str(p)]=sha(p)
refs=q['references'];references={}
for name in ['accepted_A0_fixture','science_reference_manifest','A0_reuse_binding','Dev_RGB_manifest','Train_manifest','accepted_engineering_review','parent_protocol']:
 p=Path(refs[name]);references[str(p)]=sha(p)
for r in json.loads(Path(refs['science_reference_manifest']).read_text())['images']:
 assert sha(r['reference_npz'])==r['reference_npz_sha256'];references[r['reference_npz']]=r['reference_npz_sha256']
for r in json.loads(Path(refs['A0_reuse_binding']).read_text())['probability_maps']:
 assert sha(r['path'])==r['sha256'];references[r['path']]=r['sha256']
allcode.update(references)
versions={n:md.version(n) for n in ['torch','nnunetv2','batchgenerators','batchgeneratorsv2']};assert versions==engineer['versions']
binding={'phase':PHASE,'protocol_sha256':PROTOCOL_SHA,'all_code_sources':allcode,'runtime_code_sources':runtime,'Train_sources':engineer['data_sources'],'reference_sources':references,'versions':versions,'CPU_tests':10,'no_extra_GPU_engineering':True};key=canonical(binding);binding['binding_sha256']=key;atomic(STAGE/'source_binding.json',binding)
denials=[]
for script,argv in [('science100_controller.py',['--run-authorized']),('science100_controller.py',['--run-authorized','--epochs','200']),('science100_train.py',['--authorized-train','--parent-pid','1','--nonce','wrong']),('science100_infer.py',['--authorized-final-inference','--parent-pid','1','--nonce','wrong'])]:
 p=subprocess.run([sys.executable,'-B',str(HERE/script),*argv],env=env,cwd=CODE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90);assert p.returncode!=0;denials.append({'script':script,'argv':argv,'returncode':p.returncode,'output':p.stdout[-3000:]})
atomic(ev/'CLI_refusals.json',denials);assert not torch.cuda.is_initialized() and not (STAGE/'TRAIN_SESSION.json').exists()
atomic(STAGE/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','binding_sha256':key,'tests':10,'actual_official_loop_step_CPU_equivalence':True,'fixed_input_cross_process_mid_epoch_partial_outputs_and_state_exact':True,'native_queue_resume_supported':False,'actual_constructor_Train110_sets':True,'native_export_three_sizes_and_background_pass':True,'all31_A0_fixture_metrics_exact':True,'fixed_Dev28_old_near9_77_19NA_retained':True,'inference_retry_denied':True,'CLI_denials':4,'no_new_GPU_calls':True,'no_retinal_model_initialized':True,'no_Dev_RGB_pixels_read':True,'Test_accessed':False,'tests_log_sha256':sha(ev/'fast_observer_tests.log'),'fixture_sha256':sha(ev/'A0_endpoint_fixture/endpoint_100epochs.json'),'inverse_export_review_sha256':sha(ev/'inverse_export_review.json')})
print('SCIENCE100_CPU_REVIEW_COMPLETE',key,flush=True)
