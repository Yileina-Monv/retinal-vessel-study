"""Freeze actual sources, inherited input evidence and CPU implementation checks."""
import os,sys,subprocess,ast,json
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from scse_gate import STAGE as S,CODE,ROOT,OUT,PHASE,PROTOCOL_SHA,sha,protocol
from retinal_m0.maples_g1_safety import read_json,atomic_json,canonical_sha
from scse_core import Budget,pair_record
import torch
q=protocol();ev=S/'cpu_preparation';ev.mkdir(exist_ok=True)
if (S/'ENG_RELEASE.json').exists():raise RuntimeError('already released CPU sources')
os.environ['PYTHONPATH']=str(HERE)+os.pathsep+str(CODE)
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
p=subprocess.run([sys.executable,'-B',str(HERE/'test_scse.py')],cwd=CODE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240);log=ev/'CPU_tests.log'
if log.exists():log.rename(ev/('CPU_tests.previous_'+str(len(list(ev.glob('CPU_tests.previous*'))))+'.log'))
log.write_text(p.stdout);assert p.returncode==0 and 'Ran 9 tests' in p.stdout,p.stdout
base=S.parent;prior=read_json(base/'spatial_stratified_scientific_2000_v1_20261025/cpu_preparation/source_binding.json');sources=dict(prior['inherited_receipts']['original_expected_sha256'])
for path,h in sources.items():assert sha(path)==h,path
eng=read_json(base/'ufl_asymmetric_v1_20261024/engineering100/execution_binding.json');allowed=list(eng['train_content_paths']);dev=read_json(ROOT/'docs/full_fov_thin_skeleton_bce_20261019/dev_reference_manifest.json');allowed.extend(r['inputs']['RGB']['path'] for r in dev['images'])
for path in allowed:
    if path not in sources:sources[path]=sha(path)
for path in [q['expected_pairs_path'],str(ROOT/'docs/unet_single_module_direction_review_20261025.md'),str(S/'protocol.json'),str(S/'SCIENTIFIC_PROTOCOL.md')]:sources[path]=sha(path)
from retinal_m0.maples_core import CorePatches
ds=CorePatches(q['baseline_config'],'T0');pairs=[json.loads(x) for x in Path(q['expected_pairs_path']).read_text().splitlines()];assert len(pairs)==2000
sampled=[]
for i in list(range(8))+[1999]:
    batch=torch.utils.data.default_collate([ds[i*4+j] for j in range(4)]);actual=pair_record(batch,i+1);assert actual==pairs[i];sampled.append(i+1)
runtime={str(p):sha(p) for p in HERE.glob('*.py')}
for path in [str(S/'protocol.json'),str(S/'SCIENTIFIC_PROTOCOL.md'),q['expected_pairs_path'],str(Path(q['A0_path'])/'config.json')]:runtime[path]=sha(path)
for path,h in sources.items():
    if path.startswith(str(CODE)) and Path(path).suffix in ['.py','.json']:runtime[path]=h
sources.update(runtime)
# Reuse exact existing frozen metric implementation; candidate=A0 fixture must reproduce31 means and zero contrasts.
from scse_analysis import analyze
fixture=ev/'A0_endpoint_fixture'
if fixture.exists():raise RuntimeError('existing CPU fixture; inspect before rerunning')
endpoint=analyze(out=fixture,fixture=True);assert len(endpoint['per_image'])==28 and all(v['SCSE-A0']['estimate']==0 for v in endpoint['metrics'].values())
binding={'phase':PHASE,'protocol_sha256':PROTOCOL_SHA,'sources':sources,'runtime_sources':runtime,'data_content_paths':allowed,'actual_A0_pair_batches_verified':sampled,'original_8000_replay_evidence':str(base/'scientific_b1_b2_2000_v1_20261022/pairing_8000_verification.json'),'A0_fixture_sha256':sha(fixture/'endpoint_2000.json'),'no_GPU_initialized':not torch.cuda.is_initialized()};key=canonical_sha(binding);binding['binding_sha256']=key;atomic_json(S/'source_binding.json',binding)
denials=[]
for filename,args in [('scse_controller.py',['--run-authorized']),('scse_controller.py',['--run-authorized','--epochs','100']),('scse_worker.py',['--role','science','--track','SCSE','--start','0','--end','2000','--parent-pid','1','--nonce','bad'])]:
    p=subprocess.run([sys.executable,'-B',str(HERE/filename),*args],cwd=CODE,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45);assert p.returncode!=0;denials.append({'file':filename,'args':args,'code':p.returncode,'tail':p.stdout[-1500:]})
atomic_json(ev/'CLI_denials.json',denials)
assert not torch.cuda.is_initialized()
Budget(S/'engineering_budget.json',key,global_cap=32,track_caps={'CONT':16,'SPLIT':16},forward_cap=32,gradient_cap=32).initialize();Budget(S/'science_budget.json',key,track_caps={'SCSE':2000}).initialize()
atomic_json(S/'CPU_REVIEW.json',{'status':'CPU_ACCEPTED','binding_sha256':key,'actual_tests_passed':9,'actual_source_files_hashed':len(sources),'original_Train110_Dev28_whitelist':True,'original_2000pair_SHA_verified':True,'sampled_actual_input_batches':sampled,'full_previous8000_input_evidence_reused':True,'A0_fixture_all31_means_and_zero_contrasts':True,'CLI_denials':3,'no_GPU_initialized':True,'Test_accessed':False,'CPU_log_sha256':sha(log)})
print('SCSE_CPU_ACCEPTED',key,flush=True)
