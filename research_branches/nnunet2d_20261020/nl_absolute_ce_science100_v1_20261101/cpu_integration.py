import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',nnUNet_compile='false',nnUNet_n_proc_DA='1')
from common import *
import torch,numpy as np,random,time,traceback
from nnunet2d_scientific_constructor import build_fresh_trainer
from seeded_runtime import install_seeded_worker_adapter
from sidecar import install
from selection import choose_groups
from fast_observer import digest

def main():
 out=GROUP/'cpu_integration';out.mkdir(exist_ok=False);start=time.monotonic();checks=0
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 def forbidden(*a,**k):raise RuntimeError('CPU integration forbids CUDA/model/optimizer')
 torch.cuda.init=forbidden;torch.cuda.set_device=forbidden;torch.nn.Module.cuda=forbidden
 from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
 nnUNetTrainer.initialize=forbidden;nnUNetTrainer.run_training=forbidden
 expected={}
 for line in (P/'results/selection_trace.jsonl').open():
  x=json.loads(line)
  if x['attempt']<=32:expected[(x['seed'],x['attempt'],x['crop'])]=x['selected']
 install()
 results=[]
 for seed in [2027,2028]:
  os.environ.update(nnUNet_raw=str(DATA_BASE/'raw'),nnUNet_preprocessed=str(DATA_BASE/'preprocessed'),nnUNet_results=str(out/f'native_{seed}'),MPLCONFIGDIR=str(out/'matplotlib'))
  # Each seed pool adapter must wrap original class/producer, not previous seed's adapter.
  import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as tm
  import batchgenerators.dataloading.nondet_multi_threaded_augmenter as am
  original_aug,original_producer=tm.NonDetMultiThreadedAugmenter,am.producer
  random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
  t=build_fresh_trainer(read(DATA_BASE/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'),read(DATA_BASE/'preprocessed/Dataset501_MAPLES/dataset.json'),torch.device('cpu'))
  t.batch_size=2;pools=install_seeded_worker_adapter(seed)
  tr=va=None
  try:
   tr,va=t.get_dataloaders()
   history=(BASELINE/'runs'/f'seed_{seed}'/'attempts.jsonl').open();nonempty=0
   for i in range(1,33):
    b=next(tr);h=json.loads(next(history));compare_train(b,h,i,0,.01,digest);checks+=1
    if len(b['nl_candidates'])!=2:raise RuntimeError('batch sidecars missing')
    groups=[]
    for crop,meta in enumerate(b['nl_candidates']):
     g=choose_groups(meta['candidates'],seed,i,crop)
     if g!=expected[(seed,i,crop)]:raise RuntimeError('CPU V2 selected target mismatch '+str((seed,i,crop)))
     checks+=1;groups.append(g)
    nonempty+=bool(groups[0] or groups[1])
    log(out/'batches.jsonl',{'seed':seed,'attempt':i,'IDs':list(b['keys']),'input_sha256':digest(b['data']),'target_sha256':digest(b['target']),'groups_per_crop':[len(g) for g in groups],'all_target_metadata_matched':True})
    if i%8==0:print('CPU_NATIVE',seed,i,flush=True)
   monitor=next(va);mh=json.loads((BASELINE/'runs'/f'seed_{seed}'/'monitor.jsonl').open().readline())
   assert list(monitor['keys'])==mh['ids'] and 'nl_candidates' not in monitor;checks+=1
   assert pools==[{'pool':'train','worker_count':1,'seeds':[seed]},{'pool':'internal_train_monitor','worker_count':1,'seeds':[seed+1]}];checks+=1
   results.append({'seed':seed,'matched_training_batches':32,'nonempty_batches':nonempty,'monitor_first_IDs_matched':True,'native_prefetch_discard_preserved':True})
  finally:
   for loader in [tr,va]:
    if hasattr(loader,'_finish'):loader._finish()
   tm.NonDetMultiThreadedAugmenter=original_aug;am.producer=original_producer
 assert not torch.cuda.is_initialized()
 atomic(out/'RESULT.json',{'status':'PASS','checks':checks,'seeds':results,'elapsed_seconds':time.monotonic()-start,'GPU_calls':0,'model_forwards':0,'parameter_updates':0,'Dev_Test_read':False},True)
 print('CPU_NATIVE_PASS',checks,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  atomic(GROUP/'CPU_INTEGRATION_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'GPU_used':False},True);raise
