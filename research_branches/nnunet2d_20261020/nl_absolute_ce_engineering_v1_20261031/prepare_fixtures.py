import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,torch
R=Path('${RETINA_PROJECT_ROOT}');B=R/'code/retinal-vessel-study/research_branches/nnunet2d_20261020'
CPU=B/'nl_absolute_ce_v2_alltargets_20261031';sys.path.insert(0,str(CPU))
import replay
O=replay.STAGE/'engineering_v1';F=O/'fixtures'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not F.exists();F.mkdir()
 s=json.loads((replay.OUT/'cpu_summary.json').read_text());assert s['status']=='PASS'
 records={}
 for line in (replay.OUT/'selection_trace.jsonl').open():
  a=json.loads(line);records[(a['seed'],a['attempt'],a['crop'])]=a['selected']
 valid=[i for i in range(1,1001) if records[(2027,i,0)] or records[(2027,i,1)]][:32];assert len(valid)==32
 chosen={2027:sorted(set(range(1,33))|set(valid)),2028:list(range(1,17))}
 rows=[];torch.set_num_threads(1);start=time.monotonic()
 for seed,numbers in chosen.items():
  loader,unused,active=replay.build_loader(seed);assert active[0]==0
  next(loader)
  h=(replay.HISTORY/f'seed_{seed}'/'attempts.jsonl').open()
  for i in range(1,max(numbers)+1):
   expected=json.loads(next(h));batch=next(loader)
   assert list(batch['keys'])==expected['ids'] and replay.digest(batch['data'])==expected['input_sha256'] and replay.digest(batch['target'])==expected['target_sha256']
   if i in numbers:
    p=F/f'{seed}_{i:04d}.pt';groups=[records[(seed,i,j)] for j in [0,1]]
    torch.save({'data':batch['data'],'target':batch['target'],'keys':batch['keys'],'groups':groups},p)
    rows.append({'seed':seed,'attempt':i,'path':str(p),'sha256':sha(p),'input_sha256':expected['input_sha256'],'target_sha256':expected['target_sha256'],'ids':expected['ids'],'group_counts':[len(x) for x in groups]})
   if i%20==0:print('FIXTURES',seed,i,flush=True)
 result={'status':'PASS','calibration_attempts':valid,'rows':rows,'native_attempts_generated':{str(s):max(ns) for s,ns in chosen.items()},'fixture_bytes':sum(p.stat().st_size for p in F.iterdir()),'seconds':time.monotonic()-start,'model_forwards':0,'GPU_used':False}
 with (O/'CPU_FIXTURES.json').open('x') as f:json.dump(result,f,indent=2)
 print('FIXTURES_READY',len(rows),result['fixture_bytes'],flush=True)
if __name__=='__main__':main()
