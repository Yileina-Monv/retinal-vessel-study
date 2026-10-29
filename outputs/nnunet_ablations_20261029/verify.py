"""Recompute published image-level statistics; Python standard library only."""
from pathlib import Path
import json,math,hashlib
B=Path(__file__).resolve().parent;R=B.parent.parent
e=json.loads((B/'fixed_endpoint_evidence.json').read_text(encoding='utf-8'));idx=json.loads((B/'bootstrap_indices_by_n.json').read_text(encoding='utf-8'));checks=0;maximum=0.
def close(a,b):
 global checks,maximum
 error=abs(a-b);maximum=max(maximum,error);assert error<1e-12,(a,b,error);checks+=1
def avg(v):return sum(v)/len(v)
def quantile(v,p):
 v=sorted(v);i=(len(v)-1)*p;a=int(i);return v[a]+(v[min(a+1,len(v)-1)]-v[a])*(i-a)
for name,d in e.items():
 by={x['filename']:x for x in d['per_image']};assert len(by)==28 and len(d['metrics'])==31
 for m in d['absolute_means']:
  near=[x['near_counts'][m] for x in by.values()];assert sum(x['denominator'] for x in near)==77 and sum(x['value'] is None for x in near)==19
  for k,ids in d['fixed_ids_by_metric'].items():
   values=[by[n]['methods'][m][k] for n in ids];assert all(v is not None and math.isfinite(v) for v in values);close(avg(values),d['absolute_means'][m][k])
 for k,stats in d['metrics'].items():
  ids=d['fixed_ids_by_metric'][k]
  for contrast,s in stats.items():
   candidates=[(a,b) for a in d['absolute_means'] for b in d['absolute_means'] if a+'-'+b==contrast];assert len(candidates)==1
   a,b=candidates[0];delta=[by[n]['methods'][a][k]-by[n]['methods'][b][k] for n in ids]
   boots=[avg([delta[i] for i in row]) for row in idx[str(len(ids))]]
   for a,b in [(avg(delta),s['estimate']),(quantile(boots,.025),s['lower']),(quantile(boots,.975),s['upper'])]:close(a,b)
for study in ['modules2027','deep_supervision2027']:
 for k,v in e[study]['absolute_means']['N0'].items():close(v,e['standard2027']['absolute_means']['NNUNET'][k])
d=e['deep_supervision2027'];delta=d['metrics']['dice']['N-FinalOnly-N0'];assert delta['upper']<0
assert sum(x['near_counts']['N0']['numerator'] for x in d['per_image'])==59
assert sum(x['near_counts']['N-FinalOnly']['numerator'] for x in d['per_image'])==57
for path,h in json.loads((B/'file_manifest.json').read_text(encoding='utf-8')).items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h,path
print(json.dumps({'status':'PASS','numeric_comparisons':checks,'max_error':maximum,'studies':len(e),'scope':'image-table/bootstrap verification; no training, inference or Test access'}))
