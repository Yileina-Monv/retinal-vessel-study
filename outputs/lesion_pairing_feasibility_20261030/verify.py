"""Recompute public tables only; no medical images, model, GPU or Test access."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, math, statistics
B=Path(__file__).resolve().parent
ROOT=B.parents[1]
N=0
MAX_ERROR=0.0
def check(ok, label):
 global N
 N+=1
 if not ok: raise AssertionError(label)
def close(a,b,label):
 global MAX_ERROR
 error=abs(float(a)-float(b));MAX_ERROR=max(MAX_ERROR,error)
 check(error<=1e-12,label)
def J(name):return json.loads((B/name).read_text(encoding='utf-8'))
def rows(name):return list(csv.DictReader((B/name).open(encoding='utf-8',newline='')))
manifest=J('file_manifest.json')
for name,h in manifest['sha256'].items():
 check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'file SHA '+name)
reviews=J('review_summary.json');ds=J('d0_summary.json');budget=J('budget_and_decision.json')
points=rows('d0_points_154.csv');images=rows('d0_image_regions_56.csv');index=rows('d0_point_index_77.csv')
check(len(points)==154 and len(images)==56 and len(index)==77,'D0 row counts')
check(len({p['point_id'] for p in index})==77,'unique point identities')
misses={};qualifying={}
for seed in ['2027','2028']:
 pp=[p for p in points if p['seed']==seed];ii=[r for r in images if r['seed']==seed];summary=ds['seed_statistics'][seed]
 check(len(pp)==77 and len(ii)==28,'complete seed '+seed)
 check(len({p['point_id'] for p in pp})==77,'point uniqueness '+seed)
 check(len({r['filename'] for r in ii})==28,'image uniqueness '+seed)
 check({p['point_id']:p['filename'] for p in pp}=={r['point_id']:r['filename'] for r in index},'77 index correspondence')
 misses[seed]={p['point_id'] for p in pp if not int(p['d1_hit'])}
 qualifying[seed]={p['point_id'] for p in pp if int(p['qualifying_miss'])}
 check(len(misses[seed])==summary['misses']==18,'18 misses '+seed)
 check(sum(int(p['d1_hit']) for p in pp)==59,'59 hits '+seed)
 check(len(qualifying[seed])==summary['qualifying_misses'],'qualifying count '+seed)
 check(sorted({p['filename'] for p in pp if p['point_id'] in qualifying[seed]})==summary['qualifying_images'],'qualifying image identities')
 check(len(summary['qualifying_images'])==summary['qualifying_image_count']==5,'five qualifying images')
 close(len(qualifying[seed])/18,summary['qualifying_fraction_of_misses'],'qualifying fraction')
 for p in pp:
  prob=float(p['center_probability']);clipped=min(1-1e-6,max(1e-6,prob));logit=math.log(clipped/(1-clipped))
  close(logit,p['target_logit'],'target logit')
  close(float(p['target_logit'])-float(p['background_mean_logit']),p['logit_gap'],'gap formula')
  check(int(p['d1_hit'])==int(float(p['d1_max_probability'])>=0.5),'fixed threshold d1')
  expected=int(not int(p['d1_hit']) and int(p['background_count'])>0 and float(p['logit_gap'])<1)
  check(expected==int(p['qualifying_miss']),'qualifying predicate')
  check(p['medical_review']=='UNKNOWN_NOT_REVIEWED','medical review remains unknown')
 for r in ii:
  rr=[p for p in pp if p['filename']==r['filename']];den=int(r['near_target_denominator'])
  check(den==len(rr),'per-image point denominator')
  check(sum(int(p['d1_hit']) for p in rr)==int(r['near_hits']),'per-image hits')
  check(sum(int(r[k+'_area']) for k in ['internal','near','far','uncertain'])==int(r['fov_pixels']),'region areas sum')
  check(sum(int(r[k+'_FP']) for k in ['internal','near','far','uncertain'])==int(r['full_FP']),'region FP sum')
  if den:close(int(r['near_hits'])/den,r['near_recall_d1'],'per-image d1 recall')
  else:check(r['near_recall_d1']=='' and r['near_NA_reason']=='zero_denominator','NA preserved')
  for k in ['internal','near','far','uncertain']:
   for denominator,suffix in [('area','per_area'),('background_pixels','per_background')]:
    n=int(r[k+'_'+denominator]);fp=int(r[k+'_FP']);value=r[k+'_FP_'+suffix]
    if n:close(fp/n,value,'regional ratio')
    else:check(value=='','empty regional denominator')
 check(sum(int(r['near_target_denominator'])>0 for r in ii)==9,'9 valid images')
 check(sum(not int(r['near_target_denominator']) for r in ii)==19,'19 NA images')
 for k in ['internal','near','far','uncertain']:
  s=summary['regions'][k];fp=sum(int(r[k+'_FP']) for r in ii);area=sum(int(r[k+'_area']) for r in ii)
  check(fp==s['FP'] and area==s['area'],'pooled region counts')
  close(fp/sum(int(r['full_FP']) for r in ii),s['pooled_FP_share_not_macro'],'pooled FP share')
  for den,field,valid in [('area','image_macro_FP_per_area','valid_area_images'),('background_pixels','image_macro_FP_per_background','valid_background_images')]:
   values=[int(r[k+'_FP'])/int(r[k+'_'+den]) for r in ii if int(r[k+'_'+den])]
   check(len(values)==s[valid],'macro denominator count')
   close(statistics.mean(values),s[field],'regional macro mean')
 for field,ranges in reviews['D0']['seed_diagnostics'][seed]['qualifying_probability_ranges'].items():
  vals=[float(p[field]) for p in pp if p['point_id'] in qualifying[seed]]
  for name,calc in [('min',min(vals)),('median',statistics.median(vals)),('max',max(vals))]:close(calc,ranges[name],'qualifying ranges')
check(len(misses['2027']&misses['2028'])==ds['shared_misses']==17,'shared misses')
check({p['point_id'] for p in index if int(p['both_missed'])}==misses['2027']&misses['2028'],'shared miss index')
check(sorted(qualifying['2027']&qualifying['2028'])==reviews['D0']['qualifying_in_both_seeds'],'shared qualifying')
sets=[];target_sets=[]
for tag,stage in [('v1','V1'),('v2','V2')]:
 rr=rows(tag+'_images_110.csv');ff=rows(tag+'_target_outcomes_351.csv');s=J(tag+'_summary.json')
 check(len(rr)==110 and len({r['filename'] for r in rr})==110,'Train110 '+tag)
 sets.append({r['filename'] for r in rr});target_sets.append({(r['filename'],r['NL_index']) for r in ff})
 check(len(ff)==351 and len(target_sets[-1])==351,'Train target identities '+tag)
 matched=sum(int(r['matched_pairs']) for r in rr);target_images=sum(int(r['NL_targets'])>0 for r in rr);matched_images=sum(int(r['matched_pairs'])>0 for r in rr)
 check(sum(int(r['NL_targets']) for r in rr)==351 and target_images==30,'Train original target denominator')
 check(matched==s['matched_pairs']==budget[stage+'_matched_points'],'matching count')
 check(matched_images==budget[stage+'_matched_images'],'matching image count')
 close(matched_images/target_images,s['image_coverage'],'image coverage')
 check(s['image_coverage_gate'] is False and matched_images/target_images<budget['coverage_gate'],'coverage not passed')
 for r in rr:
  targets=[f for f in ff if f['filename']==r['filename']]
  check(len(targets)==int(r['NL_targets']),'all per-image targets')
  check(sum(f['outcome']=='MATCHED' for f in targets)==int(r['matched_pairs']),'per-image matches')
  check(int(r['matched_pairs'])<=min(int(r['NL_targets']),int(r['NR_targets'])),'one-to-one cardinality bound')
 counts=dict(Counter(f['outcome'] for f in ff));expected=reviews[stage]['target_failure_modes' if stage=='V1' else 'target_outcomes']
 check(counts==expected,'full failure classification '+tag)
 check(reviews[stage]['status']=='PASS','independent review passed '+tag)
 check(s['new_training']==0 and s['new_model_forward']==0 and not s['Test_accessed'] and not s['Dev_used'],'CPU-only Train matching')
check(sets[0]==sets[1],'same Train images V1/V2')
check(target_sets[0]==target_sets[1],'same Train target identities V1/V2')
check(not (sets[0]&{r['filename'] for r in images}),'Train/Dev no overlap')
v2=rows('v2_images_110.csv');unmatched=[r for r in v2 if int(r['NL_targets'])>0 and int(r['matched_pairs'])==0]
check(len(unmatched)==7 and sum(int(r['NL_targets']) for r in unmatched)==13,'seven uncovered images13 points')
check({r['filename'] for r in unmatched}=={r['filename'] for r in reviews['V2']['images_without_match']},'uncovered identity list')
check(sum(r['geometric_edges']==0 for r in reviews['V2']['images_without_match'])==6,'six without geometry')
check(budget['augmentation_replay_patch_instances']==0 and budget['exposure50pct_gate']=='NOT_EVALUATED','no replay or exposure claim')
check(budget['new_training']==budget['new_model_forward']==budget['new_model_loads']==0 and not budget['official_Test_accessed'] and not budget['GPU_used'],'zero new experiments')
print(json.dumps({'status':'PASS','checks':N,'max_numeric_error':MAX_ERROR,'scope':'public tables and file hashes only; no image-level reconstruction','V1':'FAIL_COVERAGE_13_OF_30','V2':'FAIL_COVERAGE_23_OF_30','replay':0},ensure_ascii=False))
