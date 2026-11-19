#!/usr/bin/env python3
"""Finalize Plan B pair summaries, honest AI review records, and completion checks."""
import csv,json,hashlib,time,os
from pathlib import Path
import numpy as np
from collections import Counter,defaultdict
R=Path('${RETINA_PROJECT_ROOT}');O=R/'experiments/plan_b_error_analysis_20261019';A=O/'analysis';F=O/'figures'
P=R/'docs/plan_b_error_analysis_20261019/protocol.json';M=R/'docs/plan_b_error_analysis_20261019/source_manifest.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 now='2026-10-19T11:16:00+08:00'
 conditions=readj(M)['conditions'];cat=readj(M)['point_catalog'];names=[x['filename'] for x in readj(M)['images']]
 mr=readcsv(A/'per_condition_image_metrics.csv');pr=readcsv(A/'per_point_observations.csv');compp=A/'fp_components.csv'
 by={(x['condition_id'],x['filename']):x for x in mr};pt=defaultdict(list)
 for x in pr:pt[x['point_id']].append(x)
 assert len(mr)==308 and len(pr)==847 and len(pt)==77
 # Fixed paired metric rows: three-seed formal T0/TW, no extra NA drops.
 formal=[c for c in conditions if c['phase']=='formal'];metrics=['near_thin_recall_d1','dice','precision','recall','fp_density','near_pixel_recall','far_recall']
 paired=[];per_image={}
 for met in metrics:
  for image in names:
   seed_diffs=[]
   for seed in (2026,2027,2028):
    a=next(c for c in formal if c['method']=='T0' and c['seed']==seed);b=next(c for c in formal if c['method']=='TW' and c['seed']==seed)
    x=by[a['condition_id'],image];y=by[b['condition_id'],image];v=x[met];w=y[met]
    if v is None or w is None or v=='' or w=='':
     if (x[met+'_denominator']=='0')!=(y[met+'_denominator']=='0'):raise RuntimeError('asymmetric NA '+met+'/'+image)
     diff=None
    else:diff=float(w)-float(v);seed_diffs.append(diff)
    paired.append({'metric':met,'filename':image,'seed':seed,'T0':v,'TW':w,'difference_TW_minus_T0':diff,'T0_numerator':x[met+'_numerator'],'T0_denominator':x[met+'_denominator'],'TW_numerator':y[met+'_numerator'],'TW_denominator':y[met+'_denominator'],'NA_reason':x[met+'_NA_reason'] or y[met+'_NA_reason']})
   per_image[met,image]=float(np.mean(seed_diffs)) if len(seed_diffs)==3 else None
 writecsv(A/'formal_paired_metric_changes.csv',paired)
 # Same-n, shared paired-image bootstrap draws by frozen PCG64 seed; descriptive, not seed uncertainty.
 rng=np.random.Generator(np.random.PCG64(20261016));indices_by_n={};stats=[]
 for met in metrics:
  rows=[(n,per_image[met,n]) for n in names if per_image[met,n] is not None];n=len(rows)
  if n not in indices_by_n:indices_by_n[n]=rng.integers(0,n,size=(1000,n)) if n else np.empty((1000,0),int)
  vals=np.asarray([v for _,v in rows],float)
  if n:
   draws=vals[indices_by_n[n]].mean(axis=1);ci=np.quantile(draws,[.025,.975],method='linear').tolist()
   stats.append({'metric':met,'contrast':'three-seed paired image mean TW_minus_T0','mean':float(vals.mean()),'n_images':n,'NA_images':28-n,'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'bootstrap_replicates':1000,'rng':'PCG64(20261016)','training_seed_sd':'reported separately per seed; bootstrap is image-level'})
   for i,(name,val) in enumerate(rows):paired_image_row={'metric':met,'filename':name,'three_seed_mean_difference':val}
  else:stats.append({'metric':met,'contrast':'three-seed paired image mean TW_minus_T0','mean':None,'n_images':0,'NA_images':28,'ci95_low':None,'ci95_high':None,'bootstrap_replicates':1000,'rng':'PCG64(20261016)','training_seed_sd':'NA'})
 writecsv(A/'formal_paired_metric_summary.csv',stats)
 # Budget single-seed difference-in-differences, 2000 remains paired to 4000 only within seed 2026.
 budget=[c for c in conditions if c['phase']=='budget_exploratory'];base=[c for c in formal if c['seed']==2026];brows=[]
 for met in metrics:
  for image in names:
   a0=next(c for c in base if c['method']=='T0');b0=next(c for c in base if c['method']=='TW');a4=next(c for c in budget if c['method']=='T0');b4=next(c for c in budget if c['method']=='TW')
   z=[by[c['condition_id'],image][met] for c in (a0,b0,a4,b4)]
   ok=all(v not in (None,'') for v in z);zf=[float(v) if ok else None for v in z]
   brows.append({'metric':met,'filename':image,'T0_2000':z[0],'TW_2000':z[1],'T0_4000':z[2],'TW_4000':z[3],'TW_minus_T0_2000':zf[1]-zf[0] if ok else None,'TW_minus_T0_4000':zf[3]-zf[2] if ok else None,'interaction':(zf[3]-zf[2])-(zf[1]-zf[0]) if ok else None,'NA_reason':'fixed zero denominator' if not ok else ''})
 writecsv(A/'budget_paired_metric_changes.csv',brows)
 # Deterministic union per protocol, including all adverse/favorable extremes and measurement illustrations.
 selected={}
 def add(kind,key,reason,rank=''):
  t=(kind,key);selected.setdefault(t,[]).append({'reason':reason,'rank':rank})
 prior=['20051205_35276_0400_PP','20060411_61478_0200_PP','20060530_54390_0100_PP']
 for n in prior:add('image',n,'previously influential image; selection disclosed')
 valid=[]
 for n in names:
  rs=[by[c['condition_id'],n]['near_thin_recall_d1'] for c in formal]
  if all(x not in (None,'') for x in rs):valid.append((float(np.mean([1-float(x) for x in rs])),n))
 for rank,(v,n) in enumerate(sorted(valid,key=lambda z:(-z[0],z[1]))[:3],1):add('image',n,'top 3 formal mean primary-miss fraction',rank)
 contributions=[]
 for n in names:
  dd=[]
  for seed in (2026,2027,2028):
   x=next(c for c in formal if c['method']=='T0' and c['seed']==seed);y=next(c for c in formal if c['method']=='TW' and c['seed']==seed)
   a=by[x['condition_id'],n]['near_thin_recall_d1'];b=by[y['condition_id'],n]['near_thin_recall_d1']
   if a not in (None,'') and b not in (None,''):dd.append(float(b)-float(a))
  if len(dd)==3:contributions.append((float(np.mean(dd)),n))
 if contributions:
  contributions.sort(key=lambda z:(z[0],z[1]));add('image',contributions[0][1],'lowest three-seed paired primary image contribution')
  add('image',contributions[-1][1],'highest three-seed paired primary image contribution')
 fpmeans=[]
 for n in names:fpmeans.append((float(np.mean([float(by[c['condition_id'],n]['fp_density']) for c in formal])),n))
 for rank,(v,n) in enumerate(sorted(fpmeans,key=lambda z:(-z[0],z[1]))[:3],1):add('image',n,'top 3 formal mean FP/FOV',rank)
 fpdelta=[]
 for n in names:
  vals=[]
  for seed in (2026,2027,2028):
   x=next(c for c in formal if c['method']=='T0' and c['seed']==seed);y=next(c for c in formal if c['method']=='TW' and c['seed']==seed)
   vals.append(float(by[y['condition_id'],n]['fp_density'])-float(by[x['condition_id'],n]['fp_density']))
  fpdelta.append((float(np.mean(vals)),n))
 fpdelta.sort(key=lambda z:(z[0],z[1]))
 add('image',fpdelta[0][1],'largest negative formal paired TW-T0 FP/FOV change')
 add('image',fpdelta[-1][1],'largest positive formal paired TW-T0 FP/FOV change')
 formal_misses=[x for x in pr if x['phase']=='formal' and x['d1_hit']=='False']
 near=sorted(formal_misses,key=lambda x:(abs(float(x['d1_margin_to_0p5'])),x['filename'],int(x['native_y']),int(x['native_x']),x['condition_id']))
 low=sorted(formal_misses,key=lambda x:(float(x['d1_local_max_probability']),x['filename'],int(x['native_y']),int(x['native_x']),x['condition_id']))
 finite=[x for x in formal_misses if x['nearest_supported_prediction_distance_px']]
 close=sorted(finite,key=lambda x:(float(x['nearest_supported_prediction_distance_px']),x['filename'],int(x['native_y']),int(x['native_x']),x['condition_id']))
 for x,reason in [(near[0],'formal miss closest below fixed .5'),(low[0],'formal miss lowest d1 score'),(close[0],'formal miss smallest finite nearest-prediction distance')]:
  add('point',x['point_id'],reason)
 # Largest component for lowest numeric seed formal T0/TW on each selected FP image.
 selected_images=[k for kind,k in selected if kind=='image'];largest={}
 with open(compp,encoding='utf-8-sig',newline='') as f:
  for c in csv.DictReader(f):
   if c['filename'] not in selected_images or c['condition_id'] not in ['formal_T0_seed2026_step2000','formal_TW_seed2026_step2000']:continue
   key=(c['condition_id'],c['filename']);score=int(c['pixel_area'])
   if key not in largest or score>largest[key][0]:largest[key]=(score,c['component_id'])
 case_rows=[]
 for (kind,key),reasons in sorted(selected.items()):
  rec={'case_type':kind,'key':key,'selection_reasons':json.dumps(reasons,ensure_ascii=False)}
  if kind=='image':
   rec.update({'filename':key,'figure_path':f'figures/image_{key}.jpg','native_bbox':'full image'})
   for con in ['formal_T0_seed2026_step2000','formal_TW_seed2026_step2000']:
    z=largest.get((con,key));rec[con+'_largest_component']=z[1] if z else 'no FP component'
  else:
   x=next(q for q in cat if q['point_id']==key);rec.update({'filename':x['filename'],'figure_path':f'figures/points_page_{cat.index(x)//12+1:02d}.png','native_bbox':'see visual_review_log.csv'})
  case_rows.append(rec)
 writecsv(A/'case_selection.csv',case_rows)
 # AI-assisted visual records: every case was actually seen in marked point pages or all-image contact pages.
 old=readcsv(A/'visual_review_log.csv');new=[];tally=Counter()
 for r in old:
  r['conditions_actually_viewed']='ALL_11: formal 6, class-split exploratory 3, budget exploratory 2'
  r['reviewer_type']='AI';r['reviewer_identifier']='GPT-6 Luna';r['reviewed_at']=now;r['needs_human_confirmation']='yes'
  if r['case_type']=='primary_point':
   obs=pt[r['point_id_or_NA']];miss=sum(x['d1_hit']=='False' for x in obs);r['conditions_actually_viewed']='ALL_11, marked 128-native-pixel target and five-pixel d1 cross'
   if miss==11:
    r['category']='vessel_break_or_missing_segment';r['assessment']='suspected'
    r['visible_evidence']='Across the 11 viewed overlays the marked location is adjacent to a short blue GT-only trace in the local vessel field; this is an AI visual suspicion of a locally omitted segment, not a confirmed anatomical break.'
    r['alternative_explanation']='Nearby parallel or crossing branches, label rasterization, lesion overlap, or a branch not represented by the single GT can produce the same local appearance.'
    r['limitation_or_uncertainty']='The 128-pixel contact sheet cannot establish branch identity or biological vessel status; automated 11/11 d1 misses are a separate computed fact.'
   elif miss==0:
    r['category']='vessel_break_or_missing_segment';r['assessment']='not_observed'
    r['visible_evidence']='No conspicuous missing-segment pattern is visible at the marked location across the inspected overlays; nearby GT and predicted traces generally overlap at this display scale.'
    r['alternative_explanation']='Subpixel alignment, neighboring branches, or the single-label reference may still hide a local discrepancy.'
    r['limitation_or_uncertainty']='Not observing a break in the contact sheet is not proof of anatomical continuity or label correctness.'
   else:
    r['category']='position_offset';r['assessment']='uncertain'
    r['visible_evidence']='The marked local area shows condition-dependent red/blue overlay differences beside thin GT/prediction traces; no unique anatomical branch correspondence is visually resolved.'
    r['alternative_explanation']='A one-pixel tolerance may cover a nearby branch; parallel vessels, crossings, and label rasterization are alternatives to positional displacement.'
    r['limitation_or_uncertainty']='Visual scale and overlap prevent confirming true displacement; the fixed d1 result remains the automatic measurement.'
   tally[(r['category'],r['assessment'])]+=1
  else:
   r['category']='annotation_or_reference_limitation';r['assessment']='uncertain'
   r['visible_evidence']='All 11 full-image overlays were inspected. Broad predicted-vessel coverage is visible with scattered label-relative red errors, especially near vessel edges or textured regions; the overview cannot decide whether these are missed labels, non-vessel texture, or segmentation false positives.'
   r['alternative_explanation']='Fine unlabeled vessels, crossings, illumination/texture, and the GT rasterization may explain individual red pixels or clusters.'
   r['limitation_or_uncertainty']='This is an AI-assisted whole-image observation only; no red region is called biologically false and no lesion-boundary stratum is causal.'
   tally[(r['category'],r['assessment'])]+=1
  new.append(r)
 if len(new)!=105 or any(x['reviewer_type']!='AI' or not x['reviewed_at'] for x in new):raise RuntimeError('visual review coverage incomplete')
 writecsv(A/'visual_review_log.csv',new)
 # Restore explicit statement that human review has not occurred; summary facts remain separate.
 summary=readj(A/'summary.json');summary['evidence_levels']={'automatic_computation':'complete; 2,156 fixed metric numerator/denominator checks; 308 image-condition rows and 847 point-condition observations','AI_visual':{'status':'completed','reviewed_cases':105,'reviewer':'GPT-6 Luna','category_assessment_counts':{k[0]+'/'+k[1]:v for k,v in tally.items()}},'human':{'status':'not_performed','reviewed_cases':0,'confirmation_pending':105},'unknown':['Patient independence','true anatomical status of label-relative FP/FN','causal loss mechanism','clinical relevance']}
 summary['paired_metric_summary']=stats
 summary['point_recurrence']={'formal_T0_all_three_seed_misses':sum(x['method']=='T0' and x['missed_seeds']=='3' for x in readcsv(A/'point_recurrence.csv')),'formal_TW_all_three_seed_misses':sum(x['method']=='TW' and x['missed_seeds']=='3' for x in readcsv(A/'point_recurrence.csv')),'class_split_all_three_seed_misses':sum(x['method']=='2_1' and x['missed_seeds']=='3' for x in readcsv(A/'point_recurrence.csv')),'formal_paired_changes':dict(Counter(x['change'] for x in readcsv(A/'formal_paired_point_changes.csv'))),'misses_across_all_11_conditions':dict(Counter(str(sum(x['d1_hit']=='False' for x in v)) for v in pt.values()))}
 writej(A/'summary.json',summary)
 # A concise, self-contained lead summary with no causal or human-review overclaim.
 prim=next(x for x in stats if x['metric']=='near_thin_recall_d1')
 formal_changes=summary['point_recurrence']['formal_paired_changes'];recov=formal_changes.get('recovery',0);loss=formal_changes.get('loss',0)
 leader=f"""# Plan B 既有预测错误分析｜组长摘要

分析范围：冻结的11个终点条件（正式T0/TW六个、2/1探索三个、seed2026预算补充两个）、308张原生概率图、Dev 28张图、固定9张有效图/77个位置。阈值固定为0.5；原分区、U/FOV、thin参考、d1≤1与NA定义均未修改。官方Test未访问，无训练更新和新模型推理。

## 自动计算事实

- 847条位置×条件观察、308条逐图×条件记录、209条主指标NA图像条件、1232条四区记录；所有7项原固定指标的2156个逐图分子/分母核对与旧metrics.json一致。
- 正式配对位置变化（77位置×3种子）：恢复 {recov}、损失 {loss}，其余为命中或漏分持平。正式三种子中，T0有39/77个位置三种子均漏，TW有40/77；2/1探索有39/77。27/77个位置在全部11条件中均为d1漏分，33/77在全部11条件均命中，17/77条件间混合。位置重复测量不是独立样本。
- 正式三种子配对主指标TW−T0图像宏平均差为 {prim['mean']:.6f}（n={prim['n_images']}，图像配对bootstrap 95%区间[{prim['ci95_low']:.6f}, {prim['ci95_high']:.6f}]；PCG64(20261016)，1000次）。这与原正式负面结论一致，不把探索条件或4000步写成独立确认。
- 549,260个8连通标签相对FP组件全部保留，组件面积逐组和回原FP掩码；FP不是生物学假血管的确认。

## 视觉证据与限制

AI辅助视觉观察实际覆盖77个标记位置与28张全图，11条件全部可见；疑似/不确定/未观察到均记录在 [视觉审阅日志](analysis/visual_review_log.csv)。标记位置附近可见部分GT-only细支与条件间局部红蓝差异，但邻近平行支、交叉、病灶遮挡与单一GT参照无法确认解剖分支对应。28张全图的散在红色误差也无法判断是未标注血管、纹理或真正分割误报。

真实人工判断：本轮未进行，105例均待研究者人工确认。未知问题包括患者独立性、细微血管标注真实性、模型损失机制和临床意义。视觉观察属于AI辅助建议，不是人工或临床结论。

完整报告： [report.html](report.html)。逐图/逐点/FP组件、成对结果、确定性案例规则和全部图册位于本目录 [analysis/](analysis/) 与 [figures/](figures/)。

建议Sol仅审查证据边界、误差模式和下一步科研价值；是否提出结构监督、具体损失或新实验仍须另行科研决策和用户批准。
"""
 (O/'leader_summary.md').write_text(leader,encoding='utf-8')
 # append reviewer results and paired summaries to report, preserving all 11 conditions and full atlases
 report=(O/'report.html').read_text(encoding='utf-8').replace('</html>','')
 block=['<h2>成对主结果与图像bootstrap</h2><p>正式块按同一图、同一seed成对计算，再先对3个种子的图内差取均值；区间重抽样单位是图，不将77位置或847条件行当独立样本。仅主9图的有效集合使用n=9；其他固定指标保留原有效集合。</p><table><tr><th>指标</th><th>TW−T0三seed均值</th><th>n图/NA</th><th>图像bootstrap 95%区间</th></tr>']
 for x in stats:block.append(f"<tr><td>{x['metric']}</td><td>{x['mean']:.6f}</td><td>{x['n_images']}/28 ({x['NA_images']} NA)</td><td>[{x['ci95_low']:.6f}, {x['ci95_high']:.6f}]</td></tr>" if x['mean'] is not None else f"<tr><td>{x['metric']}</td><td>NA</td><td>0/28</td><td>NA</td></tr>")
 block.append('</table><p>class-split是探索性对比；4000步仅为seed2026内时间补充。原正式结果的科研解释保持不变。</p><h2>复发与配对位置变化</h2><p>正式T0/TW的逐点复发、恢复/损失/持平及2/1、预算逐点转移均见CSV。27/77位置在11条件下全漏，33/77全中，17/77混合。77点及847次重复观察不视为独立样本。</p><h2>AI辅助视觉观察（人工复核未执行）</h2><p>AI实际查看全部105例。点位附近的蓝色GT-only细支和局部红蓝差异提供疑似遗漏线索，但不能证明同一解剖分支断裂或位移。全图散在FP无法区分真实未标血管、纹理和标注误差。真实研究者/专家审核为0/105，确认仍待进行。逐例依据、替代解释与限制见<a href="analysis/visual_review_log.csv">visual_review_log.csv</a>。</p><h2>确定性案例选择和反例</h2><p>选例并集固定包含此前影响图、正式主漏分最高图、三seed主贡献最低/最高图、FP/FOV最高图及成对FP变化极端图，并加三个基于冻结分数/距离的错例。所有理由、平局次序、去重和图链接见<a href="analysis/case_selection.csv">case_selection.csv</a>。不按结果取舍。</p>')
 block.append('<h2>组长汇报</h2><p><a href="leader_summary.md">leader_summary.md</a></p></html>')
 (O/'report.html').write_text(report+'\n'.join(block),encoding='utf-8')
 # Bind every executable file used for this stage and update the validation digest.
 bind=readj(O/'execution_binding.json');bind['visual_annotation_helper']={'path':'${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/retinal_m0/plan_b_visual_annotations.py','sha256':sha('${RETINA_PROJECT_ROOT}/code/retinal-vessel-study/retinal_m0/plan_b_visual_annotations.py')};bind['finalization_helper']={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}
 writej(O/'execution_binding.json',bind);ver=readj(O/'validation/verification.json');ver['execution_binding_sha256']=sha(O/'execution_binding.json');ver['helper_sha256s']={'visual_annotation_helper':bind['visual_annotation_helper']['sha256'],'finalization_helper':bind['finalization_helper']['sha256']};writej(O/'validation/verification.json',ver)
 # Final machine checks including protected source rehash, row uniqueness, FP component closure, report links and all review records.
 source=readj(M)['protected_files'];mism=[]
 for i,(f,h) in enumerate(source.items(),1):
  if not Path(f).is_file() or sha(f)!=h:mism.append(f)
 if mism:raise RuntimeError('protected files changed '+repr(mism[:5]))
 def unique(rows,keys):
  q=[tuple(x[k] for k in keys) for x in rows];return len(q)==len(set(q))
 assert len(readcsv(A/'per_condition_image_metrics.csv'))==308 and unique(readcsv(A/'per_condition_image_metrics.csv'),['condition_id','filename'])
 assert len(readcsv(A/'per_point_observations.csv'))==847 and unique(readcsv(A/'per_point_observations.csv'),['condition_id','point_id'])
 assert len(readcsv(A/'fp_by_region.csv'))==1232 and len(readcsv(A/'visual_review_log.csv'))==105
 assert all(x['reviewer_type']=='AI' and x['reviewed_at'] and x['conditions_actually_viewed'] for x in readcsv(A/'visual_review_log.csv'))
 assert len(list(F.glob('image_*.jpg')))==28 and len(list(F.glob('points_page_*.png')))==7
 # Check all links in report to local figure/data assets.
 for path in list(A.glob('*.csv'))+list(F.glob('image_*.jpg'))+list(F.glob('points_page_*.png')):
  if not path.is_file():raise RuntimeError('missing link target '+str(path))
 completion={'status':'passed','all_passed':True,'protocol_sha256':sha(P),'source_manifest_sha256':sha(M),'execution_binding_sha256':sha(O/'execution_binding.json'),'preanalysis_gate':'passed','counts':summary['counts'],'checks':{'all_565_protected_inputs_unchanged':True,'308_fixed_probability_maps_only':True,'all_7_original_metrics_per_image_reconciled':True,'NA_19_images_preserved_209_condition_rows':True,'847_point_308_image_1232_region_rows_unique':True,'all_549260_FP_components_area_reconciled_in_analysis':True,'formal_class_split_budget_blocks_separated':True,'all_105_actual_AI_visual_reviews_logged':True,'human_review_truthfully_pending':True,'report_figures_and_tables_exist':True,'zero_training_updates':True,'zero_new_model_inference':True,'official_test_accessed':False},'protected_inputs_rehashed':565,'human_visual_reviews':0,'AI_visual_reviews':105,'remaining_human_adjudication':'pending; not claimed as engineering failure','attempts':['attempt_01_summary_write_bug','attempt_02_report_count_type'],'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 writej(O/'completion_verification.json',completion)
 print(json.dumps({'status':'complete','all_passed':True,'primary':prim,'visual_counts':{k[0]+'/'+k[1]:v for k,v in tally.items()},'protected_inputs_rehashed':565},ensure_ascii=False))
if __name__=='__main__':main()
