"""Frozen final-only minus N0 comparison; unique final100/.5, all31 metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,copy,argparse
from common import *
import numpy as np
from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
from retinal_m0.maples_stratified_science_analysis import hard_cldice,paired_bootstrap,bootstrap_indices

def analyze(fixture=False):
    q=protocol();base=read(N0/'analysis/endpoint_100epochs.json');ids=base['fixed_ids_by_metric']
    refs=read(q['references']['science_reference_manifest'])['images']
    if [r['filename'] for r in refs]!=q['Dev_ids']:raise RuntimeError('Dev order mismatch')
    rows=[]
    for ref in refs:
        name=ref['filename']
        if sha(ref['reference_npz'])!=ref['reference_npz_sha256']:raise RuntimeError('GT/FOV reference changed')
        with np.load(ref['reference_npz'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        values={};full={};hashes={};cache={}
        for method,mode in [('N0',None),('N-FinalOnly','final')]:
            path=N0/'dev_predictions'/(name+'.npy') if mode is None or fixture else STAGE/'science'/mode/'dev_predictions'/(name+'.npy')
            if str(path) not in cache:
                h=sha(path);p=np.load(path,allow_pickle=False)
                if p.dtype!=np.float32 or list(p.shape)!=ref['native_shape'] or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('invalid probability')
                record=evaluate_image(p,arrays,'B2',threshold=.5);v=metric_values(record)
                hard=hard_cldice((p>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV'])
                v['hard_cldice']=hard['cldice'];record.update(hard_cldice=hard['cldice'],topology_precision=hard['topology_precision'],topology_sensitivity=hard['topology_sensitivity'])
                cache[str(path)]=({k:v.get(k) for k in ids},record,h)
            values[method],full[method],hashes[method]=copy.deepcopy(cache[str(path)])
        rows.append({'filename':name,'methods':values,'full_records':full,'probability_sha256':hashes})
    by={r['filename']:r for r in rows};indices={n:bootstrap_indices(n) for n in {len(v) for v in ids.values()}}
    means={m:{} for m in ['N0','N-FinalOnly']};stats={}
    for metric,names in ids.items():
        for method in means:
            vals=[by[n]['methods'][method][metric] for n in names]
            if any(v is None or not np.isfinite(v) for v in vals):raise RuntimeError('NA on frozen effective cohort')
            means[method][metric]=float(np.mean(vals))
        if abs(means['N0'][metric]-base['absolute_means']['NNUNET'][metric])>1e-12:raise RuntimeError('baseline reconciliation failed:'+metric)
        d=paired_bootstrap([by[n]['methods']['N-FinalOnly'][metric] for n in names],[by[n]['methods']['N0'][metric] for n in names],indices=indices[len(names)])
        stats[metric]={'N-FinalOnly-N0':{k:v for k,v in d.items() if k not in ['indices','estimates']}}
    s={k:v['N-FinalOnly-N0'] for k,v in stats.items()}
    guards={k:s[k]['lower']>=0 for k in ['precision','far_recall']}
    guards['fp_density']=s['fp_density']['upper']<=0
    guards.update({k:s[k]['estimate']>=0 for k in ['recall','coarse_vessel_d0','hard_cldice']})
    signal=bool(s['dice']['lower']>0);protected=signal and all(guards.values())
    interpretation='candidate_Dice_signal' if signal else 'original_DS_Dice_advantage' if s['dice']['upper']<0 else 'inconclusive_not_equivalent'
    result={'status':'CPU_FIXTURE_ONLY' if fixture else 'COMPLETE_PENDING_SOL_REVIEW','seed':2027,'threshold':.5,'endpoint':'final100epochs','absolute_means':means,'metrics':stats,'fixed_ids_by_metric':ids,'per_image':rows,
            'primary':'ordinary image-macro Dice; N-FinalOnly minus N0','primary_interpretation':interpretation,'guard_checks':guards,'protected_Dice_gain':bool(protected),
            'bootstrap_indices_by_n':{str(n):v.tolist() for n,v in indices.items()},'repeated_Dev_not_independent_validation':True,'Test_accessed':False,
            'scope_limit':'normalized supervision allocation ablation, not matched parameter-gradient magnitude; not full attribution of nnU-Net versus T0 gain'}
    out=STAGE/('cpu/analysis_fixture' if fixture else 'analysis');out.mkdir(exist_ok=True)
    atomic(out/'endpoint_100epochs.json',result,new=True)
    with (out/'metrics_per_image.csv').open('x',encoding='utf-8-sig',newline='') as f:
        columns=['filename']+[m+'_'+k for m in means for k in ids];writer=csv.DictWriter(f,columns);writer.writeheader()
        for row in rows:writer.writerow({'filename':row['filename'],**{m+'_'+k:row['methods'][m][k] for m in means for k in ids}})
    lines=['# nnU-Net深监督分配消融'+('：CPU同模型替身校验，非新实验结果' if fixture else '：终点结果待Sol科研审查'),'','seed2027、固定100epochs、阈值0.5、Dev28；单种子重复使用Dev，不是独立验证。','','|指标|原深监督N0|仅最终输出N-FinalOnly|','|---|---:|---:|']
    for k in ['dice','precision','recall','fp_density','hard_cldice','candidate_path_d1_macro','near_thin_recall_d1','far_recall','coarse_vessel_d0']:
        if k in ids:lines.append('|'+k+'|'+'|'.join(f'{means[m][k]:.8f}' for m in means)+'|')
    d=s['dice'];lines+=['',f'Dice配对变化{100*d["estimate"]:+.4f}个百分点，95%区间[{100*d["lower"]:+.4f}, {100*d["upper"]:+.4f}]。',f'主指标判读：{interpretation}；受保护Dice增益：{bool(protected)}。',json.dumps(guards,ensure_ascii=False),'','全部31项指标、逐图分母、NA及不利案例保留。区间跨零不等于等效；梯度强度未人为配平。不自动扩种子、调阈值或启动新实验。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if fixture and any(v['N-FinalOnly-N0']['estimate']!=0 for v in stats.values()):raise RuntimeError('nonzero identical-model fixture contrast')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-fixture',action='store_true');a=p.parse_args();analyze(a.cpu_fixture)
