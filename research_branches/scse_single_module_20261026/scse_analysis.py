"""Frozen existing P1 metrics applied to nnU-Net native-space final probabilities."""
import os,sys,json,csv
from pathlib import Path
import numpy as np
from scse_gate import STAGE,CODE,OUT,protocol,sha
sys.path.insert(0,str(CODE))
from retinal_m0.maples_branch_science_analysis import evaluate_image,metric_values
from retinal_m0.maples_stratified_science_analysis import hard_cldice,paired_bootstrap,bootstrap_indices,dice_readout

def analyze(out=None,fixture=False):
    q=protocol();ref=q['references'];out=Path(out or OUT/'analysis');out.mkdir(exist_ok=False)
    baseline=json.loads(Path(ref['accepted_A0_fixture']).read_text());ids=baseline['fixed_ids_by_metric'];means_want=baseline['absolute_means']['A0']
    refs=json.loads(Path(ref['science_reference_manifest']).read_text())['images'];a0={r['filename']:r for r in json.loads(Path(ref['A0_reuse_binding']).read_text())['probability_maps']}
    if len(refs)!=28 or len(a0)!=28:raise RuntimeError('fixed Dev28 cohort required')
    rows=[]
    for r in refs:
        name=r['filename']
        if sha(r['reference_npz'])!=r['reference_npz_sha256']:raise RuntimeError('frozen GT/reference bundle digest mismatch')
        with np.load(r['reference_npz'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if sha(a0[name]['path'])!=a0[name]['sha256']:raise RuntimeError('original A0 probability changed')
        paths={'A0':Path(a0[name]['path']),'SCSE':Path(a0[name]['path']) if fixture else OUT/'evaluation/step_002000/probabilities'/(name+'.npy')};values={};full={}
        for method,path in paths.items():
            p=np.load(path,allow_pickle=False)
            if p.dtype!=np.float32 or list(p.shape)!=r['native_shape'] or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise RuntimeError('probability dtype/shape/range invalid:'+name)
            record=evaluate_image(p,arrays,'B2',threshold=.5);v=metric_values(record)
            h=hard_cldice((p>=.5)&arrays['FOV'].astype(bool),arrays['GT'],arrays['FOV']);v['hard_cldice']=h['cldice'];record.update(hard_cldice=h['cldice'],topology_precision=h['topology_precision'],topology_sensitivity=h['topology_sensitivity'])
            values[method]={k:v.get(k) for k in ids};full[method]=record
        rows.append({'filename':name,'methods':values,'full_records':full,'SCSE_probability_sha256':sha(paths['SCSE'])})
    by={r['filename']:r for r in rows};indices={n:bootstrap_indices(n) for n in {len(v) for v in ids.values()}};statistics={};means={m:{} for m in ['SCSE','A0']}
    for metric,names in ids.items():
        for method in means:
            values=[by[n]['methods'][method][metric] for n in names]
            if any(v is None or not np.isfinite(v) for v in values):raise RuntimeError('prediction NA on fixed GT valid set:'+metric)
            means[method][metric]=float(np.mean(values))
        if abs(means['A0'][metric]-means_want[metric])>1e-12:raise RuntimeError('existing A0 mean mismatch:'+metric)
        result=paired_bootstrap([by[n]['methods']['SCSE'][metric] for n in names],[by[n]['methods']['A0'][metric] for n in names],indices=indices[len(names)])
        statistics[metric]={'SCSE-A0':{k:v for k,v in result.items() if k not in ['indices','estimates']}}
        if fixture and statistics[metric]['SCSE-A0']['estimate']!=0:raise RuntimeError('CPU fixture not zero contrast')
    readout=dice_readout({k:{'SPATIAL-A0':v['SCSE-A0']} for k,v in statistics.items()})
    endpoint={'status':'CPU_FIXTURE_ONLY' if fixture else 'COMPLETE','model':'original M0 U-Net + one final-decoder sum-scSE; seed2026 fixed2000','comparison':'SCSE-A0; one module comparison; shared original M0 weights','fixed_endpoint':'step_2000.pt','threshold':.5,'absolute_means':means,'metrics':statistics,'fixed_ids_by_metric':ids,'per_image':rows,'bootstrap_indices_by_n':{str(n):v.tolist() for n,v in indices.items()},'readout':readout,'single_seed_repeated_Dev':True,'Test_accessed':False}
    (out/'endpoint_2000.json').write_text(json.dumps(endpoint,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (out/'metrics_per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['filename']+[m+'_'+k for m in ['SCSE','A0'] for k in ids];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({'filename':r['filename'],**{m+'_'+k:r['methods'][m][k] for m in ['SCSE','A0'] for k in ids}})
    lines=['# U-Net单scSE固定2000步终点'+('：CPU替身检查，非训练成绩' if fixture else ''),'','单seed、重复Dev28、单模块探索；固定2000及0.5。','|指标|SCSE|A0|差值|95%区间|','|---|---:|---:|---:|---|']
    for k in ['dice','precision','recall','fp_density','hard_cldice','candidate_path_d1_macro','far_recall','coarse_vessel_d0']:
        d=statistics[k]['SCSE-A0'];lines.append(f'|{k}|{means["SCSE"][k]:.8f}|{means["A0"][k]:.8f}|{d["estimate"]:+.8f}|[{d["lower"]:+.8f},{d["upper"]:+.8f}]|')
    lines+=['','预定规则判读：',json.dumps(readout,ensure_ascii=False,indent=2),'','全部逐图分母/NA、路径与拓扑诊断保留在endpoint JSON的full_records。原病灶邻域9有效图/77位置/19NA及Train/Dev/Test划分保持。单seed不证明跨seed稳定或独立验证；不自动增加训练。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return endpoint

if __name__=='__main__':analyze()
