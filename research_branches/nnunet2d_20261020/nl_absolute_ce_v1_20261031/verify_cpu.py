"""Independent replay evidence audit; does not import the replay/mapping implementation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,math,time,resource,collections
import numpy as np
from functools import lru_cache

BASE=Path('${RETINA_PROJECT_ROOT}/experiments/nnunet2d_branch_20261020')
STAGE=BASE/'deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v1_20261031'
OUT=STAGE/'results';OLD=STAGE.parent/'train_pairing_cpu_v1'
checks=0;errors=[];start=time.monotonic();oracle_queries=0
def check(label,value):
    global checks
    checks+=1
    if not value:
        errors.append(label)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
def csvsave(name,rows):
    with (OUT/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
@lru_cache(5)
def masks(name):
    with np.load(OLD/'cache'/(name+'.npz'),allow_pickle=False) as a:
        return a['GT'].astype(bool),a['near'].astype(bool)&a['Omega'].astype(bool)

def independent_forward(coords,r):
    p=np.asarray(coords,float).reshape(-1,2)-r['bbox_lbs'][-2:]
    if r['affine'] is None:
        out=p-np.floor(r['center'])+np.asarray(r['output_shape'])//2
    else:
        out=np.linalg.solve(np.asarray(r['affine']), (p-np.asarray(r['center'])+.5).T).T
        out+=(np.asarray(r['output_shape'])-1)*.5
    for axis in r['axes']:out[:,axis]=r['output_shape'][axis]-1-out[:,axis]
    return np.rint(out).astype(int)

def oracle(queries,r,gt,domain):
    """Independent four-neighbor native GT one-hot interpolation + nearest region mask."""
    global oracle_queries
    if not queries:return
    src=np.array([q['sampled_native'] for q in queries],float)
    actual_gt=np.array([q['GT'] for q in queries],int)
    actual_domain=np.array([q['domain'] for q in queries],bool)
    # Check sampled grid against independently derived affine at the rounded output pixel.
    out=np.array([q['pixel'] for q in queries],float)
    for axis in r['axes']:out[:,axis]=r['output_shape'][axis]-1-out[:,axis]
    if r['affine'] is None:
        expected=out+np.floor(r['center'])-np.asarray(r['output_shape'])//2+r['bbox_lbs'][-2:]
    else:
        expected=(out-(np.asarray(r['output_shape'])-1)/2)@np.asarray(r['affine']).T
        expected+=np.asarray(r['center'])-.5+r['bbox_lbs'][-2:]
    check('source affine '+str((r['seed'],r['attempt'],r['crop'])),bool(np.max(abs(src-expected))<.002))
    near_idx=np.rint(src).astype(int)
    inside=(near_idx[:,0]>=0)&(near_idx[:,0]<gt.shape[0])&(near_idx[:,1]>=0)&(near_idx[:,1]<gt.shape[1])
    dom=np.zeros(len(src),bool);idx=np.flatnonzero(inside)
    dom[idx]=domain[near_idx[idx,0],near_idx[idx,1]]
    check('domain values',np.array_equal(dom,actual_domain))
    local=src-np.asarray(r['bbox_lbs'][-2:])
    input_shape=np.asarray(r['input_shape'])
    sample_inside=np.all((local>=0)&(local<=input_shape-1),axis=1)
    floor=np.floor(src).astype(int);frac=src-floor
    values=np.zeros((len(src),3),float)
    for dy,dx in [(0,0),(0,1),(1,0),(1,1)]:
        ij=floor+[dy,dx]
        in_native=(ij[:,0]>=0)&(ij[:,0]<gt.shape[0])&(ij[:,1]>=0)&(ij[:,1]<gt.shape[1])
        lo=ij-np.asarray(r['bbox_lbs'][-2:]);in_patch=np.all((lo>=0)&(lo<input_shape),axis=1)
        label=np.full(len(src),-1,int);ii=np.flatnonzero(in_native)
        label[ii]=gt[ij[ii,0],ij[ii,1]].astype(int)
        weight=(frac[:,0] if dy else 1-frac[:,0])*(frac[:,1] if dx else 1-frac[:,1])
        for j,lab in enumerate([-1,0,1]):values[:,j]+=weight*(label==lab)*in_patch
    # Native path compares float16 one-hot magnitudes; lowest label wins exact ties.
    quant=(values*1000).astype(np.float16)
    pred=np.array([-1,0,1])[quant.argmax(axis=1)]
    pred[~sample_inside]=-1;pred[pred==-1]=0
    bad=np.flatnonzero(pred!=actual_gt)
    # Strictly report discrepancies; never silently accept oracle differences.
    check('native onehot GT values',len(bad)==0)
    if len(bad) and len(errors)<5:print('ORACLE_DETAILS',[(queries[i],int(pred[i]),values[i].tolist()) for i in bad[:2]],flush=True)
    oracle_queries+=len(src)

def main():
    mapping=json.loads((OLD/'mapping.json').read_text())
    receipt=json.loads((OUT/'execution_receipt.json').read_text())
    binding=json.loads((STAGE/'source_binding.json').read_text())
    for p,h in binding['immutable_files'].items():check('source '+p,sha(p)==h)
    for p,h in json.loads((STAGE/'SOL_CACHE_BINDING.json').read_text()).items():check('cache '+p,sha(p)==h)
    tests=json.loads((OUT/'cpu_implementation_tests.json').read_text());check('all CPU formula tests',tests['status']=='PASS' and all(x['passed'] for x in tests['tests']))
    selection_tests=json.loads((OUT/'cpu_selection_tests.json').read_text())
    check('synthetic selection cases',selection_tests['status']=='PASS' and all(x['passed'] for x in selection_tests['tests']))
    auth=json.loads((STAGE/'protocol.json').read_text())['authorization']
    check('CPU-only authorization',auth['status']=='CPU_ONLY_AUTHORIZED' and not any(auth[k] for k in ['GPU_allowed','model_forward_allowed','optimizer_step_allowed','Test_allowed','GitHub_modification_allowed']))
    expected_history={}
    for seed in [2027,2028]:
        path=BASE/f'replication_seed2027_2028_100epochs_v1_20261026/runs/seed_{seed}/attempts.jsonl'
        with path.open() as f:
            for _ in range(1000):
                x=json.loads(next(f));expected_history[(seed,x['attempt'])]=x
    batchrows=[json.loads(x) for x in (OUT/'replay_batches.jsonl').read_text().splitlines()]
    actual_batches={}
    for r in batchrows:
        if r['kind']!='AUDITED':continue
        key=(r['seed'],r['attempt']);expected=expected_history[key]
        check('unique batch',key not in actual_batches);actual_batches[key]=r
        for field in ['ids','input_sha256','target_sha256']:check('history '+field,r[field]==expected[field])
        check('sidecar RNG proof',r['native_rng_unchanged'])
    imrows={(s,name):dict(seed=s,image=name,native_target_count=len(mapping[name]['NL']),audited_crops=0,nonempty_crops=0,
         visible=0,after_GT_domain=0,after_collision=0,with_negative=0,precap=0,selected=0,negative_reuse=0,mass=0.)
         for s in [2027,2028] for name in mapping}
    targetrows={(s,name,i):dict(seed=s,image=name,native_index=i,native_y=t['y'],native_x=t['x'],visible=0,selected=0,
        OUTSIDE_OUTPUT=0,GT_DOMAIN_INVALID=0,POSITIVE_COLLISION=0,NO_NEGATIVE=0,CAP_EXCLUDED=0,SELECTED=0)
        for s in [2027,2028] for name in mapping for i,t in enumerate(mapping[name]['NL'])}
    crops=[];nonempty_batch=collections.defaultdict(set);unique_crops=set();observed_seeds=set()
    with (OUT/'selection_trace.jsonl').open() as f:
        for line in f:
            r=json.loads(line);seed=r['seed'];observed_seeds.add(seed);key=(seed,r['attempt'],r['crop'])
            check('unique crop',key not in unique_crops);unique_crops.add(key)
            if (seed,r['attempt']) not in actual_batches:check('crop bound to batch',False);continue
            check('crop ID bound',actual_batches[(seed,r['attempt'])]['ids'][r['crop']]==r['image'])
            native=mapping[r['image']]['NL'];check('complete target identities',len(r['targets'])==len(native))
            row=imrows[(seed,r['image'])];row['audited_crops']+=1
            counters=dict(visible=0,after_GT_domain=0,after_collision=0,with_negative=0,precap=0,selected=len(r['selected']))
            used=set();eligible=[];queries=[]
            for t in r['targets']:
                orig=native[t['native_index']];tr=targetrows[(seed,r['image'],t['native_index'])]
                tr[t['status']]+=1;check('native point identity',[t['native_y'],t['native_x']]==[orig['y'],orig['x']])
                point=independent_forward([[orig['y'],orig['x']]],r)[0]
                check('independent forward point',point.tolist()==t['positive'])
                visible=bool(np.all(point>=0) and np.all(point<r['output_shape']))
                if not visible:check('outside status',t['status']=='OUTSIDE_OUTPUT');continue
                counters['visible']+=1;tr['visible']+=1
                queries.append({'pixel':t['positive'],'sampled_native':t['sampled_native'],'GT':t['GT_at_positive'],'domain':t['domain_at_positive']})
                valid=t['GT_at_positive']==1 and t['domain_at_positive']
                if not valid:check('GT/domain rejection',t['status']=='GT_DOMAIN_INVALID');continue
                counters['after_GT_domain']+=1
                p=tuple(point)
                if p in used:check('collision rejection',t['status']=='POSITIVE_COLLISION');continue
                used.add(p);counters['after_collision']+=1
                bg=np.asarray(orig['background_coords']).reshape(-1,2)
                expected=independent_forward(bg,r)
                keep=np.all(expected>=0,axis=1)&np.all(expected<r['output_shape'],axis=1)
                ids=np.flatnonzero(keep).tolist()
                check('all visible backgrounds recorded',[v['native_index'] for v in t['backgrounds']]==ids)
                valid_neg=[]
                for v in t['backgrounds']:
                    check('background pixel',v['pixel']==expected[v['native_index']].tolist());queries.append(v)
                    if v['GT']==0 and v['domain']:valid_neg.append(tuple(v['pixel']))
                valid_neg=sorted(set(valid_neg))
                check('complete available negatives',list(map(list,valid_neg))==t['legal_negatives_before_global_conflict'])
                if not valid_neg:check('no-negative rejection',t['status']=='NO_NEGATIVE');continue
                eligible.append((t,valid_neg));counters['with_negative']+=1;counters['precap']+=1
            if queries:
                gt,domain=masks(r['image']);oracle(queries,r,gt,domain)
            # Independently replay prescribed selection in original target order.
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261031,seed,r['attempt'],r['crop']])))
            eligible.sort(key=lambda item:(item[0]['native_y'],item[0]['native_x']))
            if len(eligible)>16:eligible=[eligible[int(i)] for i in sorted(rng.choice(len(eligible),16,replace=False))]
            expected_selected=[]
            for t,negs in eligible:
                picked=[list(negs[int(i)]) for i in sorted(rng.choice(len(negs),min(2,len(negs)),replace=False))]
                expected_selected.append((t['native_index'],t['positive'],picked))
            check('private RNG selected groups',expected_selected==[(g['native_index'],g['positive'],g['negatives']) for g in r['selected']])
            for g in r['selected']:
                targetrows[(seed,r['image'],g['native_index'])]['selected']+=1
            check('all stage counters',counters==r['counts'])
            for k,v in counters.items():row[k]+=v
            row['negative_reuse']+=r['negative_reuse']
            if r['selected']:
                row['nonempty_crops']+=1;row['mass']+=.5;nonempty_batch[seed].add(r['attempt'])
            crops.append(dict(seed=seed,attempt=r['attempt'],crop=r['crop'],image=r['image'],**counters,negative_reuse=r['negative_reuse']))
    summaries=[]
    for seed in [2027,2028]:
        ims=[v for (s,_),v in imrows.items() if s==seed]
        batches=sorted(a for s,a in actual_batches if s==seed)
        check('contiguous attempt prefix',batches==list(range(1,len(batches)+1)))
        check('two crops per matched attempt',sum(1 for k in unique_crops if k[0]==seed)==2*len(batches))
        visible=sum(v['visible'] for v in ims);selected=sum(v['selected'] for v in ims);mass=sum(v['mass'] for v in ims)
        neff=mass*mass/sum(v['mass']**2 for v in ims) if mass else 0.
        retention=selected/visible if visible else 0.
        passed=dict(nonempty_batches=len(nonempty_batch[seed])>=32,retained_fraction=retention>=.5,Neff=neff>=3)
        complete=len(batches)==1000
        result=dict(seed=seed,attempts=len(batches),audited_crops=2*len(batches),visible=visible,selected=selected,
            nonempty_batches=len(nonempty_batch[seed]),retained_fraction=retention,Neff=neff,gates=passed,
            status='PASS' if complete and all(passed.values()) else ('NOT_RUN' if not batches else ('FAIL' if complete else 'INCOMPLETE')),
            top1_mass_fraction=max(v['mass'] for v in ims)/mass if mass else None,
            top3_mass_fraction=sum(sorted((v['mass'] for v in ims),reverse=True)[:3])/mass if mass else None)
        summaries.append(result)
        for v in ims:v['normalized_mass']=v['mass']/mass if mass else 0.;v['run_status']=result['status']
    check('all110 rows eachseed',len(imrows)==220);check('all351 targets eachseed',len(targetrows)==702)
    csvsave('replay_crops.csv',crops) if crops else None
    csvsave('replay_images_110.csv',list(imrows.values()));csvsave('replay_targets_351.csv',list(targetrows.values()))
    save('independent_review.json',{'status':'PASS' if not errors else 'FAIL','checks':checks,'errors':errors,
        'independent_GT_queries':oracle_queries,'method':'Separate source/hash/history, explicit four-neighbor GT interpolation and geometry oracle, private RNG, stage-count and exposure rederivation; no imported replay implementation',
        'elapsed_seconds':time.monotonic()-start,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'seeds':summaries})
    overall='PASS' if not errors and all(x['status']=='PASS' for x in summaries) and not receipt['failure'] else 'FAIL_OR_INCOMPLETE'
    save('cpu_summary.json',{'status':overall,'independent_checks_passed':not errors,'seeds':summaries,
        'formula_tests':tests['count'],'synthetic_selection_tests':selection_tests['count'],'oracle_queries':oracle_queries,'replay_receipt_status':receipt['status'],
        'CPU_execution_complete':True,'GPU_engineering':'HOLD','scientific_training':'HOLD','lambda':None,
        'limitations':['No real network/AMP/parameter gradients validated','No medical human validation','Prefix replay does not prove future full scientific trajectory pairing','Native target size/image concentration retained','No NR: no lesion specificity'],
        'model_forwards':0,'optimizer_updates':0,'Test_access':False,'Dev_pixel_access':False})
    print(json.dumps({'status':overall,'checks':checks,'errors':errors[:10],'seeds':summaries},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
