"""Bounded Train-only native replay. No Trainer/model construction or GPU discovery."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,csv,hashlib,time,random,resource,sys,traceback,signal,threading
from types import SimpleNamespace
from functools import lru_cache
import numpy as np
import torch
from aux_loss import choose_groups,map_points,inverse_points

ROOT=Path('${RETINA_PROJECT_ROOT}')
BASE=ROOT/'experiments/nnunet2d_branch_20261020'
STAGE=BASE/'deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v1_20261031'
OUT=STAGE/'results'
OLD=STAGE.parent/'train_pairing_cpu_v1'
HISTORY=BASE/'replication_seed2027_2028_100epochs_v1_20261026/runs'
START=time.monotonic()

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def digest(v):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for a in x:add(a)
        else:raise TypeError(type(x))
    add(v);return h.hexdigest()

def write_json(p,x):
    tmp=p.with_suffix(p.suffix+'.pending')
    with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    os.replace(tmp,p)

def append(f,x):f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def resources(enforce=True):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    elapsed=time.monotonic()-START
    if enforce and (elapsed>7200 or rss>8192):raise RuntimeError('CPU resource cap reached')
    return {'elapsed_seconds':elapsed,'peak_RSS_MiB':rss}

@lru_cache(maxsize=5)
def native(name):
    with np.load(OLD/'cache'/(name+'.npz'),allow_pickle=False) as z:
        return {k:z[k].astype(bool) for k in ['GT','FOV','Omega','near']}

def build_loader(seed):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
    from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
    pm=PlansManager(str(BASE/'preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json'))
    config=pm.get_configuration('2d')
    dsjson=json.loads((BASE/'preprocessed/Dataset501_MAPLES/dataset.json').read_text())
    lm=pm.get_label_manager(dsjson)
    patch=tuple(config.patch_size);assert patch==(1024,1536) and config.batch_size==2
    assert pm.transpose_forward==[0,1,2]
    rotation=(-np.pi,np.pi)  # aspect exactly1.5, native >1.5 branch is false
    initial=get_patch_size(patch,rotation,rotation,rotation,(.85,1.25))
    scales=list(list(i) for i in 1/np.cumprod(np.vstack(config.pool_op_kernel_sizes),axis=0))[:-1]
    transform=nnUNetTrainer.get_training_transforms(patch,rotation,scales,(0,1),False,
        use_mask_for_norm=config.use_mask_for_norm,is_cascaded=False,foreground_labels=lm.foreground_labels,
        regions=None,ignore_label=lm.ignore_label)
    folder=BASE/'preprocessed/Dataset501_MAPLES'/config.data_identifier
    cls=infer_dataset_class(str(folder));ds=cls(str(folder))
    train=json.loads((ROOT/'docs/full_fov_thin_skeleton_bce_20261019/train_manifest.json').read_text())['rows']
    expected={r['filename'] for r in train}
    assert set(ds.identifiers)==expected and len(expected)==110
    context={};records=[]
    original_load=ds.load_case
    def load_case(name):
        data,seg,prev,props=original_load(name)
        assert prev is None and tuple(data.shape[1:])==tuple(props['shape_before_cropping'])
        assert props['bbox_used_for_cropping']==[[0,s] for s in props['shape_before_cropping']]
        context.clear();context['image']=name;context['native_shape']=list(seg.shape[-2:])
        return data,seg,prev,props
    ds.load_case=load_case
    spatial=[t for t in transform.transforms if isinstance(t,SpatialTransform)]
    mirror=[t for t in transform.transforms if isinstance(t,MirrorTransform)]
    assert len(spatial)==len(mirror)==1
    st=spatial[0];mt=mirror[0]
    gp=st.get_parameters;mp=mt.get_parameters
    def get_parameters(**d):
        p=gp(**d)
        if p['elastic_offsets'] is not None:raise RuntimeError('Unexpected elastic transform')
        context['spatial']=p;context['input_shape']=list(d['image'].shape[-2:]);return p
    def mirror_parameters(**d):
        p=mp(**d);context['mirror']=list(p['axes']);return p
    st.get_parameters=get_parameters;mt.get_parameters=mirror_parameters
    class Sidecar:
        def __call__(self,**data):
            before=None
            out=transform(**data)
            # Random native transforms have finished; sidecar computation must consume no native RNG.
            before=(random.getstate(),np.random.get_state(),torch.random.get_rng_state().clone())
            c={k:v for k,v in context.items() if k!='spatial'}
            par=context['spatial']
            c.update(center=list(par['center_location_in_pixels']),affine=None if par['affine'] is None else par['affine'].tolist(),
                     axes=c.pop('mirror'),output_shape=list(out['image'].shape[-2:]))
            if audit_active[0]:
                rec=process_crop(c,par['grid'],out,seed,audit_active[0],len(records))
                records.append(rec)
            assert before[0]==random.getstate()
            ns=np.random.get_state();assert all(np.array_equal(a,b) for a,b in zip(before[1],ns))
            assert torch.equal(before[2],torch.random.get_rng_state())
            return out
    loader=nnUNetDataLoader(ds,2,initial,patch,lm,oversample_foreground_percent=.33,sampling_probabilities=None,
        pad_sides=None,transforms=Sidecar(),probabilistic_oversampling=False)
    original_bbox=loader.get_bbox
    def bbox(*a,**kw):
        lo,hi=original_bbox(*a,**kw);context['bbox_lbs']=list(map(int,lo));context['bbox_ubs']=list(map(int,hi));return lo,hi
    loader.get_bbox=bbox;loader.set_thread_id(0)
    # Reproduce exactly the single producer's CPU seeds, not the main-process post-network RNG.
    random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed);torch.set_num_threads(1)
    audit_active=[0]
    return loader,records,audit_active

def mapped_domain(points,c,grid,domain):
    """Read masks at exact sampled source centers, nearest-neighbor mask convention."""
    points=np.asarray(points,int).reshape(-1,2)
    if len(points)==0:return np.zeros(0,bool),[]
    pre=points.copy()
    for a in c['axes']:pre[:,a]=c['output_shape'][a]-1-pre[:,a]
    if grid is None:
        src=pre+(np.floor(c['center']).astype(int)-np.asarray(c['output_shape'])//2)
    else:
        # grid stores x,y; align_corners=False inverse is ((g+1)*size-1)/2
        xy=grid[pre[:,0],pre[:,1]].numpy().astype(np.float64)
        src=((xy[:,::-1]+1)*np.asarray(c['input_shape'])-1)/2
    src+=np.asarray(c['bbox_lbs'][-2:])
    expected=inverse_points(points,**{k:c[k] for k in ['bbox_lbs','input_shape','output_shape','center','affine','axes']})
    if np.max(abs(src-expected))>.002:raise RuntimeError('sampled grid/affine convention mismatch')
    ij=np.rint(src).astype(int)
    valid=(ij[:,0]>=0)&(ij[:,0]<domain.shape[0])&(ij[:,1]>=0)&(ij[:,1]<domain.shape[1])
    ix=np.flatnonzero(valid);valid[ix]&=domain[ij[ix,0],ij[ix,1]]
    return valid,src.tolist()

def process_crop(c,grid,out,seed,attempt,crop):
    shape=c['output_shape'];name=c['image'];target=out['segmentation'][0][0].numpy()
    targets=MAPPING[name]['NL'];trace=[];candidates=[];used=set()
    counts=dict(visible=0,after_GT_domain=0,after_collision=0,with_negative=0,precap=0,selected=0)
    if targets:
        masks=native(name);domain=masks['Omega']&masks['near']
    for idx,g in enumerate(targets):
        native_yx=[g['y'],g['x']]
        kw={k:c[k] for k in ['bbox_lbs','input_shape','output_shape','center','affine','axes']}
        px,continuous=map_points([native_yx],**kw);y,x=map(int,px[0])
        row={'native_index':idx,'native_y':int(g['y']),'native_x':int(g['x']),
             'positive':[y,x],'continuous':continuous[0].tolist(),'status':'OUTSIDE_OUTPUT'}
        trace.append(row)
        if not(0<=y<shape[0] and 0<=x<shape[1]):continue
        counts['visible']+=1
        valid,src=mapped_domain([[y,x]],c,grid,domain)
        row['sampled_native']=src[0];row['GT_at_positive']=int(target[y,x]);row['domain_at_positive']=bool(valid[0])
        if target[y,x]!=1 or not valid[0]:row['status']='GT_DOMAIN_INVALID';continue
        counts['after_GT_domain']+=1
        if (y,x) in used:row['status']='POSITIVE_COLLISION';continue
        used.add((y,x));counts['after_collision']+=1
        bg=np.asarray(g['background_coords'],int).reshape(-1,2)
        bpx,_=map_points(bg,**kw)
        inside=(bpx[:,0]>=0)&(bpx[:,0]<shape[0])&(bpx[:,1]>=0)&(bpx[:,1]<shape[1])
        keep=np.flatnonzero(inside);bpx=bpx[keep]
        bvalid,sources=mapped_domain(bpx,c,grid,domain)
        # Sorted unique transformed pixels; record source membership for independent verification.
        validbg=[];bgrecords=[]
        for orig,q,dom,source in zip(keep,bpx,bvalid,sources):
            qy,qx=map(int,q);gt=int(target[qy,qx]);good=bool(dom and gt==0)
            bgrecords.append({'native_index':int(orig),'pixel':[qy,qx],'GT':gt,'domain':bool(dom),'sampled_native':source})
            if good:validbg.append((qy,qx))
        validbg=sorted(set(validbg));row['background_visible']=len(bpx);row['backgrounds']=bgrecords
        row['legal_negatives_before_global_conflict']=[list(v) for v in validbg]
        if not validbg:row['status']='NO_NEGATIVE';continue
        row['status']='ELIGIBLE_PRECAP'
        candidates.append({'native_index':idx,'native_y':int(g['y']),'native_x':int(g['x']),
            'positive':[y,x],'negatives':[list(v) for v in validbg]})
    # Cross-polarity collisions invalidate the entire positive group, no resampling or fallback.
    # GT=1 vs GT=0 checks already forbid such collisions for valid data; assert this invariant.
    pos={tuple(g['positive']) for g in candidates}
    for g in candidates:
        if any(tuple(v) in pos for v in g['negatives']):raise RuntimeError('target-polarity error')
    counts['with_negative']=counts['precap']=len(candidates)
    chosen=choose_groups(candidates,seed,attempt,crop)
    chosen_ids={g['native_index'] for g in chosen}
    for row in trace:
        if row['status']=='ELIGIBLE_PRECAP':row['status']='SELECTED' if row['native_index'] in chosen_ids else 'CAP_EXCLUDED'
    counts['selected']=len(chosen)
    negatives=[tuple(p) for g in chosen for p in g['negatives']]
    rec={**c,'seed':seed,'attempt':attempt,'crop':crop,'counts':counts,'targets':trace,'selected':chosen,
        'negative_reuse':len(negatives)-len(set(negatives)),
        'native_rng_unchanged':True,'region_mask_interpolation':'nearest, align_corners=False, same actual grid; no redefinition of native distances'}
    atlas_categories=[]
    if chosen:atlas_categories.append('native_'+'x'.join(map(str,c['native_shape'])))
    if not targets:atlas_categories.append('no_native_targets')
    if counts['visible'] and not chosen:atlas_categories.append('visible_but_empty_aux')
    if any(len(g['negatives'])==1 for g in chosen):atlas_categories.append('one_negative')
    if any(len(g['negatives'])==2 for g in chosen):atlas_categories.append('two_negatives')
    if any(r['status']=='NO_NEGATIVE' for r in trace):atlas_categories.append('zero_negatives')
    if counts['precap']>16:atlas_categories.append('cap')
    if any(r['status']=='POSITIVE_COLLISION' for r in trace):atlas_categories.append('collision')
    if any(x<0 for x in c['bbox_lbs']):atlas_categories.append('padding')
    if c['affine'] is not None and chosen:atlas_categories.append('affine_with_targets')
    for category in atlas_categories:
        if category not in ATLAS and len(ATLAS)<12:
            f=OUT/'atlas_data'/(category+'.npz');f.parent.mkdir(exist_ok=True)
            # Only accepted native batch data, no model or new inference.
            np.savez_compressed(f,image=out['image'].numpy().astype(np.float32),GT=target.astype(np.int8))
            ATLAS[category]={'category':category,'seed':seed,'attempt':attempt,'crop':crop,'image':name,
                'path':str(f),'selected':chosen,'counts':counts,'native_shape':c['native_shape']}
    return rec

def main():
    global MAPPING,ATLAS
    OUT.mkdir(exist_ok=False)
    write_json(OUT/'execution_start.json',{'status':'RUNNING_CPU_ONLY','UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'protocol_sha256':sha(STAGE/'protocol.json'),'model_forwards':0,'GPU_forwards':0,'optimizer_updates':0,
        'evaluation_crop_cap':4000,'prefetch':'One mandatory native Train batch per seed is discarded before audited attempts; separately accounted, no auxiliary evaluation of prefetch.'})
    import torch.optim
    def forbidden(*a,**k):raise RuntimeError('Forbidden model/optimizer/CUDA path in CPU replay')
    # Explicit runtime barriers; no calls are made to any GPU query.
    torch.load=forbidden;torch.nn.Module.cuda=forbidden;torch.cuda.init=forbidden
    torch.optim.Optimizer.step=forbidden
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    binding=json.loads((STAGE/'source_binding.json').read_text())
    for path,expected in binding['immutable_files'].items():
        if sha(path)!=expected:raise RuntimeError('Bound source mismatch '+path)
    for path,expected in json.loads((STAGE/'SOL_CACHE_BINDING.json').read_text()).items():
        if sha(path)!=expected:raise RuntimeError('Bound cache mismatch '+path)
    from test_aux import run_tests
    tests=run_tests();write_json(OUT/'cpu_implementation_tests.json',tests)
    MAPPING=json.loads((OLD/'mapping.json').read_text());ATLAS={}
    summaries=[];fail=None
    batches=(OUT/'replay_batches.jsonl').open('x');traces=(OUT/'selection_trace.jsonl').open('x')
    try:
        for seed in [2027,2028]:
            loader,records,active=build_loader(seed)
            # Native get_dataloaders consumes/discards one Train batch before the scientific first attempt.
            pref=next(loader)
            append(batches,{'seed':seed,'attempt':0,'kind':'OFFICIAL_PREFETCH_DISCARD','ids':list(pref['keys']),
                'input_sha256':digest(pref['data']),'target_sha256':digest(pref['target']),'auxiliary_crops_evaluated':0})
            del pref
            sums=dict(seed=seed,attempts=0,audited_crops=0,visible=0,selected=0,nonempty_batches=0,nonempty_crops=0)
            image_mass={k:0. for k in MAPPING}
            with (HISTORY/f'seed_{seed}'/'attempts.jsonl').open() as history:
                for number in range(1,1001):
                    resources();old=json.loads(next(history));active[0]=number;records.clear()
                    batch=next(loader);ids=list(batch['keys']);ih=digest(batch['data']);th=digest(batch['target'])
                    match=ids==old['ids'] and ih==old['input_sha256'] and th==old['target_sha256'] and old['attempt']==number
                    row={'seed':seed,'attempt':number,'kind':'AUDITED','ids':ids,'input_sha256':ih,'target_sha256':th,
                        'historical_input_sha256':old['input_sha256'],'historical_target_sha256':old['target_sha256'],
                        'historical_ids':old['ids'],'matched':match,'native_rng_unchanged':all(r['native_rng_unchanged'] for r in records)}
                    append(batches,row)
                    if not match:
                        write_json(OUT/'FIRST_INPUT_MISMATCH.json',row)
                        raise RuntimeError('Historical input/target/ID mismatch: seed '+str(seed)+' attempt '+str(number))
                    if len(records)!=2:raise RuntimeError('Expected exactly two sidecar crop records')
                    for rec in records:
                        append(traces,rec);sums['visible']+=rec['counts']['visible'];sums['selected']+=rec['counts']['selected']
                        if rec['counts']['selected']:
                            image_mass[rec['image']]+=.5;sums['nonempty_crops']+=1
                    sums['attempts']+=1;sums['audited_crops']+=2
                    sums['nonempty_batches']+=int(any(r['counts']['selected'] for r in records))
                    if number%10==0:
                        write_json(OUT/'heartbeat.json',{'seed':seed,'attempt':number,**resources()})
                    if number%100==0:print('CPU_REPLAY',seed,number,'visible',sums['visible'],'selected',sums['selected'],resources(),flush=True)
                    del batch
            total=sum(image_mass.values());sums['retained_fraction']=sums['selected']/sums['visible'] if sums['visible'] else 0.
            sums['Neff']=total**2/sum(v*v for v in image_mass.values()) if total else 0.
            sums['gates']={'nonempty_batches':sums['nonempty_batches']>=32,'retained_fraction':sums['retained_fraction']>=.5,'Neff':sums['Neff']>=3}
            sums['pass']=all(sums['gates'].values());summaries.append(sums)
            write_json(OUT/f'seed_{seed}_summary.json',sums)
            if not sums['pass']:raise RuntimeError('Per-seed feasibility gate failed; do not run next seed')
    except BaseException as e:
        fail={'error':repr(e),'traceback':traceback.format_exc()};write_json(OUT/'failure.json',fail)
    finally:
        batches.close();traces.close()
        write_json(OUT/'atlas_selection.json',{'rule':'First qualifying actual crop for each fixed category; no visual outcome selection',
            'cases':list(ATLAS.values()),'all_categories':['native_960x1440','native_1488x2240','native_1536x2304','no_native_targets','visible_but_empty_aux','one_negative','two_negatives','zero_negatives','cap','collision','padding','affine_with_targets']})
        code=Path(__file__).parent
        write_json(OUT/'execution_receipt.json',{'status':'FAILED_OR_INCOMPLETE' if fail else 'COMPLETE_PENDING_INDEPENDENT_REVIEW',
            'executor':'Codex current session (AI engineering execution, not human medical review)',
            'code_sha256':{str(p):sha(p) for p in code.glob('*.py')},'source_sha256':binding['immutable_files'],
            'summaries':summaries,'failure':fail,**resources(False),'model_forwards':0,'GPU_forwards':0,'optimizer_updates':0,
            'Dev_pixels_access':False,'Test_access':False,'medical_review':['0/105','0/26'],
            'prefetch_accounting':'Native warm-up discarded batches are additional stream overhead, zero audited aux crops; replay ledger records them separately.'})
    if fail:raise RuntimeError(fail['error'])

if __name__=='__main__':main()
