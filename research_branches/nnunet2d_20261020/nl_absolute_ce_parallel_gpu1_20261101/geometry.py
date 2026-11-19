"""Pure mapping derived from accepted V2 replay; no CPU-environment override or replay entrypoint."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import json
from aux_loss import map_points,inverse_points
ROOT=Path('${RETINA_PROJECT_ROOT}')
OLD=ROOT/'experiments/nnunet2d_branch_20261020/deep_supervision_ablation_20261028/science100_v1/analysis/train_pairing_cpu_v1'
MAPPING=json.loads((OLD/'mapping.json').read_text())
@lru_cache(maxsize=5)
def native(name):
    with np.load(OLD/'cache'/(name+'.npz'),allow_pickle=False) as z:
        return {k:z[k].astype(bool) for k in ['GT','FOV','Omega','near']}

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
    return {'image':name,'candidates':candidates,'counts':counts,'native_rng_unchanged':True}
