"""Fixed-label local diagnostics; no predicted skeleton defines the targets."""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from retinal_release.common import ROOT, read, digest
from retinal_c1.audit import OUT
from retinal_c1.geometry import select
from retinal_data.prepare import stable_seed
from retinal_m1.metrics import score, aggregate


def geometry(key):
    path=OUT/'geometry'/(Path(key).stem+'.npz')
    assert digest(path)==read(path.with_suffix('.json'))['geometry_sha256']
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def with_chain_ownership(geo,skeleton,label):
    from retinal_c1.geometry import chains
    skeleton=np.asarray(skeleton,bool)
    chain_map=np.full(skeleton.shape,-1,np.int32)
    for cid,points in enumerate(chains(skeleton)[0]):chain_map[tuple(points.T)]=cid
    nearest=ndi.distance_transform_edt(~skeleton,return_distances=False,return_indices=True)
    geo['_chain_owner']=chain_map[tuple(nearest)];geo['_label']=np.asarray(label,bool)
    return geo


def local(probability,geo,key):
    h,w=probability.shape
    pairs=select(geo,[0,0,h,w,0,0,0],stable_seed('C1-eval',2026,key),limit=None,margin=0)
    records=[];disk=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
    for pair in pairs:
        q=pair['pos'];b=pair['bg'];pid=pair['id']
        a=probability[tuple(q.T)];n=probability[tuple(b.T)]
        rank=float(((a[:,None]<n[None,:])+.5*(a[:,None]==n[None,:])).mean())
        owned=geo['fg'][geo['fg_offsets'][pid]:geo['fg_offsets'][pid+1]]
        lo=np.maximum(np.minimum(owned.min(0),q.min(0))-2,0)
        hi=np.minimum(np.maximum(owned.max(0),q.max(0))+3,[h,w])
        region=np.zeros(tuple(hi-lo),bool);region[tuple((owned-lo).T)]=True
        corridor=ndi.binary_dilation(region,structure=disk)
        foreign=0
        if '_chain_owner' in geo:
            owners=geo['_chain_owner'][lo[0]:hi[0],lo[1]:hi[1]]
            reference_label=geo['_label'][lo[0]:hi[0],lo[1]:hi[1]]
            foreign=int((corridor&reference_label&(owners>=0)&(owners!=geo['chain'][pid])).sum())
        pred=(probability[lo[0]:hi[0],lo[1]:hi[1]]>=.5)&corridor
        components,_=ndi.label(pred,structure=np.ones((3,3),bool))
        ends=[]
        for point in (q[0],q[-1]):
            endpoint=np.zeros(region.shape,bool);endpoint[tuple(point-lo)]=True
            endpoint=ndi.binary_dilation(endpoint,structure=disk)
            ends.append(set(components[endpoint].tolist())-{0})
        # The label itself must connect the endpoints inside this frozen corridor.
        reference,_=ndi.label(region,structure=np.ones((3,3),bool))
        gt_pass=reference[tuple(q[0]-lo)]!=0 and reference[tuple(q[0]-lo)]==reference[tuple(q[-1]-lo)]
        longest=current=0
        for missed in a<.5:
            current=current+1 if missed else 0;longest=max(longest,current)
        records.append(dict(path_id=int(pid),rank_inversion=rank,rank_conflict=rank>=.05,
                            path_pass=bool(ends[0]&ends[1]),label_path_pass=bool(gt_pass),
                            foreign_chain_foreground_pixels=foreign,path_metric_eligible=bool(gt_pass and foreign==0),
                            longest_missed_run=int(longest),positive_recall=float((a>=.5).mean()),
                            background_positive_fraction=float((n>=.5).mean())))
    valid=[r for r in records if r['path_metric_eligible']]
    summary=dict(paths=len(records),defined_label_paths=len(valid),
                 foreign_chain_paths_excluded=sum(r['foreign_chain_foreground_pixels']>0 for r in records),
                 rank_inversion=float(np.mean([r['rank_inversion'] for r in records])) if records else None,
                 rank_conflict_fraction=float(np.mean([r['rank_conflict'] for r in records])) if records else None,
                 path_pass_fraction=float(np.mean([r['path_pass'] for r in valid])) if valid else None,
                 longest_missed_run=float(np.mean([r['longest_missed_run'] for r in valid])) if valid else None,
                 background_positive_fraction=float(np.mean([r['background_positive_fraction'] for r in records])) if records else None)
    return summary,records


def summary(rows):
    result=aggregate(rows)
    result['local']={k:float(np.mean([r['local'][k] for r in rows if r['local'][k] is not None])) for k in
                     ('rank_inversion','rank_conflict_fraction','path_pass_fraction','longest_missed_run','background_positive_fraction')}
    return result


def evaluate_item(model,item):
    from retinal_m1.inference import probability
    p,tiles=probability(model,item['image'],item['fov'][0],batch_size=2)
    p=p.numpy();geo=with_chain_ownership(geometry(item['key']),item['skeleton'][0].numpy(),item['mask'][0].numpy())
    ordinary=score(p,item['mask'][0].numpy(),item['fov'][0].numpy(),item['thin'][0].numpy())
    loc,paths=local(p,geo,item['key'])
    return dict(key=item['key'],tiles=tiles,**ordinary,local=loc),paths,p
