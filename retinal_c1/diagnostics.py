"""Supplement the locked HOLD decision with error coverage and auditable visuals."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
from PIL import Image,ImageDraw
from retinal_release.common import ROOT,read,write,digest
from retinal_c1.audit import OUT
from retinal_c1.preflight import CHECKPOINTS,gradients
from retinal_c1.evaluation import geometry,local,with_chain_ownership
from retinal_c1.geometry import chains,select
from retinal_data.prepare import stable_seed
from retinal_m1.data import SkeletonDataset
from retinal_m1.inference import probability
from retinal_m0.model import UNet


def montage(row,item,prob,geo,records,path):
    pairs={p['id']:p for p in select(geo,[0,0,*prob.shape,0,0,0],stable_seed('C1-eval',2026,row['key']),limit=None,margin=0)}
    with Image.open(ROOT/row['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
    label=item['mask'][0].numpy().astype(bool)
    canvas=Image.new('RGB',(1000,888),(20,20,20));draw=ImageDraw.Draw(canvas)
    draw.text((8,3),row['key']+' | original / label / path(yellow)+background(cyan) / probability / prediction',fill='white')
    for i,(conflict,passed) in enumerate(((True,False),(True,True),(False,False),(False,True))):
        found=next((r for r in sorted(records,key=lambda x:x['path_id']) if r['rank_conflict']==conflict and r['path_pass']==passed and r['path_metric_eligible']),None)
        y0=22+i*216;text=f'conflict={conflict} pass={passed}'
        if found is None:draw.text((8,y0),text+' : absent',fill='white');continue
        pair=pairs[found['path_id']];q=pair['pos'];bg=pair['bg'];center=np.rint(q.mean(0)).astype(int)
        top,left=np.clip(center-64,0,np.array(prob.shape)-128)
        raw=rgb[top:top+128,left:left+128].copy();ref=label[top:top+128,left:left+128]
        p=prob[top:top+128,left:left+128];pred=raw.copy();pred[p>=.5]=(pred[p>=.5]*.35+np.array([0,255,0])*.65).astype('uint8')
        marked=Image.fromarray(raw);pen=ImageDraw.Draw(marked)
        for y,x in bg:
            if top<=y<top+128 and left<=x<left+128:pen.point((int(x-left),int(y-top)),fill=(0,255,255))
        pen.line([(int(x-left),int(y-top)) for y,x in q],fill=(255,255,0),width=1)
        pictures=(Image.fromarray(raw),Image.fromarray((ref*255).astype('uint8')).convert('RGB'),marked,Image.fromarray((p*255).astype('uint8')).convert('RGB'),Image.fromarray(pred))
        for col,picture in enumerate(pictures):canvas.paste(picture.resize((192,192),Image.Resampling.NEAREST),(col*200,y0+18))
        draw.text((8,y0),text+f" | path={found['path_id']} inversion={found['rank_inversion']:.4f}",fill='white')
    canvas.save(path)


def main():
    dest=OUT/'diagnostics';dest.mkdir(exist_ok=True)
    definitions=dict(scope='post-preflight explanatory diagnostics; original thresholds and HOLD decision remain unchanged',
                     corridor_exclusion='conservative: exclude path-connectivity metric if dilated corridor includes even one labelled foreground pixel owned by another regular chain; ranking denominator unchanged',
                     outlier='highest M0 per-image conflict; displayed and leave-one-image-out sensitivity only, not removed from primary result',
                     error_coverage='thin pixels missed within distance1 of prediction; fraction on paths having >=16 eligible background pixels',
                     gradients='with respect to logits; sparse vs dense norm ratio is diagnostic, not a parameter-gradient or optimization-dominance claim')
    write(dest/'definitions.json',definitions)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ds=SkeletonDataset('validation',full=True,cache_size=1);indices={r['key']:i for i,r in enumerate(ds.rows)}
    selected=read(OUT/'selected_rows.json')['validation'];config=read(ROOT/'configs/development_v1.json')
    m0=read(OUT/'preflight/M0.json');outlier=max(m0['rows'],key=lambda r:r['local']['rank_conflict_fraction'])['key']
    for name,(folder,sha) in CHECKPOINTS.items():
        checkpoint=ROOT/'runs/prefetch_round1'/folder/'last.pt';assert digest(checkpoint)==sha
        model=UNet(config['widths'],config['group_norm_groups']);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model']);del saved;model.cuda().eval();rows=[]
        original={r['key']:r for r in read(OUT/'preflight'/(name+'.json'))['rows']}
        for i,row in enumerate(selected):
            item=ds[indices[row['key']]];geo=geometry(row['key']);s=item['skeleton'][0].numpy().astype(bool)
            geo=with_chain_ownership(geo,s,item['mask'][0].numpy())
            p,_=probability(model,item['image'],item['fov'][0],batch_size=2);p=p.numpy()
            loc,paths=local(p,geo,row['key'])
            assert abs(loc['rank_inversion']-original[row['key']]['local']['rank_inversion'])<1e-10
            assert loc['rank_conflict_fraction']==original[row['key']]['local']['rank_conflict_fraction']
            thin=item['thin'][0].numpy().astype(bool);pred=p>=.5
            distance=ndi.distance_transform_edt(~pred) if pred.any() else np.full(p.shape,np.inf)
            missed=thin&(distance>1);covered=np.zeros(p.shape,bool)
            pos=geo['pos'][np.diff(geo['bg_offsets'])>=16].reshape(-1,2);covered[tuple(pos.T)]=True
            record=dict(key=row['key'],disease=row['disease'],local=loc,thin_missed=int(missed.sum()),thin_missed_on_eligible_paths=int((missed&covered).sum()),
                        missed_thin_coverage=float((missed&covered).sum()/missed.sum()) if missed.any() else None)
            probpath=dest/(name+'_'+Path(row['key']).stem+'_probability.npy')
            np.save(probpath,p,allow_pickle=False);record['probability_sha256']=digest(probpath)
            write(dest/(name+'_'+Path(row['key']).stem+'.json'),dict(image=record,paths=paths));rows.append(record)
            if i==0 or row['key']==outlier:montage(row,item,p,geo,paths,dest/(name+'_'+Path(row['key']).stem+'_examples.png'))
            print(f'{name} supplementary {i+1}/24',flush=True)
        write(dest/(name+'.json'),dict(checkpoint_sha256=sha,rows=rows,gradients=gradients(model),outlier_key=outlier,definitions=definitions))
        del model;torch.cuda.empty_cache()


if __name__=='__main__':main()
