"""Inspect frozen development checkpoints and real-patch auxiliary gradients."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from retinal_release.common import ROOT,read,write,digest
from retinal_m0.model import UNet
from retinal_m1.data import SkeletonDataset
from retinal_release.data import masked_loss
from retinal_c1.audit import OUT
from retinal_c1.evaluation import evaluate_item,summary,geometry
from retinal_c1.geometry import select
from retinal_c1.objectives import auxiliary
from retinal_data.prepare import stable_seed

CHECKPOINTS={'M0':('m0_seed2026_dev10k_w1','8bce5b8f47c7fe43d553d4467adc46f5387eade014265b7a6a6f4b5211e67d79'),
             'M1':('m1_lambda01_seed2026_dev10k_w1','74b2aab9afc7d14702508914f49deaf63b33b411671a6140a8c78c4dd8eafd56')}


def gradients(model):
    ds=SkeletonDataset('train',cache_size=1)
    keys=read(OUT/'protocol.json')['training_32'][:8]
    indices={r['key']:i for i,r in enumerate(ds.rows)};records=[]
    for key in keys:
        item=ds[(0,indices[key]*4)]
        seed=stable_seed('C1',2026,item['epoch'],item['draw'],key)
        pairs=select(geometry(key),item['geometry'].tolist(),seed)
        for pair in pairs:
            assert item['mask'][0][tuple(pair['pos'].T)].bool().all()
            assert not item['mask'][0][tuple(pair['bg'].T)].bool().any()
            assert item['fov'][0][tuple(pair['bg'].T)].bool().all()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):z=model(item['image'][None].cuda())
        z=z.detach().float().requires_grad_();gs={};losses={}
        for mode in ('PAIR','SHUFFLE','POOL'):
            loss,info=auxiliary(z,[pairs],mode,[seed])
            gs[mode]=torch.autograd.grad(loss,z)[0].flatten();losses[mode]=float(loss.detach())
            if mode=='PAIR':pair_gaps=info['gaps']
        base=masked_loss(z,item['mask'][None].cuda(),item['fov'][None].cuda())
        gbase=torch.autograd.grad(base,z)[0].flatten()
        gp=gs['PAIR'];gn=float(gp.norm())
        records.append(dict(key=key,pairs=len(pairs),loss=losses,finite=bool(all(torch.isfinite(g).all() for g in gs.values())),
                            pair_gradient_norm=gn,weighted_aux_to_base_norm=.1*gn/max(float(gbase.norm()),1e-12),
                            shuffle_relative_gradient_difference=float((gp-gs['SHUFFLE']).norm())/max(gn,1e-12),
                            pool_relative_gradient_difference=float((gp-gs['POOL']).norm())/max(gn,1e-12),pair_gaps=pair_gaps))
    return records


def panels(row,prob,paths,geo,destination):
    # First ID in each predetermined category, rather than the visually strongest example.
    with Image.open(ROOT/row['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
    canvas=Image.new('RGB',(768,4*210),(20,20,20));draw=ImageDraw.Draw(canvas)
    for i,(conflict,passed) in enumerate(((True,False),(True,True),(False,False),(False,True))):
        found=next((r for r in paths if r['rank_conflict']==conflict and r['path_pass']==passed and r['label_path_pass']),None)
        text=f'conflict={conflict} pass={passed}'
        if found is None:draw.text((5,i*210+3),text+' : absent',fill='white');continue
        q=geo['pos'][found['path_id']];center=np.rint(q.mean(0)).astype(int)
        top,left=np.clip(center-64,0,np.array(prob.shape)-128)
        a=rgb[top:top+128,left:left+128].copy();b=a.copy()
        patchprob=prob[top:top+128,left:left+128]
        pred=patchprob>=.5;b[pred]=(b[pred]*.35+np.array([0,255,0])*.65).astype('uint8')
        marked=Image.fromarray(a);md=ImageDraw.Draw(marked)
        md.line([(int(x-left),int(y-top)) for y,x in q],fill=(255,255,0),width=1)
        for col,picture in enumerate((Image.fromarray(a),marked,Image.fromarray(b))):
            canvas.paste(picture.resize((192,192),Image.Resampling.NEAREST),(col*256,i*210+18))
        draw.text((5,i*210+3),text+f" rank={found['rank_inversion']:.3f}",fill='white')
    canvas.save(destination)


def main():
    assert read(OUT/'geometry_audit.json')['status']=='PASS_GEOMETRY'
    dest=OUT/'preflight';dest.mkdir(exist_ok=True)
    definition=dict(primary_gate='M0 macro image fraction of paths with inversion >= .05 must be >= .10; M1 reported as auxiliary persistence check',
                    local_path='8-connected prediction >=.5 within dilation1 of label foreground owned by the 32-point path; endpoint disks radius1; exclude label-disconnected references',
                    limitations='Corridor connectivity is a local label-conditioned diagnostic, not independent clinical topology; no false-connection-rate claim',
                    gradient_gate='all real-patch gradients finite; >=4 of 8 patches have >=2 pairs, nonzero PAIR gradient and relative PAIR/SHUFFLE gradient difference >1e-4')
    if (dest/'definitions.json').exists():assert read(dest/'definitions.json')==definition
    else:write(dest/'definitions.json',definition)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ds=SkeletonDataset('validation',full=True,cache_size=1)
    indices={r['key']:i for i,r in enumerate(ds.rows)};selected=read(OUT/'selected_rows.json')['validation']
    config=read(ROOT/'configs/development_v1.json');all_results={}
    for name,(folder,sha) in CHECKPOINTS.items():
        checkpoint=ROOT/'runs/prefetch_round1'/folder/'last.pt';assert digest(checkpoint)==sha
        model=UNet(config['widths'],config['group_norm_groups'])
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);del saved
        model.cuda().eval();rows=[];start=time.perf_counter()
        for i,row in enumerate(selected):
            path=dest/(name+'_'+Path(row['key']).stem+'.json')
            if path.exists():record=read(path)['image']
            else:
                record,paths,prob=evaluate_item(model,ds[indices[row['key']]])
                record['disease']=row['disease'];write(path,dict(image=record,paths=paths))
                if i==0:panels(row,prob,paths,geometry(row['key']),dest/(name+'_local_examples.png'))
            rows.append(record)
            print(f'{name} validation {i+1}/24 seconds={time.perf_counter()-start:.1f}',flush=True)
        grad=gradients(model)
        result=dict(checkpoint_sha256=sha,rows=rows,summary=summary(rows),gradients=grad)
        write(dest/(name+'.json'),result);all_results[name]=result;del model;torch.cuda.empty_cache()
    primary=all_results['M0']['summary']['local']['rank_conflict_fraction']
    gradients_ok=all(all(r['finite'] for r in v['gradients']) and sum(r['pairs']>=2 and r['pair_gradient_norm']>0 and r['shuffle_relative_gradient_difference']>1e-4 for r in v['gradients'])>=4 for v in all_results.values())
    result=dict(status='PASS_PREFLIGHT' if primary>=.10 and gradients_ok else 'HOLD_PREFLIGHT',
                M0_rank_conflict_fraction=primary,M1_rank_conflict_fraction=all_results['M1']['summary']['local']['rank_conflict_fraction'],
                gradient_gate=gradients_ok,definitions=definition,source_test_images_opened=0)
    write(dest/'decision.json',result);print(result,flush=True)


if __name__=='__main__':main()
