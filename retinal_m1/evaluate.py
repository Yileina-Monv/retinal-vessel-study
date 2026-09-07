"""Development validation only; never reads official test or external images."""
import argparse
import torch
from retinal_release.common import ROOT,read,write,digest
from retinal_release.exchange import check_ticket
from retinal_m0.model import UNet
from retinal_m1.verify import verify
from retinal_m1.data import SkeletonDataset
from retinal_m1.inference import probability
from retinal_m1.metrics import score,aggregate

def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');p.add_argument('output');p.add_argument('--threshold',type=float,default=.5);a=p.parse_args()
    from pathlib import Path
    folder=Path(a.run_dir);output=Path(a.output)
    if output.exists():raise ValueError('Never overwrite evaluation results')
    if (folder/'running.lock').exists():raise ValueError('Evaluate an exported or stopped checkpoint')
    release=verify();ticket=read(folder/'ticket.json');check_ticket(ticket);progress=read(folder/'progress.json')
    if ticket.get('addon_id')!=release['addon_id']:raise ValueError('Wrong addon')
    if digest(folder/'last.pt')!=progress['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    if saved['ticket_id']!=ticket['ticket_id'] or saved['step']!=progress['step']:raise ValueError('Checkpoint/ticket mismatch')
    config=read(ROOT/'configs/development_v1.json');model=UNet(config['widths'],config['group_norm_groups'])
    model.load_state_dict(saved['model']);del saved;model.cuda().eval()
    ds=SkeletonDataset('validation',full=True,cache_size=1);rows=[]
    for i in range(len(ds)):
        item=ds[i];prob,tiles=probability(model,item['image'],item['fov'][0],batch_size=8)
        result=score(prob.numpy(),item['mask'][0].numpy(),item['fov'][0].numpy(),item['thin'][0].numpy(),threshold=a.threshold)
        rows.append(dict(key=item['key'],disease=item['disease'],tiles=tiles,**result))
        if (i+1)%10==0:print(f'validation {i+1}/{len(ds)}',flush=True)
    write(output,dict(ticket=ticket,checkpoint_sha256=progress['checkpoint_sha256'],step=progress['step'],threshold=a.threshold,
        scope='development_validation_only_not_final_performance',rows=rows,summary=aggregate(rows)))

if __name__=='__main__':main()
