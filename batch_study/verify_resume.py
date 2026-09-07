"""Compare actual independent-process resume against uninterrupted execution."""
from pathlib import Path
import torch
from retinal_m0.state import state_digest
from retinal_batchstudy.run import dump

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'runs/batch_study_v1'
relative=Path('seed_2026/m2_a2_e4/last.pt')
resumed=torch.load(base/'rtx5080'/relative,map_location='cpu',weights_only=True)
reference=torch.load(base/'rtx5080_resume_reference'/relative,map_location='cpu',weights_only=True)
keys=('model','optimizer','scaler','rng','consumed','step','signatures','lr_next')
checks={key:state_digest(resumed[key])==state_digest(reference[key]) for key in keys}
losses=[abs(a['loss']-b['loss']) for a,b in zip(resumed['history'],reference['history'],strict=True)]
result={'status':'passed' if all(checks.values()) and max(losses)==0 else 'failed',
        'checks':checks,'maximum_loss_difference':max(losses),'updates_compared':len(losses),
        'pause_at_draw':256,'resume_to_draw':2048,'scope':'RTX 5080 only, two independent processes vs uninterrupted'}
dump(ROOT/'batch_study/resume_verification_5080.json',result)
print(result)
if result['status']!='passed': raise SystemExit(1)
