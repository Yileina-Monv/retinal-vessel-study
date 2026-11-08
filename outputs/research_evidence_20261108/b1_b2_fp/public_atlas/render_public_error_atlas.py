"""CPU-only public error maps. Reads fixed Dev28 files; never opens RGB or Test.

Run on the authorized project host. Standard output is a ZIP archive encoded
as base64, so the run writes no files on that host. This is not model inference.
"""
import base64, csv, hashlib, io, json, os, zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root=Path(os.environ['RETINA_PROJECT_ROOT'])
stage=root/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022'
analysis=stage/'analysis/fp_error_analysis_v1'
source=json.loads((analysis/'input_manifest.json').read_text())['files']
figs=json.loads((analysis/'figures_manifest.json').read_text())
figs+=json.loads((analysis/'additional_figures_manifest.json').read_text())
expected={(r['filename'],r['method']):r for r in csv.DictReader((analysis/'per_image_metrics.csv').open(encoding='utf-8-sig'))}
used={}; generated=[]; out=io.BytesIO()
font=ImageFont.load_default()

def checked(p):
 b=p.read_bytes(); h=hashlib.sha256(b).hexdigest()
 assert source.get(str(p))==h, 'input fingerprint mismatch: '+p.name
 used[str(p.relative_to(root))]=h
 return b

def rgb_map(pred,gt,fov):
 a=np.zeros((*gt.shape,3),np.uint8)
 a[~fov]=[35,35,35]
 a[pred&gt&fov]=[0,210,80]
 a[~pred&gt&fov]=[50,100,255]
 a[pred&~gt&fov]=[255,85,20]
 return a

with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name in sorted({x['filename'] for x in figs}):
  gt=np.asarray(Image.open(io.BytesIO(checked(root/f'data/maples_handoff_20261008/labels/Vessels/{name}.png'))))>0
  fov=np.asarray(Image.open(io.BytesIO(checked(root/f'experiments/a15_1_maples_dev_evaluation_20261015/dev/{name}/fov.png'))))>0
  maps={}; masks={}
  for m in ['A0','B1','B2']:
   p=(root/f'experiments/t0_tw_core_20261016/runs/seed_2026/T0/evaluation/step_002000/probabilities/{name}.npy') if m=='A0' else stage/f'runs/{m}/evaluation/step_002000/probabilities/{name}.npy'
   prob=np.load(io.BytesIO(checked(p)),allow_pickle=False)
   assert prob.shape==gt.shape and fov.shape==gt.shape
   assert np.isfinite(prob).all()
   pred=prob>=np.float32(.5); masks[m]=pred
   counts=dict(TP=int((pred&gt&fov).sum()),FP=int((pred&~gt&fov).sum()),FN=int((~pred&gt&fov).sum()))
   assert all(v==int(expected[name,m][k]) for k,v in counts.items()), (name,m,counts)
   maps[m]=rgb_map(pred,gt,fov)
  for entry in [x for x in figs if x['filename']==name]:
   bbox=entry.get('bbox'); h,w=gt.shape
   x0,y0,x1,y1=bbox or [0,0,w,h]
   arrays=[maps[m][y0:y1,x0:x1] for m in ['A0','B1','B2']]
   pw=min(768,x1-x0); ph=round((y1-y0)*pw/(x1-x0))
   canvas=Image.new('RGB',(3*pw,ph+84),'white'); draw=ImageDraw.Draw(canvas)
   draw.text((8,5),name+' | seed2026 step2000 threshold0.5 | exact pixels; NO RGB',fill='black',font=font)
   crop_counts={}
   for j,m in enumerate(['A0','B1','B2']):
    g=gt[y0:y1,x0:x1]; f=fov[y0:y1,x0:x1]; p=masks[m][y0:y1,x0:x1]
    cc=dict(TP=int((p&g&f).sum()),FN=int((~p&g&f).sum()),FP=int((p&~g&f).sum()));crop_counts[m]=cc
    label='T0 (A0)' if m=='A0' else m
    draw.text((j*pw+8,24),label+f" | TP {cc['TP']} FN {cc['FN']} FP {cc['FP']}",fill='black',font=font)
    im=Image.fromarray(arrays[j]).resize((pw,ph),Image.Resampling.NEAREST)
    canvas.paste(im,(j*pw,44))
   draw.text((8,ph+52),'Green: TP | Blue: FN | Orange: FP | Black: TN | Gray: outside FOV',fill='black',font=font)
   draw.text((8,ph+66),'GT-relative errors only. No anatomical or connectivity claim. Counts before display resizing.',fill='black',font=font)
   buf=io.BytesIO();canvas.save(buf,format='PNG');b=buf.getvalue()
   target=entry['path'].replace('atlas/','')
   z.writestr(target,b)
   generated.append(dict(path=target,filename=name,bbox=bbox,kind=entry.get('kind'),counts=crop_counts,sha256=hashlib.sha256(b).hexdigest(),size=list(canvas.size)))
 manifest=dict(images=generated,inputs=used,source_figures=len(figs),dev_images=len({x['filename'] for x in generated}),count_checks=84,threshold=.5,seed=2026,step=2000,RGB_read=False,Test_read=False,model_forwards=0)
 z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
print(base64.b64encode(out.getvalue()).decode())
