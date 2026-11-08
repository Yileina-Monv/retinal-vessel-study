"""Fixed six-case historical metric-coverage atlas, CPU only, no RGB or inference.
Set RETINA_PROJECT_ROOT; stdout is a base64 ZIP; no server files are written.
"""
import base64,csv,hashlib,io,json,os,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(os.environ['RETINA_PROJECT_ROOT'])
base=root/'experiments/nnunet2d_branch_20261020'
term=base/'deep_supervision_ablation_20261028/science100_v1/analysis/nl_absolute_ce_cpu_v2_alltargets_20261031/science100_v1/terminal_evaluation_20261103'
analysis=term/'analysis'
cases=json.loads((analysis/'metric_review_20261103_atlas_manifest.json').read_text())
src=json.loads((analysis/'historical_near_path_20261103_sources.json').read_text())
fp=root/'experiments/full_fov_thin_skeleton_bce_20261019/target_definition_review_v1/scientific_b1_b2_2000_v1_20261022/analysis/fp_error_analysis_v1/input_manifest.json'
src.update(json.loads(fp.read_text())['files'])
rows=list(csv.DictReader((analysis/'historical_near_path_20261103_per_image.csv').open(encoding='utf-8-sig')))
used={};records=[];checks=0;buf=io.BytesIO()
def checked(p):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest();assert src.get(str(p))==h,p.name
 used[str(p.relative_to(root))]=h;return b
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for c in cases:
  name=c['selection']['filename'];y0,x0,y1,x1=c['crop_native_yx']
  gt=np.asarray(Image.open(io.BytesIO(checked(root/f'data/maples_handoff_20261008/labels/Vessels/{name}.png'))))>0
  fov=np.asarray(Image.open(io.BytesIO(checked(root/f'experiments/a15_1_maples_dev_evaluation_20261015/dev/{name}/fov.png'))))>0
  pw=max(300,min(480,(x1-x0)*2));ph=round((y1-y0)*pw/(x1-x0));step=ph+62
  canvas=Image.new('RGB',(4*pw,2*step+90),'white');draw=ImageDraw.Draw(canvas)
  draw.text((8,5),f"Case {c['case_id']} | {name} | fixed historical selection; no RGB",fill='black',font=font)
  cr=[]
  for row,seed in enumerate([2027,2028]):
   for col,m in enumerate(['T0','TW','N0','NL']):
    if m in ['T0','TW']:p=root/f'experiments/t0_tw_core_20261016/runs/seed_{seed}/{m}/evaluation/step_002000/probabilities/{name}.npy'
    elif m=='N0':p=base/f'replication_seed2027_2028_100epochs_v1_20261026/runs/seed_{seed}/dev_predictions/{name}.npy'
    else:p=term/f'runs/seed_{seed}/dev_predictions/{name}.npy'
    prob=np.load(io.BytesIO(checked(p)),allow_pickle=False);assert prob.shape==gt.shape and np.isfinite(prob).all()
    pred=prob>=np.float32(.5);tp=int((pred&gt&fov).sum());fn=int((~pred&gt&fov).sum());fpn=int((pred&~gt&fov).sum())
    ref=next(r for r in rows if r['filename']==name and int(r['seed'])==seed and r['method']==m)
    assert abs(2*tp/(2*tp+fn+fpn)-float(ref['dice']))<1e-12
    assert abs(fpn/int(fov.sum())-float(ref['fp_density']))<1e-12;checks+=2
    g=gt[y0:y1,x0:x1];f=fov[y0:y1,x0:x1];q=pred[y0:y1,x0:x1]
    a=np.zeros((*g.shape,3),np.uint8);a[~f]=35;a[q&g&f]=[0,210,80];a[~q&g&f]=[50,100,255];a[q&~g&f]=[255,85,20]
    counts=dict(TP=int((q&g&f).sum()),FN=int((~q&g&f).sum()),FP=int((q&~g&f).sum()))
    yy=30+row*step;draw.text((col*pw+5,yy),f'{m}, seed{seed}',fill='black',font=font)
    draw.text((col*pw+5,yy+20),f"TP {counts['TP']} FN {counts['FN']} FP {counts['FP']}",fill='black',font=font)
    canvas.paste(Image.fromarray(a).resize((pw,ph),Image.Resampling.NEAREST),(col*pw,yy+42))
    cr.append(dict(method=m,seed=seed,counts=counts))
  draw.text((8,2*step+38),'Green TP / Blue FN / Orange FP. Entire crop pixel counts, not near-path recall or anatomical truth.',fill='black',font=font)
  draw.text((8,2*step+59),'T0/TW: step2000; N0/NL: epoch100. Threshold0.5. Native counts before nearest-neighbor display resizing.',fill='black',font=font)
  b=io.BytesIO();canvas.save(b,format='PNG');img=b.getvalue();fn=f"case_{c['case_id']:02d}.png";z.writestr(fn,img)
  records.append(dict(path=fn,case_id=c['case_id'],filename=name,crop_native_yx=c['crop_native_yx'],selection_reason=c['selection']['selection_reason'],counts=cr,sha256=hashlib.sha256(img).hexdigest()))
 z.writestr('manifest.json',json.dumps(dict(cases=records,inputs=used,metric_checks=checks,RGB_read=False,Test_read=False,model_forwards=0,scope='six original metric-coverage cases; not representative prevalence'),ensure_ascii=False,indent=2))
print(base64.b64encode(buf.getvalue()).decode())
