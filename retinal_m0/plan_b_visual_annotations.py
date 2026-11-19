"""Add non-destructive target and 1-pixel d1 support markers to frozen 77-point sheets."""
import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path('${RETINA_PROJECT_ROOT}/experiments/plan_b_error_analysis_20261019');fig=ROOT/'figures';backup=fig/'unmarked'
m=json.loads(Path('${RETINA_PROJECT_ROOT}/docs/plan_b_error_analysis_20261019/source_manifest.json').read_text())
for page in range(1,8):
 p=fig/f'points_page_{page:02d}.png'
 if not p.exists():raise FileNotFoundError(p)
 original=backup/p.name
 if original.exists():shutil.copy2(original,p)
 im=Image.open(p).convert('RGB');d=ImageDraw.Draw(im)
 for idx,q in enumerate(m['point_catalog']):
  if idx//12+1!=page:continue
  row=idx%12;y0=148*row+18+64
  for col in range(13):
   x0=116*col+58
   # Outlined circles mark the target and four d1 neighbors without filling the target pixel.
   for dx,dy,r in [(0,0,4),(0,-1,2),(0,1,2),(-1,0,2),(1,0,2)]:
    x=x0+dx;y=y0+dy
    d.ellipse((x-r,y-r,x+r,y+r),outline=(0,0,0),width=3)
    d.ellipse((x-r+1,y-r+1,x+r-1,y+r-1),outline=(255,255,255),width=1)
 im.save(p)
print('marked_all_13_panels',7,'pages',77,'points')
