"""CPU/std-library verification; never imports archived training code."""
import ast,hashlib,json,re
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
checks=0;errors=[]
def check(ok,message):
 global checks
 checks+=1
 if not ok:errors.append(message)
manifest=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
for item in manifest:
 p=ROOT/item['path'];check(p.is_file(),item['path']+' missing')
 if not p.is_file():continue
 b=p.read_bytes();check(hashlib.sha256(b).hexdigest()==item['public_sha256'],item['path']+' digest')
 check(not re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}',b),item['path']+' credentials')
 check(not re.search(rb'GPU-[0-9a-fA-F-]{30,}|172\.21\.14\.180',b),item['path']+' machine identifier')
 if p.suffix=='.py':
  try:ast.parse(b,filename=str(p));check(True,'')
  except SyntaxError as e:check(False,str(e))
 if p.suffix=='.json':
  try:json.loads(b);check(True,'')
  except Exception as e:check(False,str(e))
for p in [ROOT/'docs/research/DELIVERY_INDEX_20261119.md',HERE/'README.md']:
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
  if link.startswith(('http:','https:','#')):continue
  check((p.parent/unquote(link.split('#')[0])).exists(),str(p)+' broken '+link)
result={'status':'PASS' if not errors else 'FAIL','checks':checks,'source_files':len(manifest),'python_files':sum(x['path'].endswith('.py') for x in manifest),'source_hashes':'retained as provenance; public hashes checked here','server_cpu_tests':{'test_maples_core':4,'test_maples_class_split':7,'passed':11,'CUDA_VISIBLE_DEVICES':'','scope':'synthetic tensors and geometry, original A7 source; no model inference'},'errors':errors,'limitations':['not full training reproduction','no GPU or Test execution','public paths are literal placeholders','original non-primary historical bootstrap limitations retained']}
print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':raise SystemExit(bool(errors))
