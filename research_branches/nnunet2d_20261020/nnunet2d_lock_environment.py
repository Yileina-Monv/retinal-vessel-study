"""Record the separate venv's exact read-only inherited package set."""
import hashlib,json,subprocess
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import torch
ROOT=Path('${RETINA_PROJECT_ROOT}'); ENV=ROOT/'envs/nnunet2d_v2'
EXP=ROOT/'experiments/nnunet2d_branch_20261020'; PY=ENV/'bin/python'
freeze=subprocess.check_output([str(PY),'-m','pip','freeze','--all'],text=True)
lockfile=EXP/'engineering/environment_freeze.txt'
lockfile.write_text(freeze,encoding='utf-8')
check=subprocess.run([str(PY),'-m','pip','check'],text=True,capture_output=True)
if check.returncode: raise SystemExit(check.stdout+check.stderr)
packages=('nnunetv2','torch','torchvision','dynamic-network-architectures','numpy','Pillow','batchgenerators','batchgeneratorsv2','blosc2','acvl-utils')
versions={}
for p in packages:
    try: versions[p]=version(p)
    except PackageNotFoundError: versions[p]='NOT_INSTALLED'
lock=EXP/'engineering/environment_lock.json'
data=json.loads(lock.read_text(encoding='utf-8'))
data['versions']={'python':'3.10.21','torch':versions['torch'],'cuda_build':'12.1',
  'nnunetv2':versions['nnunetv2'],'torchvision':versions['torchvision'],
  'dynamic_network_architectures':versions['dynamic-network-architectures'],
  'numpy':versions['numpy'],'Pillow':versions['Pillow'],
  'batchgenerators':versions['batchgenerators'],'batchgeneratorsv2':versions['batchgeneratorsv2'],
  'blosc2':versions['blosc2'],'acvl_utils':versions['acvl-utils']}
data['exact_package_freeze_path']=str(lockfile)
data['exact_package_freeze_sha256']=hashlib.sha256(lockfile.read_bytes()).hexdigest()
data['exact_package_freeze_lines']=len([x for x in freeze.splitlines() if x.strip()])
data['torch_runtime_version']=torch.__version__
data['torch_runtime_cuda_build']=torch.version.cuda
data['pip_check_exit_code']=check.returncode; data['pip_check_output']=check.stdout.strip() or check.stderr.strip()
lock.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data,indent=2))
