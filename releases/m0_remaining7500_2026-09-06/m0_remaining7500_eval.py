import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retinal_prefetch.resources import Budget
guard = Budget()
import torch
from retinal_release.common import write
from retinal_prefetch.release import verify
import retinal_m1.evaluate as evaluate

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
guard.cuda()
release = verify()
original_score = evaluate.score

def monitored_score(*args, **kwargs):
    guard.check()
    result = original_score(*args, **kwargs)
    guard.cuda_sample()
    return result

evaluate.score = monitored_score
output = Path(sys.argv[2])
evaluate.main()
write(output.with_name(output.stem + '_resources.json'),
      dict(**guard.finish(), execution_id=release['execution_id']))
