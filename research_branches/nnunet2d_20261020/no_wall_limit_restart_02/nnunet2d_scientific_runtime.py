from __future__ import annotations
import hashlib, json, math, os, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs

ROOT = Path('${RETINA_PROJECT_ROOT}/experiments/nnunet2d_branch_20261020/scientific_seed2026_100epochs_v2_no_wall_limit')
PROTO = Path('${RETINA_PROJECT_ROOT}/docs/nnunet2d_seed2026_100epochs_20261020/no_wall_limit_restart_02/protocol.json')
CAP_ATTEMPTS = 25000
CAP_SECONDS = None
TRAIN_STOP_SECONDS = None  # user-approved: complete fixed epochs, no wall-clock truncation
TRAIN_WORKER_SEED = 2026
MONITOR_WORKER_SEED = 2027

class StopScientificRun(RuntimeError):
    pass

class HardBudget:
    def __init__(self, attempts: int = 0, elapsed_seconds: float = 0.0):
        self.attempts = int(attempts)
        self.elapsed_seconds = float(elapsed_seconds)
    def before_attempt(self):
        if self.attempts >= CAP_ATTEMPTS:
            raise StopScientificRun('optimizer-attempt cap reached before next attempt')
        # Wall time is recorded, never used to truncate normal training.
    def record(self, *, finite: bool, amp_skipped: bool, grad_finite: bool):
        self.attempts += 1
        return not (finite and grad_finite and not amp_skipped)

def finite_update(loss: float, grad_norm: float, scale_before: float, scale_after: float) -> dict:
    return {'finite_loss': math.isfinite(float(loss)),
            'finite_gradient_norm': math.isfinite(float(grad_norm)),
            'amp_skipped': bool(scale_after < scale_before)}

def tensor_sha256(value: Any) -> str:
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, (list, tuple)):
            for x in v: add(x)
        else:
            a = v.detach().cpu().contiguous().numpy() if torch.is_tensor(v) else np.ascontiguousarray(v)
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(memoryview(a).cast('B'))
    add(value)
    return h.hexdigest()

def set_all_seeds(seed: int = 2026):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def install_seeded_worker_adapter():
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_module
    from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
    allowed = int(get_allowed_n_proc_DA())
    if allowed != 1:
        raise RuntimeError(f'configured nnU-Net augmentation worker count must be 1, got {allowed}')
    original = trainer_module.NonDetMultiThreadedAugmenter
    import batchgenerators.dataloading.nondet_multi_threaded_augmenter as augmenter_module
    original_producer = augmenter_module.producer
    def seeded_producer(queue, data_loader, transform, thread_id, seed, abort_event, pause_event=None, wait_time=0.02):
        if seed is not None:
            random.seed(int(seed))
        return original_producer(queue, data_loader, transform, thread_id, seed, abort_event, pause_event, wait_time)
    augmenter_module.producer = seeded_producer
    used = []
    def seeded_augmenter(*args, **kwargs):
        if len(used) >= 2:
            raise RuntimeError('unexpected extra augmentation worker pool')
        n = int(kwargs.get('num_processes', args[2] if len(args) > 2 else 1))
        if n != 1:
            raise RuntimeError(f'augmentation pool requested {n} workers, expected exactly 1')
        seed = TRAIN_WORKER_SEED if len(used) == 0 else MONITOR_WORKER_SEED
        kwargs['num_processes'] = 1
        kwargs['seeds'] = [seed]
        used.append({'pool': 'train' if len(used) == 0 else 'internal_train_monitor', 'worker_count': 1, 'seeds': [seed]})
        return original(*args, **kwargs)
    trainer_module.NonDetMultiThreadedAugmenter = seeded_augmenter
    return used

def install_step_instrumentation(trainer, run_dir: Path, budget: HardBudget, started_monotonic: float):
    """Wrap instance methods; keep the official nnUNetTrainer_100epochs class identity."""
    trainer.scientific_run_dir = Path(run_dir)
    trainer.scientific_budget = budget
    trainer.scientific_started_monotonic = float(started_monotonic)
    trainer.attempt_log_path = trainer.scientific_run_dir / 'per_update.jsonl'
    trainer.monitor_log_path = trainer.scientific_run_dir / 'internal_train_monitor.jsonl'
    trainer._monitor_index = 0
    original_train_step = trainer.train_step
    original_validation_step = trainer.validation_step
    def train_step(batch: dict) -> dict:
        elapsed = time.monotonic() - trainer.scientific_started_monotonic
        trainer.scientific_budget.elapsed_seconds = elapsed
        trainer.scientific_budget.before_attempt()
        ids = batch.get('keys', batch.get('case_ids', []))
        data_hash = tensor_sha256(batch['data'])
        target_hash = tensor_sha256(batch['target'])
        scaler = trainer.grad_scaler
        scale_before = float(scaler.get_scale()) if scaler is not None else 1.0
        captured = {'grad_norm': float('nan')}
        original_clip = torch.nn.utils.clip_grad_norm_
        def capture_clip(parameters, max_norm, *args, **kwargs):
            result = original_clip(parameters, max_norm, *args, **kwargs)
            captured['grad_norm'] = float(result.detach().float().cpu().item() if torch.is_tensor(result) else result)
            return result
        torch.nn.utils.clip_grad_norm_ = capture_clip
        try:
            result = original_train_step(batch)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        loss = float(np.asarray(result['loss']).reshape(-1)[0])
        scale_after = float(scaler.get_scale()) if scaler is not None else 1.0
        state = finite_update(loss, captured['grad_norm'], scale_before, scale_after)
        failed = trainer.scientific_budget.record(finite=state['finite_loss'], amp_skipped=state['amp_skipped'], grad_finite=state['finite_gradient_norm'])
        event = {
            'attempt': trainer.scientific_budget.attempts,
            'epoch_zero_based': (trainer.scientific_budget.attempts - 1) // int(trainer.num_iterations_per_epoch),
            'iteration_zero_based': (trainer.scientific_budget.attempts - 1) % int(trainer.num_iterations_per_epoch),
            'train_case_ids': [str(x) for x in ids],
            'input_sha256': data_hash, 'target_sha256': target_hash,
            'lr': float(trainer.optimizer.param_groups[0]['lr']), 'loss': loss if state['finite_loss'] else str(loss),
            'preclip_gradient_norm': captured['grad_norm'] if state['finite_gradient_norm'] else str(captured['grad_norm']),
            'amp_scale_before': scale_before, 'amp_scale_after': scale_after,
            'amp_skipped': state['amp_skipped'], 'successful_update': not failed,
            'elapsed_gpu_process_seconds': elapsed,
            'cuda_memory_allocated_bytes': int(torch.cuda.memory_allocated(trainer.device)),
            'cuda_memory_reserved_bytes': int(torch.cuda.memory_reserved(trainer.device)),
        }
        with open(trainer.attempt_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n'); f.flush()
        if failed:
            raise StopScientificRun(f'failed update telemetry at attempt {trainer.scientific_budget.attempts}')
        return result
    def validation_step(batch: dict) -> dict:
        result = original_validation_step(batch)
        loss = float(np.asarray(result['loss']).reshape(-1)[0])
        if not math.isfinite(loss):
            raise StopScientificRun('nonfinite internal Train-only validation loss')
        ids = batch.get('keys', batch.get('case_ids', []))
        event = {'validation_batch': trainer._monitor_index + 1,
                 'epoch_zero_based': int(trainer.current_epoch),
                 'train_case_ids': [str(x) for x in ids], 'loss': loss,
                 'optimizer_calls': 0, 'source': 'fold=all internal Train only'}
        with open(trainer.monitor_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n'); f.flush()
        trainer._monitor_index += 1
        return result
    trainer.train_step = train_step
    trainer.validation_step = validation_step

def lr_poly(initial_lr: float, step: int, max_steps: int = 100, exponent: float = 0.9) -> float:
    if not 0 <= step < max_steps: raise ValueError('step outside frozen 0..99 LR horizon')
    return float(initial_lr * (1.0 - step / max_steps) ** exponent)
