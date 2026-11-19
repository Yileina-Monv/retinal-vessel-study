"""Honest checkpoint metadata and CPU audits; no native worker-queue resume claim."""
from __future__ import annotations
import copy
import random
import numpy as np
import torch
from revised_policy import RevisionStop, TrainingCursor


def capture_rng_state(include_cuda=False):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state().clone(), "cuda": None}
    if include_cuda:
        if not torch.cuda.is_initialized():
            raise RevisionStop("CUDA RNG capture requested before context exists")
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_main_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_initialized():
            raise RevisionStop("Refuse implicit CUDA initialization during RNG restore")
        torch.cuda.set_rng_state_all(state["cuda"])


def build_revision_bundle(native_checkpoint, ledger, cursor, rng_state,
                          scheduler_state, partial_epoch_aggregates):
    """Not a vanilla nnU-Net checkpoint: generic -c must not silently load it."""
    cursor.validate()
    ledger.validate()
    if cursor.total_attempts != ledger.attempts:
        raise RevisionStop("Cannot snapshot disagreeing cursor/counters")
    checkpoint = copy.copy(native_checkpoint)
    checkpoint["current_epoch"] = cursor.epoch_index  # not unconditional +1
    metadata = {
        "schema": "amp_recovery_cpu_candidate_v1",
        "cursor": cursor.snapshot(), "ledger": ledger.snapshot(),
        "rng_state": rng_state, "scheduler_state": scheduler_state,
        "partial_epoch_aggregates": partial_epoch_aggregates,
        "requires_revision_aware_training_loop": True,
        "native_augmentation_queues_saved": False,
        "strict_native_queue_resume_supported": False,
        "GPU_resume_authorized": False,
    }
    return {"native_checkpoint": checkpoint, "revision_state": metadata}


def require_strict_native_resume(bundle):
    # Main RNG/scheduler/cursor alone are not sufficient to restore producer queues.
    raise RevisionStop("STRICT_RESUME_UNSUPPORTED: native workers/queues/in-flight batches not serialized")


def audit_legacy_checkpoint(checkpoint, records, monitor_count):
    if [row["attempt"] for row in records] != list(range(1, len(records) + 1)):
        raise RevisionStop("Non-contiguous legacy attempt log")
    attempts = len(records)
    successes = sum(row.get("successful_update") is True for row in records)
    skips = sum(row.get("amp_skipped") is True for row in records)
    epoch, iteration = divmod(attempts, 250)
    if iteration == 0:
        raise RevisionStop("Boundary checkpoint requires explicit monitor/finalization audit")
    if monitor_count != epoch * 50:
        raise RevisionStop("Legacy monitor counts do not match completed epochs")
    cursor = TrainingCursor(100, 250, 50, epoch_index=epoch,
                            next_train_iteration=iteration)
    cursor.validate()
    return {
        "attempts": attempts, "successful_updates": successes, "amp_skips": skips,
        "reconstructed_cursor": cursor.snapshot(),
        "native_saved_epoch": checkpoint["current_epoch"],
        "saved_epoch_matches_cursor": checkpoint["current_epoch"] == epoch,
        "remaining_attempts_current_epoch": 250 - iteration,
        "remaining_total_attempts_if_fixed_25000": 25000 - attempts,
        "native_checkpoint_keys": list(checkpoint),
        "main_rng_state_present": any("rng" in key.lower() for key in checkpoint),
        "scheduler_state_present": any("scheduler" in key.lower() for key in checkpoint),
        "native_augmentation_queue_state_present": False,
        "strict_resume_supported": False,
        "CPU_audit_only": True, "GPU_resume_authorized": False,
        "warning": "Counts/phase reconstructed; missing historical RNG/queues cannot be recovered retroactively.",
    }
