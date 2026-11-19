"""Read-only recovery metadata and state restoration; does not open a training ledger."""
import copy
from fast_observer import digest, states_finite, restore_rng

def recovery_manifest(bundle, ledger):
    if bundle["budget"] != ledger or ledger["status"] != "STOPPED" or ledger["pending"] is not None:
        raise RuntimeError("snapshot and stopped budget mismatch")
    for key, expected in bundle["digests"].items():
        if digest(bundle[key]) != expected:
            raise RuntimeError("snapshot digest mismatch: " + key)
    if not states_finite(bundle["model"], bundle["optimizer"]):
        raise RuntimeError("nonfinite checkpoint state")
    completed = ledger["epoch"] * 250 + ledger["iteration"]
    if ledger["fatal"] or ledger["unknown"] != 1 or completed != ledger["success"] + ledger["skip"]:
        raise RuntimeError("unexpected interruption accounting")
    if ledger["attempts"] != completed + 1 or ledger["monitor_complete"] != ledger["epoch"] * 50:
        raise RuntimeError("unexpected logical/physical cursor")
    if bundle["cursor"]["phase"] != "train" or len(bundle["partial_train"]) != ledger["iteration"] or bundle["partial_monitor"]:
        raise RuntimeError("partial epoch evidence mismatch")
    return {"status": "PREPARED_NOT_RELEASED", "logical_completed": completed,
            "next_logical_input": completed + 1, "next_physical_attempt": ledger["attempts"] + 1,
            "charged_attempts_preserved": ledger["attempts"], "unknown_preserved": ledger["unknown"],
            "successful_updates_preserved": ledger["success"], "monitor_completed": ledger["monitor_complete"],
            "remaining_logical_inputs": 25000 - completed, "original_physical_cap": 25000,
            "proposed_physical_cap_needs_approval": 25001, "extra_physical_attempts_requested": 1,
            "proposed_logical_terminal": 25000, "epochs": 100, "training_released": False,
            "epoch": ledger["epoch"], "iteration": ledger["iteration"],
            "partial_epoch_outputs": len(bundle["partial_train"]),
            "restore_scheduler_without_extra_step": True,
            "monitor_historical_tensor_equivalence_proved": False,
            "live_worker_queue_restore_proved": False}

def restore_state(trainer, bundle, include_cuda_rng=False):
    """Restore only state. Caller reconstructs and verifies loaders before restoring main RNG.
    CPU validation intentionally leaves the saved CUDA RNG opaque and unchanged.
    """
    trainer.network.load_state_dict(bundle["model"], strict=True)
    trainer.optimizer.load_state_dict(bundle["optimizer"])
    trainer.grad_scaler.load_state_dict(bundle["scaler"])
    trainer.lr_scheduler.load_state_dict(bundle["scheduler"])
    trainer.current_epoch = bundle["native_current_epoch"]
    trainer.logger.load_checkpoint(bundle["logging"])
    trainer._best_ema = bundle["best_ema"]
    rng = dict(bundle["rng"])
    if not include_cuda_rng:
        rng.pop("cuda0", None)
    restore_rng(rng)
    actual = {"model": trainer.network.state_dict(), "optimizer": trainer.optimizer.state_dict(),
              "scaler": trainer.grad_scaler.state_dict(), "scheduler": trainer.lr_scheduler.state_dict(),
              "logging": trainer.logger.get_checkpoint(), "best_ema": trainer._best_ema}
    for key, value in actual.items():
        if digest(value) != bundle["digests"][key]:
            raise RuntimeError("restored state mismatch: " + key)
    return {"partial_train": copy.deepcopy(bundle["partial_train"]),
            "partial_monitor": copy.deepcopy(bundle["partial_monitor"]),
            "cursor": copy.deepcopy(bundle["cursor"]),
            "CUDA_RNG_restored": include_cuda_rng}
