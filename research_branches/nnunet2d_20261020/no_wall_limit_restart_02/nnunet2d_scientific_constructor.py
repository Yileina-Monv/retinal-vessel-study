"""Shared fresh-Trainer constructor; CPU probes never initialize a network."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def build_fresh_trainer(plans, dataset_json, device, trainer_class=None):
    """Add a runtime-only fresh-start flag without mutating serialized plans."""
    if plans.get("continue_training", False) is not False:
        raise ValueError("Scientific run must be fresh, never continue_training=True")
    runtime_plans = copy.deepcopy(plans)
    runtime_plans["continue_training"] = False
    if trainer_class is None:
        from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs
        trainer_class = nnUNetTrainer_100epochs
    trainer = trainer_class(runtime_plans, "2d", "all", copy.deepcopy(dataset_json), device=device)
    # Trainer.batch_size does not exist until initialize(). Read the plans manager here.
    config = trainer.configuration_manager
    actual = {
        "epochs": trainer.num_epochs,
        "train_iterations": trainer.num_iterations_per_epoch,
        "monitor_iterations": trainer.num_val_iterations_per_epoch,
        "batch_size": int(config.batch_size),
        "patch_size": list(map(int, config.patch_size)),
        "initial_lr": trainer.initial_lr,
        "weight_decay": trainer.weight_decay,
    }
    expected = {
        "epochs": 100, "train_iterations": 250, "monitor_iterations": 50,
        "batch_size": 2, "patch_size": [1024, 1536],
        "initial_lr": 0.01, "weight_decay": 3e-5,
    }
    if actual != expected:
        raise RuntimeError(f"Frozen constructor configuration mismatch: {actual}")
    if trainer.network is not None or trainer.optimizer is not None or trainer.was_initialized:
        raise RuntimeError("Constructor unexpectedly initialized scientific model/optimizer")
    return trainer


def validate_initialized_batch_size(trainer):
    if not trainer.was_initialized or getattr(trainer, "batch_size", None) != 2:
        raise RuntimeError("Initialized scientific Trainer must retain batch_size=2")


def cpu_probe(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    project = Path("${RETINA_PROJECT_ROOT}")
    base = project / "experiments/nnunet2d_branch_20261020"
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "", "nnUNet_compile": "false",
        "nnUNet_raw": str(base / "raw"),
        "nnUNet_preprocessed": str(base / "preprocessed"),
        "nnUNet_results": str(output_dir / "temporary_results"),
        "nnUNet_n_proc_DA": "1",
        "MPLCONFIGDIR": str(output_dir / "matplotlib"),
    })
    import torch
    from unittest.mock import patch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import nnUNetTrainer_100epochs

    plan_path = base / "preprocessed/Dataset501_MAPLES/nnUNetPlans_8GBtarget.json"
    dataset_path = plan_path.parent / "dataset.json"
    original_bytes = plan_path.read_bytes()
    plans = json.loads(original_bytes)
    dataset = json.loads(dataset_path.read_text())
    before = copy.deepcopy(plans)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU constructor probe must not initialize CUDA/model or train")
    with patch.object(torch.cuda, "init", side_effect=forbidden), \
         patch.object(torch.cuda, "set_device", side_effect=forbidden), \
         patch.object(nnUNetTrainer, "initialize", side_effect=forbidden), \
         patch.object(nnUNetTrainer, "run_training", side_effect=forbidden):
        # Confirm the original regression really exists in this installed version.
        missing_flag_reproduced = False
        try:
            nnUNetTrainer_100epochs(copy.deepcopy(plans), "2d", "all", copy.deepcopy(dataset), device=torch.device("cpu"))
        except KeyError as exc:
            if exc.args != ("continue_training",):
                raise
            missing_flag_reproduced = True
        trainer = build_fresh_trainer(plans, dataset, torch.device("cpu"))
        missing_batch_before_initialize = not hasattr(trainer, "batch_size")
        checks = {
            "original_missing_flag_regression_reproduced": missing_flag_reproduced,
            "shared_actual_constructor_passed": trainer.__class__.__name__ == "nnUNetTrainer_100epochs",
            "preinitialize_batch_size_read_from_configuration": trainer.configuration_manager.batch_size == 2,
            "runtime_batch_attribute_not_accessed_early": missing_batch_before_initialize,
            "fresh_runtime_flag_consumed": "continue_training" not in trainer.plans_manager.plans,
            "input_plans_not_mutated": plans == before,
            "serialized_plans_unchanged": plan_path.read_bytes() == original_bytes,
            "runtime_output_isolated": Path(trainer.output_folder).is_relative_to(output_dir),
            "model_not_initialized": trainer.network is None and not trainer.was_initialized,
            "optimizer_not_initialized": trainer.optimizer is None,
            "cuda_not_initialized": not torch.cuda.is_initialized(),
        }
    result = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "scope": "CPU-only shared constructor smoke; not a GPU launch authorization",
        "checks": checks, "plans_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "trainer_class": trainer.__class__.__name__,
        "epochs": trainer.num_epochs, "train_iterations_per_epoch": trainer.num_iterations_per_epoch,
        "internal_monitor_iterations_per_epoch": trainer.num_val_iterations_per_epoch,
        "batch_size_from_configuration": trainer.configuration_manager.batch_size,
        "cuda_initialized": torch.cuda.is_initialized(), "model_initialized": False,
        "optimizer_attempts": 0, "dev_pixels_read": False, "official_test_accessed": False,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CONSTRUCTOR_PROBE_RESULT=" + json.dumps(result), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-probe", action="store_true", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cpu_probe(args.output_dir)
