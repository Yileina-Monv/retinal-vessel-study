"""CPU-only review against preserved logs/checkpoint. No training/inference entry."""
from __future__ import annotations
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CPU-only audit requires CUDA_VISIBLE_DEVICES=''")
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import torch
from revised_recovery import audit_legacy_checkpoint
from revised_runtime import state_is_finite

BASE = Path("${RETINA_PROJECT_ROOT}")
OLD = BASE / "experiments/nnunet2d_branch_20261020/scientific_seed2026_100epochs_v2_no_wall_limit"
BRANCH = Path(__file__).resolve().parent
FREEZE = BASE / "docs/nnunet2d_seed2026_100epochs_20261020/no_wall_limit_restart_02"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(OLD) or output == OLD:
        raise SystemExit("Audit outputs must stay in a new subdirectory of the old run")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    old_binding = json.loads((OLD / "execution_binding.json").read_text())
    old_branch = BASE / "code/retinal-vessel-study/research_branches/nnunet2d_20261020/no_wall_limit_restart_02"
    protected = [old_branch / name for name in old_binding["implementation_sha256s"]]
    protected += [FREEZE / name for name in ("protocol.json", "source_manifest.json", "freeze_receipt.json")]
    protected += [OLD / name for name in ("budget_ledger.json", "per_update.jsonl", "stop_record.json", "job_state.json")]
    checkpoint_path = OLD / "nnunet_results/Dataset501_MAPLES/nnUNetTrainer_100epochs__nnUNetPlans_8GBtarget__2d/fold_all/checkpoint_interrupted_unresumable.pth"
    protected.append(checkpoint_path)
    before = {str(path): sha(path) for path in protected}
    unit = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", "test_amp_recovery_revision"],
                          cwd=BRANCH, env=os.environ.copy(), capture_output=True, text=True, timeout=180)
    (output / "unit_tests.log").write_text(unit.stdout + unit.stderr)
    if unit.returncode:
        raise SystemExit("CPU regression failed; preserve log, no GPU allowed")
    match = re.search(r"Ran (\d+) tests", unit.stdout + unit.stderr)
    tests = int(match.group(1)) if match else None
    records = [json.loads(line) for line in (OLD / "per_update.jsonl").read_text().splitlines() if line.strip()]
    monitor_count = len((OLD / "internal_train_monitor.jsonl").read_text().splitlines())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = audit_legacy_checkpoint(checkpoint, records, monitor_count)
    report["model_state_all_finite"] = state_is_finite(checkpoint["network_weights"])
    report["optimizer_state_all_finite"] = state_is_finite(checkpoint["optimizer_state"])
    report["grad_scaler_state"] = checkpoint["grad_scaler_state"]
    report["checkpoint_sha256"] = before[str(checkpoint_path)]
    report["original_stop_summary_success_count"] = json.loads((OLD / "stop_record.json").read_text())["successful_updates"]
    report["original_stop_summary_count_error"] = report["original_stop_summary_success_count"] - report["successful_updates"]
    report["original_history_status_unchanged"] = json.loads((OLD / "job_state.json").read_text())["status"]
    report["interpretation"] = "CPU-only reconstructed phase/counts. Not an exact-resume checkpoint conversion or authorization."
    (output / "legacy_checkpoint_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    after = {path: sha(Path(path)) for path in before}
    checks = {
        "CPU_regressions_passed": unit.returncode == 0 and tests is not None,
        "original_code_protocol_logs_checkpoint_unchanged": before == after,
        "actual_attempts_6527": report["attempts"] == 6527,
        "actual_successes_6526": report["successful_updates"] == 6526,
        "one_AMP_skip": report["amp_skips"] == 1,
        "model_and_optimizer_finite": report["model_state_all_finite"] and report["optimizer_state_all_finite"],
        "legacy_epoch_mismatch_identified": not report["saved_epoch_matches_cursor"],
        "correct_epoch26_iteration27": report["reconstructed_cursor"]["epoch_index"] == 26 and report["reconstructed_cursor"]["next_train_iteration"] == 27,
        "remaining223_attempts_in_epoch": report["remaining_attempts_current_epoch"] == 223,
        "strict_queue_resume_not_claimed": report["strict_resume_supported"] is False,
        "CUDA_not_initialized": torch.cuda.is_initialized() is False,
        "no_new_Dev_predictions": not (OLD / "dev_predictions").exists(),
    }
    result = {
        "status": "CPU_REVISION_VERIFIED_GPU_NOT_AUTHORIZED" if all(checks.values()) else "STOP",
        "utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "unit_test_count": tests, "protected_file_count": len(before),
        "protected_sha256_before": before, "protected_sha256_after": after,
        "candidate_code_sha256": {path.name: sha(path) for path in BRANCH.glob("*.py")},
        "retinal_model_instantiated": False, "retinal_model_forward_calls": 0,
        "retinal_optimizer_updates": 0, "GPU_optimizer_updates": 0,
        "validation_note": "Real GradScaler behavior tested only on CPU scalar parameters; no retinal model/data execution.",
        "official_test_accessed": False, "launch_authorized": False,
        "strict_resume_supported": False,
        "pending": ["scientific attempt-versus-success budget approval",
                    "explicit persistent/cumulative AMP pause thresholds",
                    "explicit non-exact continuation versus fresh-run decision",
                    "revision-aware full training-loop integration and new launch gates"],
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "tests": tests, "checks": checks,
                      "verification": str(output / "verification.json"),
                      "GPU_updates": 0, "launch_authorized": False}, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
