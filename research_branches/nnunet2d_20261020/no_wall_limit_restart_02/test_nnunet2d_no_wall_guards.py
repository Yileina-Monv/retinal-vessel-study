"""CPU-only regression tests: no model initialization or GPU updates."""
import copy
import json
from pathlib import Path
import unittest
from nnunet2d_scientific_runtime import HardBudget, StopScientificRun, finite_update, lr_poly
from nnunet2d_scientific_supervisor import (
    PROTO, parse_gpu_snapshot, validate_scientific_invariants,
    validate_fresh_authorization, validate_final_training)

GPU0 = "GPU-0"
GPU1 = "GPU-1"


class NoWallLimitTests(unittest.TestCase):
    def test_wall_clock_never_truncates(self):
        for elapsed in (34200, 36000, 86400, 604800):
            HardBudget(24999, elapsed).before_attempt()

    def test_attempt_cap_exact(self):
        b = HardBudget(24999, 86400)
        b.before_attempt()
        self.assertFalse(b.record(finite=True, amp_skipped=False, grad_finite=True))
        with self.assertRaises(StopScientificRun):
            b.before_attempt()

    def test_failure_guards(self):
        for values in ((False, False, True), (True, True, True), (True, False, False)):
            finite, skipped, grad = values
            self.assertTrue(HardBudget().record(finite=finite, amp_skipped=skipped, grad_finite=grad))

    def test_amp_and_nonfinite_telemetry(self):
        self.assertTrue(finite_update(1, 1, 8, 4)["amp_skipped"])
        self.assertFalse(finite_update(float("nan"), 1, 8, 8)["finite_loss"])
        self.assertFalse(finite_update(1, float("inf"), 8, 8)["finite_gradient_norm"])

    def test_gpu1_busy_does_not_block_gpu0(self):
        result = parse_gpu_snapshot(
            "0, GPU-0, 121 MiB, 0 %\n1, GPU-1, 10000 MiB, 100 %",
            "GPU-1, 99, python, 9900 MiB", GPU0)
        self.assertTrue(result["idle"])
        self.assertEqual(result["compute_apps_on_gpu0"], [])

    def test_gpu0_app_blocks_even_low_memory(self):
        result = parse_gpu_snapshot("0, GPU-0, 20 MiB, 0 %",
                                    "GPU-0, 88, python, 5 MiB", GPU0)
        self.assertFalse(result["idle"])

    def test_gpu0_memory_and_utilization_gate(self):
        self.assertFalse(parse_gpu_snapshot("0, GPU-0, 600 MiB, 0 %", "", GPU0)["idle"])
        self.assertFalse(parse_gpu_snapshot("0, GPU-0, 20 MiB, 1 %", "", GPU0)["idle"])

    def test_uuid_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            parse_gpu_snapshot("0, GPU-1, 0 MiB, 0 %", "", GPU0)

    def test_recipe_unchanged(self):
        parent = json.loads(Path(PROTO["operational_amendment"]["parent_protocol"]).read_text())
        validate_scientific_invariants(parent, PROTO)

    def test_scientific_drift_rejected(self):
        parent = json.loads(Path(PROTO["operational_amendment"]["parent_protocol"]).read_text())
        for section in ("recipe", "data", "evaluation", "randomness"):
            changed = copy.deepcopy(PROTO)
            changed[section]["unexpected_change"] = True
            with self.assertRaises(RuntimeError):
                validate_scientific_invariants(parent, changed)

    def test_scope_and_cap_drift_rejected(self):
        parent = json.loads(Path(PROTO["operational_amendment"]["parent_protocol"]).read_text())
        for field, value in (("epochs", 101), ("seed", 2027), ("planned_optimizer_attempts", 25001)):
            changed = copy.deepcopy(PROTO)
            changed["scope"][field] = value
            with self.assertRaises(RuntimeError):
                validate_scientific_invariants(parent, changed)
        changed = copy.deepcopy(PROTO)
        changed["budget"]["cumulative_gpu_wall_cap_seconds"] = 36000
        with self.assertRaises(RuntimeError):
            validate_scientific_invariants(parent, changed)

    def auth(self):
        return {"status": "USER_APPROVED_FRESH_NO_WALL_LIMIT_RESTART",
                "seed": 2026, "epochs": 100, "attempt_cap": 25000,
                "wall_clock_hard_cap_seconds": None, "prior_successful_updates": 916,
                "preflight_sha256": "pre", "execution_binding_sha256": "binding",
                "prior_budget_ledger_sha256": "prior", "automatic_restart": False}

    def test_exact_authorization(self):
        validate_fresh_authorization(self.auth(), "pre", "binding", "prior")

    def test_stale_or_expanded_authorization_rejected(self):
        for field, value in (("prior_successful_updates", 0), ("preflight_sha256", "stale"),
                             ("seed", 2027), ("automatic_restart", True)):
            changed = self.auth()
            changed[field] = value
            with self.assertRaises(RuntimeError):
                validate_fresh_authorization(changed, "pre", "binding", "prior")

    def test_terminal_gate(self):
        training = {"status": "TRAINING_COMPLETE", "attempts": 25000,
                    "successful_updates": 25000, "internal_train_monitor_batches": 5000}
        ledger = {"optimizer_attempts": 25000, "successful_updates": 25000,
                  "internal_train_monitor_batches": 5000}
        validate_final_training(training, ledger)
        for key in ("attempts", "successful_updates", "internal_train_monitor_batches"):
            changed = dict(training)
            changed[key] -= 1
            with self.assertRaises(RuntimeError):
                validate_final_training(changed, ledger)

    def test_lr_horizon_unchanged(self):
        self.assertEqual(lr_poly(.01, 0), .01)
        self.assertGreater(lr_poly(.01, 99), 0)
        with self.assertRaises(ValueError):
            lr_poly(.01, 100)


if __name__ == "__main__":
    unittest.main()
