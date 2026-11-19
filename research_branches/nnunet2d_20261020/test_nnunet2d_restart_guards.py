"""CPU-only guards for a single explicitly approved zero-update restart."""
import copy
import unittest
from nnunet2d_scientific_supervisor import validate_zero_update_restart, watchdog_stop_threshold


class RestartGuards(unittest.TestCase):
    def setUp(self):
        self.prior = {"stages": [{"status": "INCOMPLETE", "exit_code": 1}],
                      "optimizer_attempts": 0, "successful_updates": 0,
                      "cumulative_gpu_process_seconds": 30.037703668989707}
        self.auth = {"status": "USER_APPROVED_SINGLE_ZERO_UPDATE_RESTART",
                     "seed": 2026, "epochs": 100, "optimizer_attempt_cap": 25000,
                     "gpu_process_cap_seconds": 36000,
                     "prior_budget_ledger_sha256": "ledger",
                     "preflight_sha256": "preflight", "execution_binding_sha256": "binding",
                     "prior_gpu_process_seconds": self.prior["cumulative_gpu_process_seconds"]}
        self.kw = {"startup_failure": {"model_initialized": False, "optimizer_attempts": 0},
                   "ledger_sha": "ledger", "preflight_sha": "preflight", "binding_sha": "binding"}

    def test_explicit_zero_update_restart_passes_without_mutating_ledger(self):
        before = copy.deepcopy(self.prior)
        validate_zero_update_restart(self.auth, self.prior, **self.kw)
        self.assertEqual(self.prior, before)

    def test_reject_updated_model(self):
        self.prior["optimizer_attempts"] = 1
        with self.assertRaises(RuntimeError):
            validate_zero_update_restart(self.auth, self.prior, **self.kw)

    def test_reject_second_retry(self):
        self.prior["stages"].append({"status": "INCOMPLETE", "exit_code": 1})
        with self.assertRaises(RuntimeError):
            validate_zero_update_restart(self.auth, self.prior, **self.kw)

    def test_reject_budget_reset_or_changed_receipt(self):
        self.auth["prior_gpu_process_seconds"] = 0
        with self.assertRaises(RuntimeError):
            validate_zero_update_restart(self.auth, self.prior, **self.kw)
        self.auth["prior_gpu_process_seconds"] = self.prior["cumulative_gpu_process_seconds"]
        self.kw["binding_sha"] = "different"
        with self.assertRaises(RuntimeError):
            validate_zero_update_restart(self.auth, self.prior, **self.kw)

    def test_reject_recipe_change(self):
        self.auth["epochs"] = 101
        with self.assertRaises(RuntimeError):
            validate_zero_update_restart(self.auth, self.prior, **self.kw)

    def test_watchdog_reserves_cleanup_within_total_budget(self):
        remaining = 36000 - self.prior["cumulative_gpu_process_seconds"]
        self.assertEqual(watchdog_stop_threshold(remaining), remaining - 120)
        with self.assertRaises(RuntimeError):
            watchdog_stop_threshold(120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
