"""CPU scalar probes and metadata tests only; no retinal model/data/GPU."""
import copy
import json
import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from revised_policy import (AmpLimits, AttemptLedger, StepEvidence, TrainingCursor,
                            RevisionStop, classify_step, require_gpu_execution_authorization)
from revised_recovery import (capture_rng_state, restore_main_rng_state,
                              build_revision_bundle, require_strict_native_resume,
                              audit_legacy_checkpoint)
from revised_runtime import observe_train_step, state_is_finite


def evidence(**changes):
    values = dict(loss=.2, gradient_norm=.4, scale_before=8., scale_after=8.,
                  optimizer_step_calls=1, model_state_finite=True,
                  optimizer_state_finite=True)
    values.update(changes)
    return StepEvidence(**values)


def skip(**changes):
    values = dict(gradient_norm=float("inf"), scale_after=4., optimizer_step_calls=0)
    values.update(changes)
    return evidence(**values)


def ledger(cap=25000, consecutive=3, total=5):
    # Test-only thresholds; not the proposed/frozen scientific thresholds.
    return AttemptLedger(cap, AmpLimits(consecutive, total))


class PolicyTests(unittest.TestCase):
    def test_finite_success(self):
        self.assertEqual(classify_step(evidence())[0], "SUCCESS")

    def test_growth_is_success(self):
        self.assertEqual(classify_step(evidence(scale_after=16.))[0], "SUCCESS")

    def test_mitigated_overflow_is_skip(self):
        self.assertEqual(classify_step(skip())[0], "AMP_SKIP")

    def test_nonfinite_forward_loss_fatal(self):
        self.assertEqual(classify_step(skip(loss=float("nan")))[0], "FATAL")

    def test_corrupted_parameter_state_fatal(self):
        self.assertEqual(classify_step(skip(model_state_finite=False))[0], "FATAL")

    def test_corrupted_optimizer_state_fatal(self):
        self.assertEqual(classify_step(skip(optimizer_state_finite=False))[0], "FATAL")

    def test_step_with_backoff_is_inconsistent(self):
        self.assertEqual(classify_step(skip(optimizer_step_calls=1))[0], "FATAL")

    def test_no_step_without_backoff_is_fatal(self):
        self.assertEqual(classify_step(evidence(optimizer_step_calls=0))[0], "FATAL")

    def test_unhandled_nonfinite_gradient_fatal(self):
        self.assertEqual(classify_step(evidence(gradient_norm=float("inf")))[0], "FATAL")

    def test_multiple_optimizer_calls_fatal(self):
        self.assertEqual(classify_step(evidence(optimizer_step_calls=2))[0], "FATAL")

    def test_invalid_scaler_values_fatal(self):
        for value in (0., -1., float("inf"), float("nan")):
            self.assertEqual(classify_step(skip(scale_after=value))[0], "FATAL")

    def test_explicit_thresholds_required(self):
        with self.assertRaises(TypeError):
            AmpLimits()
        for values in ((0, 1), (1, 0), (1.5, 2)):
            with self.assertRaises(ValueError):
                AmpLimits(*values)
        with self.assertRaises(ValueError):
            AttemptLedger(10, None)

    def test_attempt_success_skip_separated(self):
        b = ledger()
        b.record(evidence())
        b.record(skip())
        self.assertEqual((b.attempts, b.successful_updates, b.amp_skips,
                          b.optimizer_step_calls), (2, 1, 1, 1))
        self.assertIsNone(b.stopped_reason)

    def test_success_resets_consecutive_skips(self):
        b = ledger()
        b.record(skip())
        b.record(evidence())
        self.assertEqual(b.consecutive_skips, 0)

    def test_persistent_overflow_stops(self):
        b = ledger(consecutive=2)
        b.record(skip())
        b.record(skip())
        self.assertEqual(b.stopped_reason, "consecutive_amp_skip_limit")
        with self.assertRaises(RevisionStop):
            b.before_attempt()

    def test_total_skips_stop_even_nonconsecutive(self):
        b = ledger(consecutive=4, total=2)
        b.record(skip())
        b.record(evidence())
        b.record(skip())
        self.assertEqual(b.stopped_reason, "total_amp_skip_limit")

    def test_no_extra_update_to_replace_skip(self):
        b = ledger(cap=3)
        b.record(evidence())
        b.record(skip())
        b.record(evidence())
        self.assertTrue(b.fixed_attempts_complete())
        self.assertEqual(b.successful_updates, 2)
        with self.assertRaises(RevisionStop):
            b.record(evidence())

    def test_fatal_attempt_not_success(self):
        b = ledger()
        b.record(evidence(loss=float("nan")))
        self.assertEqual((b.attempts, b.successful_updates, b.fatal_attempts,
                          b.optimizer_step_calls), (1, 0, 1, 1))

    def test_exception_counts_and_stops(self):
        b = ledger()
        b.record_exception("OOM", 0)
        self.assertEqual((b.attempts, b.fatal_attempts, b.successful_updates), (1, 1, 0))
        with self.assertRaises(RevisionStop):
            b.before_attempt()

    def test_corrupted_counter_snapshot_rejected(self):
        with self.assertRaises(ValueError):
            AttemptLedger(10, AmpLimits(2, 3), attempts=1, successful_updates=2)

    def test_gpu_execution_always_blocked(self):
        with self.assertRaises(RevisionStop):
            require_gpu_execution_authorization()


class CursorTests(unittest.TestCase):
    def test_legacy_mid_epoch_position(self):
        c = TrainingCursor(100, 250, 50, epoch_index=26, next_train_iteration=27)
        c.validate()
        self.assertEqual(c.total_attempts, 6527)
        self.assertEqual(c.total_monitors, 1300)

    def test_train_to_monitor_boundary(self):
        c = TrainingCursor(100, 250, 50, next_train_iteration=249)
        c.record_train_attempt()
        self.assertEqual((c.phase, c.epoch_index, c.next_train_iteration), ("monitor", 0, 250))

    def test_monitor_to_finalize_boundary(self):
        c = TrainingCursor(100, 250, 50, next_train_iteration=250,
                           completed_monitor_batches=49, phase="monitor")
        c.record_monitor_batch()
        self.assertEqual(c.phase, "epoch_finalize")
        self.assertEqual(c.epoch_index, 0)

    def test_cannot_skip_unfinished_epoch(self):
        c = TrainingCursor(100, 250, 50, epoch_index=26, next_train_iteration=27)
        with self.assertRaises(RevisionStop):
            c.finalize_epoch()

    def test_finalize_advances_exactly_one_epoch(self):
        c = TrainingCursor(100, 250, 50, epoch_index=26, next_train_iteration=250,
                           completed_monitor_batches=50, phase="epoch_finalize")
        c.finalize_epoch()
        self.assertEqual((c.epoch_index, c.next_train_iteration, c.phase), (27, 0, "train"))
        self.assertEqual(c.total_attempts, 6750)

    def test_terminal_cursor(self):
        c = TrainingCursor(100, 250, 50, epoch_index=99, next_train_iteration=250,
                           completed_monitor_batches=50, phase="epoch_finalize")
        c.finalize_epoch()
        self.assertEqual((c.phase, c.total_attempts, c.total_monitors), ("complete", 25000, 5000))

    def test_wrong_phase_training_rejected(self):
        c = TrainingCursor(100, 250, 50, next_train_iteration=250, phase="monitor")
        with self.assertRaises(RevisionStop):
            c.record_train_attempt()


class RecoveryTests(unittest.TestCase):
    def test_main_rng_roundtrip(self):
        original = capture_rng_state()
        try:
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            state = capture_rng_state()
            a = (random.random(), np.random.random(), torch.rand(3))
            restore_main_rng_state(state)
            b = (random.random(), np.random.random(), torch.rand(3))
            self.assertEqual(a[:2], b[:2])
            self.assertTrue(torch.equal(a[2], b[2]))
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            restore_main_rng_state(original)

    def test_snapshot_does_not_increment_mid_epoch(self):
        native = {"current_epoch": 27, "network_weights": {"scalar": torch.tensor(1.)}}
        b = ledger()
        b.attempts = 6527; b.successful_updates = 6526; b.amp_skips = 1
        b.optimizer_step_calls = 6526; b.consecutive_skips = 1
        c = TrainingCursor(100, 250, 50, epoch_index=26, next_train_iteration=27)
        bundle = build_revision_bundle(native, b, c, capture_rng_state(),
                                       {"last_epoch": 26}, {"train_batches": 27})
        self.assertEqual(native["current_epoch"], 27)
        self.assertEqual(bundle["native_checkpoint"]["current_epoch"], 26)
        self.assertEqual(bundle["revision_state"]["cursor"]["next_train_iteration"], 27)
        self.assertEqual(bundle["revision_state"]["ledger"]["successful_updates"], 6526)
        self.assertNotIn("network_weights", bundle)  # not a vanilla-loadable file
        self.assertFalse(bundle["revision_state"]["GPU_resume_authorized"])

    def test_strict_native_queue_resume_rejected(self):
        with self.assertRaises(RevisionStop):
            require_strict_native_resume({"revision_state": {"rng_state": "present"}})

    def test_legacy_audit_counts_and_epoch_warning(self):
        records = [{"attempt": i + 1, "successful_update": i < 6526,
                    "amp_skipped": i == 6526} for i in range(6527)]
        report = audit_legacy_checkpoint({"current_epoch": 27}, records, 1300)
        self.assertEqual(report["successful_updates"], 6526)
        self.assertEqual(report["remaining_attempts_current_epoch"], 223)
        self.assertEqual(report["remaining_total_attempts_if_fixed_25000"], 18473)
        self.assertFalse(report["saved_epoch_matches_cursor"])
        self.assertFalse(report["strict_resume_supported"])


class ScalarNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5))
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1
        return self.weight.square()


class ScalarTrainer:
    def __init__(self, inject_overflow=False):
        self.network = ScalarNetwork()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=.1, momentum=.99)
        self.grad_scaler = torch.amp.GradScaler("cpu", init_scale=8., growth_interval=2000)
        self.inject_overflow = inject_overflow

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.network()
        self.grad_scaler.scale(loss).backward()
        if self.inject_overflow:
            self.network.weight.grad.fill_(float("inf"))
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return {"loss": loss.detach().numpy()}


class RuntimeScalarTests(unittest.TestCase):
    def test_real_cpu_amp_skip_then_success(self):
        t = ScalarTrainer(inject_overflow=True); b = ledger(); c = TrainingCursor(100, 250, 50)
        initial = t.network.weight.detach().clone()
        _, first = observe_train_step(t, {}, b, c)
        self.assertEqual(first["classification"], "AMP_SKIP")
        self.assertEqual(first["optimizer_step_calls"], 0)
        self.assertEqual(t.grad_scaler.get_scale(), 4.)
        self.assertTrue(torch.equal(initial, t.network.weight))
        t.inject_overflow = False
        _, second = observe_train_step(t, {}, b, c)
        self.assertEqual(second["classification"], "SUCCESS")
        self.assertEqual((b.attempts, b.successful_updates, b.amp_skips), (2, 1, 1))
        self.assertEqual(t.network.forward_calls, 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_observer_preserves_native_step_and_rng(self):
        plain = ScalarTrainer(); wrapped = ScalarTrainer()
        state = torch.get_rng_state().clone()
        baseline = plain.train_step({})
        _, event = observe_train_step(wrapped, {}, ledger(), TrainingCursor(100, 250, 50))
        self.assertEqual(float(baseline["loss"]), event["loss"])
        self.assertTrue(torch.equal(plain.network.weight, wrapped.network.weight))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(plain.network.forward_calls, wrapped.network.forward_calls)

    def test_stop_event_logged_before_raise(self):
        t = ScalarTrainer(True); b = ledger(consecutive=1); c = TrainingCursor(100, 250, 50)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with self.assertRaises(RevisionStop):
                observe_train_step(t, {}, b, c, path)
            event = json.loads(path.read_text())
            self.assertEqual(event["counts"]["successful_updates"], 0)
            self.assertEqual(event["classification"], "AMP_SKIP")
            self.assertEqual(event["cursor_after_attempt"]["next_train_iteration"], 1)

    def test_exception_counts_and_restores_hooks(self):
        t = ScalarTrainer(); b = ledger(); c = TrainingCursor(100, 250, 50)
        clip = torch.nn.utils.clip_grad_norm_
        def fail(batch):
            raise RuntimeError("synthetic_train_step_failure")
        t.train_step = fail
        with self.assertRaises(RevisionStop):
            observe_train_step(t, {}, b, c)
        self.assertEqual((b.attempts, b.fatal_attempts), (1, 1))
        self.assertIs(torch.nn.utils.clip_grad_norm_, clip)
        self.assertEqual(len(t.optimizer._optimizer_step_pre_hooks), 0)

    def test_state_finiteness_scan(self):
        self.assertTrue(state_is_finite({"x": torch.tensor(1.)}))
        self.assertFalse(state_is_finite({"x": torch.tensor(float("inf"))}))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
