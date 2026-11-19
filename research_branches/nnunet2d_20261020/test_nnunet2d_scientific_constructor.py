"""CPU-only regression tests for scientific construction; no network forward."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from nnunet2d_scientific_constructor import build_fresh_trainer, validate_initialized_batch_size


class FakeTrainer:
    """Model the installed constructor's runtime flag pop and late batch field."""
    def __init__(self, plans, configuration, fold, dataset, device):
        assert plans.pop("continue_training") is False
        assert (configuration, fold) == ("2d", "all")
        self.configuration_manager = SimpleNamespace(batch_size=2, patch_size=[1024, 1536])
        self.num_epochs = 100
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.network = self.optimizer = None
        self.was_initialized = False
        # Mutation here must not affect the caller's plans or dataset.
        plans.setdefault("nested", {})["constructor_only"] = True
        dataset["constructor_only"] = True


class ConstructorRegression(unittest.TestCase):
    def test_fresh_flag_added_without_mutating_inputs(self):
        plans = {"nested": {"value": 7}}
        dataset = {"labels": {"background": 0, "vessel": 1}}
        before = copy.deepcopy((plans, dataset))
        build_fresh_trainer(plans, dataset, "cpu", trainer_class=FakeTrainer)
        self.assertEqual((plans, dataset), before)

    def test_configuration_is_available_before_runtime_batch_attribute(self):
        trainer = build_fresh_trainer({}, {}, "cpu", trainer_class=FakeTrainer)
        self.assertFalse(hasattr(trainer, "batch_size"))
        self.assertEqual(trainer.configuration_manager.batch_size, 2)

    def test_reject_continuation(self):
        with self.assertRaisesRegex(ValueError, "fresh"):
            build_fresh_trainer({"continue_training": True}, {}, "cpu", trainer_class=FakeTrainer)

    def test_reject_configuration_drift(self):
        class WrongTrainer(FakeTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.configuration_manager.batch_size = 3
        with self.assertRaisesRegex(RuntimeError, "configuration mismatch"):
            build_fresh_trainer({}, {}, "cpu", trainer_class=WrongTrainer)

    def test_reject_eager_initialization(self):
        class EagerTrainer(FakeTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.was_initialized = True
        with self.assertRaisesRegex(RuntimeError, "unexpectedly initialized"):
            build_fresh_trainer({}, {}, "cpu", trainer_class=EagerTrainer)

    def test_postinitialize_batch_guard(self):
        trainer = build_fresh_trainer({}, {}, "cpu", trainer_class=FakeTrainer)
        with self.assertRaisesRegex(RuntimeError, "Initialized"):
            validate_initialized_batch_size(trainer)
        trainer.was_initialized = True
        trainer.batch_size = 2
        validate_initialized_batch_size(trainer)
        trainer.batch_size = 3
        with self.assertRaises(RuntimeError):
            validate_initialized_batch_size(trainer)

    def test_runner_uses_shared_constructor_and_no_early_batch_read(self):
        tree = ast.parse(Path(__file__).with_name("nnunet2d_scientific_train.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        shared = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "build_fresh_trainer"]
        initializes = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "initialize"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(initializes), 1)
        self.assertLess(shared[0].lineno, initializes[0].lineno)
        early = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and isinstance(n.value, ast.Name) and n.value.id == "trainer"
                 and n.attr == "batch_size" and n.lineno < initializes[0].lineno]
        self.assertEqual(early, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
