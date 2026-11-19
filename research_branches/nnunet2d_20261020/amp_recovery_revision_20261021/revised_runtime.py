"""Candidate observer around one existing train_step; never starts a training loop."""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
from pathlib import Path
from unittest.mock import patch
import torch
from revised_policy import StepEvidence, RevisionStop


class OptimizerCallObserver:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.calls = 0
        self.handle = None

    def __enter__(self):
        self.handle = self.optimizer.register_step_pre_hook(self._before_step)
        return self

    def _before_step(self, optimizer, args, kwargs):
        self.calls += 1

    def __exit__(self, *exc):
        self.handle.remove()
        self.handle = None


def state_is_finite(state):
    flags = []
    def visit(value):
        if torch.is_tensor(value):
            if value.is_floating_point() or value.is_complex():
                flags.append(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(state)
    # No forward or optimizer calls; compatible with CPU and one-device state.
    return all(bool(flag.item()) for flag in flags)


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value.reshape(-1)[0]) if hasattr(value, "reshape") else float(value)


def encode_number(value):
    return value if math.isfinite(value) else str(value)


def observe_train_step(trainer, batch, ledger, cursor, event_path=None):
    """Explicit CPU-tested candidate helper, not wired into the old scientific runner."""
    ledger.before_attempt()
    cursor.validate()
    if cursor.phase != "train" or cursor.total_attempts != ledger.attempts:
        raise RevisionStop("cursor_and_attempt_ledger_disagree")
    scaler = trainer.grad_scaler
    before = float(scaler.get_scale()) if scaler is not None else 1.0
    norm = {"value": float("nan")}
    original_clip = torch.nn.utils.clip_grad_norm_
    def capture_clip(parameters, max_norm, *args, **kwargs):
        result = original_clip(parameters, max_norm, *args, **kwargs)
        norm["value"] = scalar(result)
        return result
    # Existing nnU-Net forward/backward/step runs exactly once. Hooks never step.
    observer = OptimizerCallObserver(trainer.optimizer)
    try:
        with observer:
            with patch("torch.nn.utils.clip_grad_norm_", capture_clip):
                output = trainer.train_step(batch)
    except BaseException as exc:
        event = ledger.record_exception("train_step_exception:" + type(exc).__name__,
                                        observer.calls)
        cursor.record_train_attempt()
        event.update({"exception": repr(exc), "counts": ledger.snapshot(),
                      "cursor_after_attempt": cursor.snapshot(),
                      "optimizer_step_calls": observer.calls})
        if event_path is not None:
            with Path(event_path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        raise RevisionStop(event["reason"]) from exc
    after = float(scaler.get_scale()) if scaler is not None else 1.0
    evidence = StepEvidence(
        loss=scalar(output["loss"]), gradient_norm=norm["value"],
        scale_before=before, scale_after=after, optimizer_step_calls=observer.calls,
        model_state_finite=state_is_finite(trainer.network.state_dict()),
        optimizer_state_finite=state_is_finite(trainer.optimizer.state_dict()))
    event = ledger.record(evidence)
    cursor.record_train_attempt()
    event.update({"loss": encode_number(evidence.loss),
                  "gradient_norm": encode_number(evidence.gradient_norm),
                  "amp_scale_before": before, "amp_scale_after": after,
                  "optimizer_step_calls": observer.calls,
                  "model_state_finite": evidence.model_state_finite,
                  "optimizer_state_finite": evidence.optimizer_state_finite,
                  "counts": ledger.snapshot(), "cursor_after_attempt": cursor.snapshot()})
    if event_path is not None:
        with Path(event_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
    if ledger.stopped_reason:
        raise RevisionStop(ledger.stopped_reason)
    return output, event
