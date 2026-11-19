"""CPU-reviewed candidate policy; not a GPU launch or scientific budget approval."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math


class RevisionStop(RuntimeError):
    pass


@dataclass(frozen=True)
class AmpLimits:
    # Explicit configuration required: no implicit thresholds or default approval.
    max_consecutive_skips: int
    max_total_skips: int

    def __post_init__(self):
        if (type(self.max_consecutive_skips) is not int or
            type(self.max_total_skips) is not int or
            self.max_consecutive_skips < 1 or self.max_total_skips < 1):
            raise ValueError("Positive explicit AMP limits required")


@dataclass(frozen=True)
class StepEvidence:
    loss: float
    gradient_norm: float
    scale_before: float
    scale_after: float
    optimizer_step_calls: int
    model_state_finite: bool
    optimizer_state_finite: bool


def classify_step(evidence: StepEvidence):
    e = evidence
    valid_scale = (math.isfinite(e.scale_before) and math.isfinite(e.scale_after)
                   and e.scale_before > 0 and e.scale_after > 0)
    if not math.isfinite(e.loss):
        return "FATAL", "nonfinite_forward_loss"
    if not e.model_state_finite or not e.optimizer_state_finite:
        return "FATAL", "nonfinite_parameter_or_optimizer_state"
    if not valid_scale:
        return "FATAL", "invalid_amp_scale"
    if (e.optimizer_step_calls == 0 and e.scale_after < e.scale_before):
        # Nonfinite gradients are expected here: GradScaler prevented their update.
        return "AMP_SKIP", "scaler_backoff_without_optimizer_update"
    if (e.optimizer_step_calls == 1 and e.scale_after >= e.scale_before
            and math.isfinite(e.gradient_norm)):
        return "SUCCESS", "finite_optimizer_update"
    return "FATAL", "inconsistent_step_or_unhandled_nonfinite_gradient"


@dataclass
class AttemptLedger:
    attempt_cap: int
    limits: AmpLimits
    attempts: int = 0
    successful_updates: int = 0
    amp_skips: int = 0
    fatal_attempts: int = 0
    optimizer_step_calls: int = 0
    consecutive_skips: int = 0
    stopped_reason: str | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.limits, AmpLimits):
            raise ValueError("Explicit AMP limit policy required")
        for key in ("attempt_cap", "attempts", "successful_updates", "amp_skips",
                    "fatal_attempts", "optimizer_step_calls", "consecutive_skips"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError("Invalid nonnegative counter: " + key)
        if self.attempt_cap < 1 or self.attempts > self.attempt_cap:
            raise ValueError("Attempt budget exceeded/invalid")
        if self.attempts != self.successful_updates + self.amp_skips + self.fatal_attempts:
            raise ValueError("Attempt/success/skip/failure counters disagree")
        if self.optimizer_step_calls < self.successful_updates:
            raise ValueError("Successful updates cannot exceed actual optimizer calls")
        if self.consecutive_skips > self.amp_skips:
            raise ValueError("Consecutive skips exceed total")

    def before_attempt(self):
        if self.stopped_reason:
            raise RevisionStop(self.stopped_reason)
        if self.attempts >= self.attempt_cap:
            raise RevisionStop("fixed_attempt_budget_exhausted")

    def record(self, evidence: StepEvidence):
        self.before_attempt()
        kind, reason = classify_step(evidence)
        self.attempts += 1
        self.optimizer_step_calls += max(0, evidence.optimizer_step_calls)
        if kind == "SUCCESS":
            self.successful_updates += 1
            self.consecutive_skips = 0
        elif kind == "AMP_SKIP":
            self.amp_skips += 1
            self.consecutive_skips += 1
            if self.consecutive_skips >= self.limits.max_consecutive_skips:
                self.stopped_reason = "consecutive_amp_skip_limit"
            elif self.amp_skips >= self.limits.max_total_skips:
                self.stopped_reason = "total_amp_skip_limit"
        else:
            self.fatal_attempts += 1
            self.stopped_reason = reason
        self.validate()
        return {"attempt": self.attempts, "classification": kind, "reason": reason,
                "successful_update": kind == "SUCCESS",
                "amp_skipped": kind == "AMP_SKIP",
                "stop_reason": self.stopped_reason}

    def snapshot(self):
        self.validate()
        return asdict(self)

    def record_exception(self, reason, optimizer_step_calls=0):
        self.before_attempt()
        self.attempts += 1
        self.fatal_attempts += 1
        self.optimizer_step_calls += optimizer_step_calls
        self.stopped_reason = reason
        self.validate()
        return {"attempt": self.attempts, "classification": "FATAL",
                "reason": reason, "successful_update": False,
                "amp_skipped": False, "stop_reason": reason}

    def fixed_attempts_complete(self):
        return (self.attempts == self.attempt_cap and self.fatal_attempts == 0
                and self.stopped_reason is None)


@dataclass
class TrainingCursor:
    epochs: int
    attempts_per_epoch: int
    monitors_per_epoch: int
    epoch_index: int = 0
    next_train_iteration: int = 0
    completed_monitor_batches: int = 0
    phase: str = "train"

    def validate(self):
        if min(self.epochs, self.attempts_per_epoch, self.monitors_per_epoch) < 1:
            raise ValueError("Invalid schedule")
        if self.phase not in ("train", "monitor", "epoch_finalize", "complete"):
            raise ValueError("Unknown execution phase")
        if not 0 <= self.epoch_index <= self.epochs:
            raise ValueError("Epoch index outside schedule")
        if self.phase == "complete":
            if (self.epoch_index, self.next_train_iteration, self.completed_monitor_batches) != (self.epochs, 0, 0):
                raise ValueError("Invalid terminal cursor")
        elif self.epoch_index >= self.epochs:
            raise ValueError("Nonterminal cursor beyond last epoch")
        elif self.phase == "train":
            if not 0 <= self.next_train_iteration < self.attempts_per_epoch or self.completed_monitor_batches != 0:
                raise ValueError("Invalid in-training cursor")
        elif self.phase == "monitor":
            if self.next_train_iteration != self.attempts_per_epoch or not 0 <= self.completed_monitor_batches < self.monitors_per_epoch:
                raise ValueError("Invalid monitor cursor")
        elif (self.next_train_iteration != self.attempts_per_epoch or
              self.completed_monitor_batches != self.monitors_per_epoch):
            raise ValueError("Epoch finalization before all work completed")

    @property
    def total_attempts(self):
        return self.epoch_index * self.attempts_per_epoch + self.next_train_iteration

    @property
    def total_monitors(self):
        return self.epoch_index * self.monitors_per_epoch + self.completed_monitor_batches

    def record_train_attempt(self):
        self.validate()
        if self.phase != "train":
            raise RevisionStop("train_attempt_in_wrong_phase")
        self.next_train_iteration += 1
        if self.next_train_iteration == self.attempts_per_epoch:
            self.phase = "monitor"
        self.validate()

    def record_monitor_batch(self):
        self.validate()
        if self.phase != "monitor":
            raise RevisionStop("monitor_in_wrong_phase")
        self.completed_monitor_batches += 1
        if self.completed_monitor_batches == self.monitors_per_epoch:
            self.phase = "epoch_finalize"
        self.validate()

    def finalize_epoch(self):
        self.validate()
        if self.phase != "epoch_finalize":
            raise RevisionStop("cannot_skip_unfinished_epoch")
        self.epoch_index += 1
        self.next_train_iteration = 0
        self.completed_monitor_batches = 0
        self.phase = "complete" if self.epoch_index == self.epochs else "train"
        self.validate()

    def snapshot(self):
        self.validate()
        return asdict(self)


def require_gpu_execution_authorization(*args, **kwargs):
    raise RevisionStop("CPU_ONLY_REVISION: GPU execution requires a separately approved protocol and runner")
