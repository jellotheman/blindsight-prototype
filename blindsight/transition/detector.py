"""Trained heads, calibration, decision policy, selection rule, and ONNX export for Stage 3.

This module builds the two detector heads that ``train_transition_detector.py`` fits and evaluates:

- **Logistic head**: a standardized linear model over the 3076-value causal feature vector from
  :mod:`blindsight.transition.features`.
- **Causal GRU head**: linear(1024, 32) -> tanh -> GRU(32) -> linear(32, 1), consuming the trailing
  8 time steps of the *normalized world state* component of the feature vector (feature indices
  ``[0:1024]``). Both heads therefore derive from the one causal feature computation; the GRU simply
  uses a slice of it as a time series instead of the whole vector at a single step.

Both heads emit a raw logit. A :class:`PlattCalibration` (one slope, one bias) turns a raw logit into
a calibrated probability. :class:`DecisionPolicyState` turns a calibrated probability stream into
transition events with fixed hysteresis, persistence, and cooldown. :func:`select_detector_head`
picks between the two calibrated heads with a fixed, pre-registered rule. The ``export_*_to_onnx``
functions bake standardization/embedding, the trained weights, and calibration into one ONNX graph
per head, so serving needs no PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from torch import nn

from blindsight.transition.features import WORLD_STATE_DIM, StreamingFeatureState

FEATURE_DIM = 3 * WORLD_STATE_DIM + 4
GRU_HISTORY_LENGTH = 8
GRU_HIDDEN_SIZE = 32

ACTIVATE_THRESHOLD = 0.8
RELEASE_THRESHOLD = 0.4
PERSISTENCE_STEPS = 2
COOLDOWN_SECONDS = 10.0

_STANDARDIZE_EPSILON = 1e-8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------------------
# Logistic head
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogisticHead:
    """A standardized linear model over the 3076-value causal feature vector.

    ``mean``/``scale`` are fit once on the training split and frozen; every later call
    standardizes with those fixed values, never re-fitting on evaluation data.
    """

    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        for name, array in (("mean", self.mean), ("scale", self.scale), ("weight", self.weight)):
            if array.shape != (FEATURE_DIM,):
                raise ValueError(f"LogisticHead.{name} must have shape ({FEATURE_DIM},), got {array.shape}")

    def logit(self, features: np.ndarray) -> np.ndarray:
        """Return the raw logit for a ``[N, 3076]`` (or ``[3076]``) feature array."""

        features = np.asarray(features, dtype=np.float64)
        standardized = (features - self.mean) / self.scale
        return standardized @ self.weight + self.bias

    @staticmethod
    def fit(features: np.ndarray, labels: np.ndarray, *, C: float = 1.0) -> "LogisticHead":
        """Fit mean/scale/weight/bias on the training split via scikit-learn logistic regression."""

        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(f"features must have shape (N, {FEATURE_DIM}), got {features.shape}")
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < _STANDARDIZE_EPSILON, 1.0, scale)
        standardized = (features - mean) / scale
        model = LogisticRegression(C=C, max_iter=5000)
        model.fit(standardized, labels)
        return LogisticHead(mean=mean, scale=scale, weight=model.coef_[0].astype(np.float64), bias=float(model.intercept_[0]))


# --------------------------------------------------------------------------------------
# Causal GRU head
# --------------------------------------------------------------------------------------


class CausalGRUHead(nn.Module):
    """linear(1024, 32) -> tanh -> GRU(32 hidden units) -> linear(32, 1), history length 8.

    A parameter-count sanity check: this architecture, as literally specified (a 1024-to-32 input
    projection, one GRU layer, and a 32-to-1 output projection), has roughly 39000 parameters, not
    the "about 4000" the prose estimate in the specification states. The input projection alone
    (``1024 * 32 + 32 = 32832``) already exceeds that estimate. This is flagged rather than silently
    "corrected" by shrinking the architecture, because the architecture itself
    (1024 -> 32 -> GRU(32) -> 1, history 8) is stated unambiguously and is not in conflict with any
    other requirement; the parameter-count prose appears to be an estimation error in the source
    document. See :data:`EXPECTED_PARAMETER_COUNT_RANGE` and the accompanying test.
    """

    def __init__(
        self,
        world_state_dim: int = WORLD_STATE_DIM,
        hidden_size: int = GRU_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(world_state_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_linear = nn.Linear(hidden_size, 1)

    def forward(self, world_state_window: torch.Tensor) -> torch.Tensor:
        """``world_state_window``: ``[batch, history_length, world_state_dim]`` -> ``[batch]`` logits."""

        projected = torch.tanh(self.input_linear(world_state_window))
        _, hidden = self.gru(projected)
        logit = self.output_linear(hidden[-1])
        return logit.squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


EXPECTED_PARAMETER_COUNT_RANGE = (30_000, 45_000)


def build_gru_windows(normalized_world_states: np.ndarray, history_length: int = GRU_HISTORY_LENGTH) -> np.ndarray:
    """Build causal, left-padded sliding windows: window ``t`` covers steps ``[t-7, ..., t]``.

    Padding for ``t < history_length - 1`` repeats the first row, matching the same "no true
    predecessor yet" convention used in :mod:`blindsight.transition.features`. No window ever
    contains a future world state.
    """

    world_states = np.asarray(normalized_world_states, dtype=np.float32)
    if world_states.ndim != 2:
        raise ValueError(f"normalized_world_states must be 2-D, got shape {world_states.shape}")
    steps, dim = world_states.shape
    if steps == 0:
        return np.zeros((0, history_length, dim), dtype=np.float32)
    padded = np.vstack([np.repeat(world_states[0:1], history_length - 1, axis=0), world_states])
    windows = np.stack([padded[i : i + history_length] for i in range(steps)], axis=0)
    return windows


def fit_gru_head(
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> CausalGRUHead:
    """Fit the causal GRU head by gradient descent on binary cross-entropy.

    ``labels`` must already have ignored steps removed by the caller — this function trains on
    exactly the rows it is given. Positive-class weighting compensates for the heavy class
    imbalance a transition detector always has (positives are a handful of seconds; negatives are
    everywhere else).
    """

    windows = np.asarray(windows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if windows.shape[0] != labels.shape[0]:
        raise ValueError("windows and labels must have the same leading (sample) dimension")
    if windows.shape[0] == 0:
        raise ValueError("Cannot fit the GRU head on zero training rows")

    torch.manual_seed(seed)
    head = CausalGRUHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate, weight_decay=weight_decay)

    positive_rate = float(labels.mean())
    positive_weight = 1.0
    if 0.0 < positive_rate < 1.0:
        positive_weight = (1.0 - positive_rate) / positive_rate
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))

    windows_tensor = torch.from_numpy(windows)
    labels_tensor = torch.from_numpy(labels)

    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = head(windows_tensor)
        loss = loss_fn(logits, labels_tensor)
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def gru_logits(head: CausalGRUHead, windows: np.ndarray) -> np.ndarray:
    """Run the trained GRU head over a batch of windows and return raw logits (no grad)."""

    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(np.asarray(windows, dtype=np.float32)))
    return logits.numpy().astype(np.float64)


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlattCalibration:
    """One slope and one bias over a head's raw logit: ``sigmoid(slope * logit + bias)``."""

    slope: float
    bias: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        return _sigmoid(self.slope * logits + self.bias)

    @staticmethod
    def fit(logits: np.ndarray, labels: np.ndarray) -> "PlattCalibration":
        logits = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        labels = np.asarray(labels)
        if logits.shape[0] == 0:
            raise ValueError("Cannot fit calibration on zero rows")
        if len(np.unique(labels)) < 2:
            # Degenerate calibration set (all one class): identity-ish mapping that still saturates
            # toward the observed class, rather than raising inside a training run.
            bias = 8.0 if float(np.mean(labels)) > 0.5 else -8.0
            return PlattCalibration(slope=1.0, bias=bias)
        model = LogisticRegression(C=1e6, max_iter=5000)
        model.fit(logits, labels)
        return PlattCalibration(slope=float(model.coef_[0][0]), bias=float(model.intercept_[0]))


# --------------------------------------------------------------------------------------
# Decision policy
# --------------------------------------------------------------------------------------


@dataclass
class DecisionPolicyState:
    """Streaming hysteresis/persistence/cooldown state machine, O(1) per step.

    Hysteresis: ``consecutive_high`` increments only when the probability reaches the activate
    threshold (0.8), and resets to zero only when the probability drops below the lower release
    threshold (0.4). Between the two thresholds the counter is held unchanged — this dead band is
    exactly what stops a probability dithering near 0.8 from oscillating the persistence count.

    Persistence: an event needs 2 consecutive high steps once inside the dead band's high side.

    Cooldown: once an event fires, no further event can fire for 10 seconds of stream time (tracked
    via caller-supplied ``dt_seconds``, so it is correct even if the step cadence is not exactly
    1 Hz). Firing also resets ``consecutive_high`` to zero, so a later event needs its own fresh
    persistence build-up.
    """

    consecutive_high: int = 0
    cooldown_remaining_seconds: float = 0.0

    def step(self, probability: float, dt_seconds: float = 1.0) -> bool:
        """Advance by one time step; return True exactly when a transition event fires now."""

        if dt_seconds < 0:
            raise ValueError("dt_seconds cannot be negative")
        if self.cooldown_remaining_seconds > 0.0:
            self.cooldown_remaining_seconds = max(0.0, self.cooldown_remaining_seconds - dt_seconds)

        if probability >= ACTIVATE_THRESHOLD:
            self.consecutive_high = min(self.consecutive_high + 1, PERSISTENCE_STEPS)
        elif probability < RELEASE_THRESHOLD:
            self.consecutive_high = 0
        # else: probability is in the [release, activate) dead band -> hold steady.

        if self.consecutive_high >= PERSISTENCE_STEPS and self.cooldown_remaining_seconds <= 0.0:
            self.consecutive_high = 0
            self.cooldown_remaining_seconds = COOLDOWN_SECONDS
            return True
        return False


# --------------------------------------------------------------------------------------
# Streaming detector (Artifacts item 3: one incremental path per head)
# --------------------------------------------------------------------------------------


@dataclass
class DetectionStep:
    logit: float
    probability: float
    transition_event: bool


@dataclass
class StreamingLogisticDetector:
    """Accepts one world state at a time; returns a calibrated probability and a decision.

    Holds the EMA state, the feature history (via :class:`StreamingFeatureState`), and the policy
    state, exactly as the specification's "Streaming detector" artifact requires.
    """

    head: LogisticHead
    calibration: PlattCalibration
    feature_state: StreamingFeatureState = field(default_factory=StreamingFeatureState)
    policy_state: DecisionPolicyState = field(default_factory=DecisionPolicyState)

    def step(self, world_state: np.ndarray, dt_seconds: float = 1.0) -> DetectionStep:
        feature = self.feature_state.step(world_state)
        logit = float(self.head.logit(feature[None, :])[0])
        probability = float(self.calibration.apply(np.array([logit]))[0])
        fired = self.policy_state.step(probability, dt_seconds)
        return DetectionStep(logit=logit, probability=probability, transition_event=fired)


@dataclass
class StreamingGRUDetector:
    """The GRU-head equivalent of :class:`StreamingLogisticDetector`.

    Holds the same causal feature state (for the normalized-world-state slice the GRU consumes),
    a rolling history buffer of the last 8 normalized world states, and the same decision-policy
    state shape.
    """

    head: CausalGRUHead
    calibration: PlattCalibration
    feature_state: StreamingFeatureState = field(default_factory=StreamingFeatureState)
    policy_state: DecisionPolicyState = field(default_factory=DecisionPolicyState)
    history_length: int = GRU_HISTORY_LENGTH
    _history: list[np.ndarray] = field(default_factory=list)

    def step(self, world_state: np.ndarray, dt_seconds: float = 1.0) -> DetectionStep:
        feature = self.feature_state.step(world_state)
        normalized = feature[:WORLD_STATE_DIM]
        if not self._history:
            self._history = [normalized] * self.history_length
        else:
            self._history.append(normalized)
            self._history = self._history[-self.history_length :]
        window = np.stack(self._history, axis=0)[None, :, :]
        logit = float(gru_logits(self.head, window)[0])
        probability = float(self.calibration.apply(np.array([logit]))[0])
        fired = self.policy_state.step(probability, dt_seconds)
        return DetectionStep(logit=logit, probability=probability, transition_event=fired)


# --------------------------------------------------------------------------------------
# Selection rule
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadSelectionResult:
    selected: str  # "logistic" or "gru"
    logistic_average_precision: float
    gru_average_precision: float
    gru_group_wins: int
    total_groups: int
    reason: str


def select_detector_head(
    logistic_average_precision: float,
    gru_average_precision: float,
    per_group_average_precision: Sequence[tuple[float, float]],
) -> HeadSelectionResult:
    """Pure function over precomputed per-group AP values; fixed before training, per spec.

    ``per_group_average_precision`` is one ``(logistic_ap, gru_ap)`` pair per held-out group (a
    held-out group is a distinct source video, so correlated clips from one video cannot each cast
    a separate "vote"). The GRU is selected only when its overall AP beats the logistic head's by at
    least 0.02 *and* it wins on strictly more than half of the groups; the logistic head wins in
    every other case, including a tie on either condition.
    """

    total_groups = len(per_group_average_precision)
    gru_group_wins = sum(1 for logistic_ap, gru_ap in per_group_average_precision if gru_ap > logistic_ap)
    ap_margin = gru_average_precision - logistic_average_precision
    beats_by_margin = ap_margin >= 0.02
    wins_majority = total_groups > 0 and gru_group_wins > total_groups / 2

    if beats_by_margin and wins_majority:
        selected = "gru"
        reason = (
            f"GRU average precision exceeds logistic by {ap_margin:.4f} (>= 0.02 required) and wins "
            f"{gru_group_wins}/{total_groups} held-out groups (> half required)."
        )
    else:
        selected = "logistic"
        reason = (
            f"Selection rule not satisfied: AP margin {ap_margin:.4f} (>= 0.02 required: {beats_by_margin}), "
            f"group wins {gru_group_wins}/{total_groups} (> half required: {wins_majority}). "
            "Logistic head wins by default."
        )
    return HeadSelectionResult(
        selected=selected,
        logistic_average_precision=logistic_average_precision,
        gru_average_precision=gru_average_precision,
        gru_group_wins=gru_group_wins,
        total_groups=total_groups,
        reason=reason,
    )


# --------------------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------------------


class _LogisticExportModule(nn.Module):
    """Standardization + linear logit + Platt calibration, as one traceable graph."""

    # Explicit annotations so mypy resolves these as Tensor rather than nn.Module's
    # `Tensor | Module` stub for arbitrary attribute access after register_buffer.
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    calib_slope: torch.Tensor
    calib_bias: torch.Tensor

    def __init__(self, head: LogisticHead, calibration: PlattCalibration) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(head.mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(head.scale, dtype=torch.float32))
        self.register_buffer("weight", torch.tensor(head.weight, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(head.bias, dtype=torch.float32))
        self.register_buffer("calib_slope", torch.tensor(calibration.slope, dtype=torch.float32))
        self.register_buffer("calib_bias", torch.tensor(calibration.bias, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        standardized = (features - self.mean) / self.scale
        logit = standardized @ self.weight + self.bias
        probability = torch.sigmoid(self.calib_slope * logit + self.calib_bias)
        return logit, probability


class _GRUExportModule(nn.Module):
    """The trained :class:`CausalGRUHead` plus Platt calibration, as one traceable graph."""

    calib_slope: torch.Tensor
    calib_bias: torch.Tensor

    def __init__(self, head: CausalGRUHead, calibration: PlattCalibration) -> None:
        super().__init__()
        self.head = head
        self.register_buffer("calib_slope", torch.tensor(calibration.slope, dtype=torch.float32))
        self.register_buffer("calib_bias", torch.tensor(calibration.bias, dtype=torch.float32))

    def forward(self, world_state_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.head(world_state_window)
        probability = torch.sigmoid(self.calib_slope * logit + self.calib_bias)
        return logit, probability


def export_logistic_to_onnx(head: LogisticHead, calibration: PlattCalibration, path: Path) -> None:
    """Export the calibrated logistic head to ONNX. Input: ``[batch, 3076]``. Outputs: (logit, probability)."""

    module = _LogisticExportModule(head, calibration)
    module.eval()
    example = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        module,
        (example,),
        str(path),
        input_names=["features"],
        output_names=["logit", "probability"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
        # The newer dynamo-based exporter (torch's default since this build) mishandles
        # `dynamic_axes` for a batch dimension here, silently baking the example's batch size of 1
        # into the graph. The legacy TorchScript-based exporter honors `dynamic_axes` correctly and
        # needs no extra dependency (`onnxscript`) beyond what this package already requires.
        dynamo=False,
    )


def export_gru_to_onnx(head: CausalGRUHead, calibration: PlattCalibration, path: Path) -> None:
    """Export the calibrated GRU head to ONNX. Input: ``[batch, 8, 1024]``. Outputs: (logit, probability)."""

    module = _GRUExportModule(head, calibration)
    module.eval()
    example = torch.zeros(1, GRU_HISTORY_LENGTH, WORLD_STATE_DIM, dtype=torch.float32)
    torch.onnx.export(
        module,
        (example,),
        str(path),
        input_names=["world_state_window"],
        output_names=["logit", "probability"],
        dynamic_axes={
            "world_state_window": {0: "batch"},
            "logit": {0: "batch"},
            "probability": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
