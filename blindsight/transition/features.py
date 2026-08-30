"""Causal 3076-value feature vector for the Stage 3 transition detector.

Every value in the feature vector uses only the current world state and past world states. There
is no acausal path in this module: a live decision at time ``t`` must be reproducible from a
one-step-at-a-time replay, so the offline (whole-array) path and the streaming (one-vector-at-a-time)
path are two independent implementations that are tested against each other for equality.

Feature layout (3076 values total), per ``docs/spec/phase-3-transition.md`` ("Features"):

- ``[0:1024]``    the current world state, L2-normalized.
- ``[1024:2048]`` the delta: current world state minus the previous world state.
- ``[2048:3072]`` the residual: current world state minus the previous EMA (smoothing factor 0.1).
- ``[3072]``      cosine(current world state, previous world state).
- ``[3073]``      cosine(current world state, previous EMA).
- ``[3074]``      norm(delta).
- ``[3075]``      norm(residual).

Boundary convention at the first time step of a stream: there is no true predecessor. Both paths
resolve this identically by treating the first world state as its own previous world state and its
own previous EMA, which makes the first step's delta and residual exactly zero and both cosines
exactly one. This is a documented convention, not an accident of implementation order, and it is
covered by a test so a change to it is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

WORLD_STATE_DIM = 1024
FEATURE_DIM = 3 * WORLD_STATE_DIM + 4
EMA_ALPHA = 0.1
_NORM_EPSILON = 1e-12


def _validate_world_state(world_state: np.ndarray) -> np.ndarray:
    array = np.asarray(world_state, dtype=np.float64)
    if array.shape != (WORLD_STATE_DIM,):
        raise ValueError(f"world_state must have shape ({WORLD_STATE_DIM},), got {array.shape}")
    return array


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_norms = np.where(norms < _NORM_EPSILON, 1.0, norms)
    normalized = vectors / safe_norms
    return np.where(norms < _NORM_EPSILON, 0.0, normalized)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.einsum("...i,...i->...", a, b)
    return np.where(denom < _NORM_EPSILON, 0.0, dot / np.where(denom < _NORM_EPSILON, 1.0, denom))


@dataclass
class StreamingFeatureState:
    """Incremental state for the causal feature stream.

    Holds exactly the two pieces of memory the feature vector needs: the previous world state and
    the previous exponential moving average. Call :meth:`step` once for each new world state, in
    order; it mutates the held state and returns that step's 3076-value feature vector.
    """

    previous_world_state: np.ndarray | None = None
    previous_ema: np.ndarray | None = None

    def step(self, world_state: np.ndarray) -> np.ndarray:
        current = _validate_world_state(world_state)
        if self.previous_world_state is None or self.previous_ema is None:
            previous_world_state = current
            previous_ema = current
        else:
            previous_world_state = self.previous_world_state
            previous_ema = self.previous_ema

        delta = current - previous_world_state
        residual = current - previous_ema
        normalized = _l2_normalize_rows(current)
        cosine_prev_state = float(_row_cosine(current, previous_world_state))
        cosine_prev_ema = float(_row_cosine(current, previous_ema))
        delta_norm = float(np.linalg.norm(delta))
        residual_norm = float(np.linalg.norm(residual))

        feature = np.concatenate(
            [
                normalized,
                delta,
                residual,
                np.array(
                    [cosine_prev_state, cosine_prev_ema, delta_norm, residual_norm],
                    dtype=np.float64,
                ),
            ]
        )

        self.previous_ema = EMA_ALPHA * current + (1.0 - EMA_ALPHA) * previous_ema
        self.previous_world_state = current
        return feature


def compute_features_streaming(world_states: np.ndarray) -> np.ndarray:
    """Compute the feature matrix by replaying :class:`StreamingFeatureState` one step at a time.

    This is a thin convenience wrapper around the incremental path, useful for tests and for any
    caller that already has a full array in hand but wants the literal streaming code path. The
    deployed streaming detector should hold its own long-lived :class:`StreamingFeatureState`
    instead of reconstructing one from a full array.
    """

    world_states = np.asarray(world_states, dtype=np.float64)
    if world_states.ndim != 2 or world_states.shape[1] != WORLD_STATE_DIM:
        raise ValueError(f"world_states must have shape (T, {WORLD_STATE_DIM}), got {world_states.shape}")
    state = StreamingFeatureState()
    return np.stack([state.step(world_states[t]) for t in range(world_states.shape[0])])


def compute_features_offline(world_states: np.ndarray) -> np.ndarray:
    """Compute the causal feature matrix for a full ``[T, 1024]`` array of world states.

    This is a genuinely vectorized implementation independent of :class:`StreamingFeatureState`: it
    operates across the whole time axis at once rather than looping through
    :meth:`StreamingFeatureState.step`. The two implementations are expected, and tested, to agree
    within floating-point tolerance — that agreement is the acceptance criterion, not an artifact of
    sharing code.

    The EMA recurrence (``ema[0] = ws[0]``, ``ema[t] = alpha * ws[t] + (1 - alpha) * ema[t-1]``) is
    computed with ``scipy.signal.lfilter``, which evaluates the identical linear recurrence directly
    rather than via a numerically unstable closed-form power series. The one-tap initial condition
    ``zi = (1 - alpha) * ws[0]`` is chosen so the filter's first output is exactly ``ws[0]``: with
    ``b = [alpha]`` and ``a = [1, -(1 - alpha)]``, direct-form-II-transposed gives
    ``y[0] = alpha * x[0] + zi``, so ``zi = (1 - alpha) * x[0]`` yields ``y[0] = x[0]``, and every
    subsequent step follows the plain recurrence.
    """

    world_states = np.asarray(world_states, dtype=np.float64)
    if world_states.ndim != 2 or world_states.shape[1] != WORLD_STATE_DIM:
        raise ValueError(f"world_states must have shape (T, {WORLD_STATE_DIM}), got {world_states.shape}")
    steps = world_states.shape[0]
    if steps == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float64)

    normalized = _l2_normalize_rows(world_states)

    previous_world_state = np.vstack([world_states[0:1], world_states[:-1]])
    delta = world_states - previous_world_state

    if steps == 1:
        ema = world_states.copy()
    else:
        initial_condition = ((1.0 - EMA_ALPHA) * world_states[0:1])
        ema, _ = lfilter(
            [EMA_ALPHA],
            [1.0, -(1.0 - EMA_ALPHA)],
            world_states,
            axis=0,
            zi=initial_condition,
        )
    previous_ema = np.vstack([world_states[0:1], ema[:-1]])
    residual = world_states - previous_ema

    cosine_prev_state = _row_cosine(world_states, previous_world_state)
    cosine_prev_ema = _row_cosine(world_states, previous_ema)
    delta_norm = np.linalg.norm(delta, axis=-1)
    residual_norm = np.linalg.norm(residual, axis=-1)

    scalars = np.stack([cosine_prev_state, cosine_prev_ema, delta_norm, residual_norm], axis=-1)
    return np.concatenate([normalized, delta, residual, scalars], axis=-1)
