import numpy as np
import pytest

from blindsight.transition.features import (
    EMA_ALPHA,
    FEATURE_DIM,
    WORLD_STATE_DIM,
    StreamingFeatureState,
    compute_features_offline,
    compute_features_streaming,
)


def _random_world_states(steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(steps, WORLD_STATE_DIM)).astype(np.float32)


def test_feature_dimension_is_3076() -> None:
    assert FEATURE_DIM == 3076


def test_offline_and_streaming_agree_on_a_random_sequence() -> None:
    world_states = _random_world_states(40)

    offline = compute_features_offline(world_states)
    streaming = compute_features_streaming(world_states)

    assert offline.shape == (40, FEATURE_DIM)
    assert streaming.shape == (40, FEATURE_DIM)
    np.testing.assert_allclose(offline, streaming, rtol=1e-6, atol=1e-6)


def test_streaming_state_step_by_step_matches_offline() -> None:
    world_states = _random_world_states(25, seed=1)
    offline = compute_features_offline(world_states)

    state = StreamingFeatureState()
    for t in range(world_states.shape[0]):
        one_step = state.step(world_states[t])
        np.testing.assert_allclose(one_step, offline[t], rtol=1e-6, atol=1e-6)


def test_first_step_has_zero_delta_zero_residual_and_unit_cosines() -> None:
    world_states = _random_world_states(5, seed=2)

    feature = compute_features_offline(world_states)[0]
    delta = feature[WORLD_STATE_DIM : 2 * WORLD_STATE_DIM]
    residual = feature[2 * WORLD_STATE_DIM : 3 * WORLD_STATE_DIM]
    cosine_prev_state, cosine_prev_ema, delta_norm, residual_norm = feature[3 * WORLD_STATE_DIM :]

    np.testing.assert_allclose(delta, 0.0, atol=1e-9)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)
    assert cosine_prev_state == pytest.approx(1.0, abs=1e-6)
    assert cosine_prev_ema == pytest.approx(1.0, abs=1e-6)
    assert delta_norm == pytest.approx(0.0, abs=1e-9)
    assert residual_norm == pytest.approx(0.0, abs=1e-9)


def test_normalized_component_is_unit_norm() -> None:
    world_states = _random_world_states(10, seed=3)
    feature = compute_features_offline(world_states)
    normalized = feature[:, :WORLD_STATE_DIM]

    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


def test_zero_world_state_normalizes_to_zero_without_nan() -> None:
    world_states = np.zeros((3, WORLD_STATE_DIM), dtype=np.float32)

    feature = compute_features_offline(world_states)

    assert np.all(np.isfinite(feature))
    np.testing.assert_allclose(feature[:, :WORLD_STATE_DIM], 0.0)


def test_feature_at_time_t_is_unaffected_by_future_world_states() -> None:
    """No future frame may enter a causal feature: truncating the array must not change the past."""

    world_states = _random_world_states(30, seed=4)
    full = compute_features_offline(world_states)

    for truncate_at in (5, 10, 20):
        truncated = compute_features_offline(world_states[:truncate_at])
        np.testing.assert_allclose(truncated, full[:truncate_at], rtol=1e-6, atol=1e-6)


def test_ema_smoothing_factor_is_one_tenth() -> None:
    assert EMA_ALPHA == pytest.approx(0.1)


def test_delta_and_residual_reflect_ema_recurrence() -> None:
    world_states = np.array(
        [
            np.full(WORLD_STATE_DIM, 1.0, dtype=np.float32),
            np.full(WORLD_STATE_DIM, 2.0, dtype=np.float32),
            np.full(WORLD_STATE_DIM, 2.0, dtype=np.float32),
        ]
    )

    feature = compute_features_offline(world_states)

    # step 0: ema = 1.0 (bootstrap), delta = 0, residual = 0
    np.testing.assert_allclose(feature[0, WORLD_STATE_DIM : 2 * WORLD_STATE_DIM], 0.0, atol=1e-6)
    # step 1: previous world state = 1.0 -> delta = 1.0; previous ema = 1.0 -> residual = 1.0
    np.testing.assert_allclose(feature[1, WORLD_STATE_DIM : 2 * WORLD_STATE_DIM], 1.0, atol=1e-6)
    np.testing.assert_allclose(feature[1, 2 * WORLD_STATE_DIM : 3 * WORLD_STATE_DIM], 1.0, atol=1e-6)
    # ema after step 1 = 0.1*2.0 + 0.9*1.0 = 1.1
    # step 2: previous world state = 2.0 -> delta = 0; previous ema = 1.1 -> residual = 0.9
    np.testing.assert_allclose(feature[2, WORLD_STATE_DIM : 2 * WORLD_STATE_DIM], 0.0, atol=1e-6)
    np.testing.assert_allclose(feature[2, 2 * WORLD_STATE_DIM : 3 * WORLD_STATE_DIM], 0.9, atol=1e-5)


def test_rejects_wrong_world_state_dimension() -> None:
    with pytest.raises(ValueError):
        compute_features_offline(np.zeros((5, 10)))
    with pytest.raises(ValueError):
        StreamingFeatureState().step(np.zeros(10))
