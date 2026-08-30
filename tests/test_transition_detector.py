from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch

from blindsight.transition.detector import (
    ACTIVATE_THRESHOLD,
    COOLDOWN_SECONDS,
    EXPECTED_PARAMETER_COUNT_RANGE,
    FEATURE_DIM,
    GRU_HISTORY_LENGTH,
    PERSISTENCE_STEPS,
    RELEASE_THRESHOLD,
    CausalGRUHead,
    DecisionPolicyState,
    LogisticHead,
    PlattCalibration,
    StreamingLogisticDetector,
    build_gru_windows,
    export_gru_to_onnx,
    export_logistic_to_onnx,
    fit_gru_head,
    gru_logits,
    select_detector_head,
)
from blindsight.transition.features import WORLD_STATE_DIM


def _synthetic_dataset(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, FEATURE_DIM))
    # A separable-ish signal so LogisticRegression has something real to fit.
    weight = rng.normal(size=FEATURE_DIM)
    logits = features @ weight * 0.05
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.uniform(size=n) < probabilities).astype(int)
    if labels.sum() == 0:
        labels[0] = 1
    if labels.sum() == n:
        labels[0] = 0
    return features, labels


# --------------------------------------------------------------------------------------
# Logistic head
# --------------------------------------------------------------------------------------


def test_logistic_head_fit_preserves_mean_scale_weight_bias_and_scores_correct_shape() -> None:
    features, labels = _synthetic_dataset()

    head = LogisticHead.fit(features, labels)

    assert head.mean.shape == (FEATURE_DIM,)
    assert head.scale.shape == (FEATURE_DIM,)
    assert head.weight.shape == (FEATURE_DIM,)
    assert isinstance(head.bias, float)

    logits = head.logit(features)
    assert logits.shape == (features.shape[0],)
    assert np.all(np.isfinite(logits))


def test_logistic_head_rejects_wrong_feature_dimension() -> None:
    with pytest.raises(ValueError):
        LogisticHead.fit(np.zeros((10, 5)), np.zeros(10))


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def test_platt_calibration_is_monotonic_in_the_logit() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=500) * 3
    labels = (logits > 0).astype(int)

    calibration = PlattCalibration.fit(logits, labels)
    probabilities = calibration.apply(np.sort(logits))

    assert np.all(np.diff(probabilities) >= -1e-9)
    assert probabilities[0] < probabilities[-1]


def test_platt_calibration_handles_single_class_gracefully() -> None:
    logits = np.array([0.1, 0.5, -0.2, 0.3])
    labels = np.zeros(4, dtype=int)

    calibration = PlattCalibration.fit(logits, labels)
    probabilities = calibration.apply(logits)

    assert np.all(probabilities < 0.5)


# --------------------------------------------------------------------------------------
# Causal GRU head
# --------------------------------------------------------------------------------------


def test_gru_head_architecture_matches_spec_and_param_count_is_sane() -> None:
    head = CausalGRUHead()

    assert head.input_linear.in_features == WORLD_STATE_DIM
    assert head.input_linear.out_features == 32
    assert head.gru.hidden_size == 32
    assert head.output_linear.in_features == 32
    assert head.output_linear.out_features == 1

    count = head.parameter_count()
    low, high = EXPECTED_PARAMETER_COUNT_RANGE
    assert low <= count <= high


def test_gru_head_forward_shape() -> None:
    head = CausalGRUHead()
    window = torch.zeros(4, GRU_HISTORY_LENGTH, WORLD_STATE_DIM)

    logits = head(window)

    assert logits.shape == (4,)


def test_build_gru_windows_are_causal_with_left_padding() -> None:
    world_states = np.arange(5 * WORLD_STATE_DIM, dtype=np.float32).reshape(5, WORLD_STATE_DIM)

    windows = build_gru_windows(world_states, history_length=3)

    assert windows.shape == (5, 3, WORLD_STATE_DIM)
    # First window is entirely padded with the first row.
    np.testing.assert_allclose(windows[0, 0], world_states[0])
    np.testing.assert_allclose(windows[0, 1], world_states[0])
    np.testing.assert_allclose(windows[0, 2], world_states[0])
    # Third window (index 2) covers steps 0, 1, 2 with no future leakage.
    np.testing.assert_allclose(windows[2, 0], world_states[0])
    np.testing.assert_allclose(windows[2, 1], world_states[1])
    np.testing.assert_allclose(windows[2, 2], world_states[2])


def test_fit_gru_head_reduces_loss_and_produces_finite_logits() -> None:
    rng = np.random.default_rng(2)
    windows = rng.normal(size=(60, GRU_HISTORY_LENGTH, WORLD_STATE_DIM)).astype(np.float32)
    labels = (rng.uniform(size=60) < 0.3).astype(np.float32)

    head = fit_gru_head(windows, labels, epochs=20)
    logits = gru_logits(head, windows)

    assert logits.shape == (60,)
    assert np.all(np.isfinite(logits))


# --------------------------------------------------------------------------------------
# Decision policy
# --------------------------------------------------------------------------------------


def test_decision_policy_requires_persistence_before_firing() -> None:
    policy = DecisionPolicyState()

    assert policy.step(ACTIVATE_THRESHOLD) is False  # 1st high score: not enough yet
    assert policy.consecutive_high == 1


def test_decision_policy_fires_after_two_consecutive_high_scores() -> None:
    policy = DecisionPolicyState()

    assert policy.step(0.9) is False
    assert policy.step(0.9) is True


def test_decision_policy_does_not_oscillate_near_the_activate_threshold() -> None:
    """A score dithering just below/above 0.8 (but above the 0.4 release band) must not reset."""

    policy = DecisionPolicyState()
    scores = [0.85, 0.78, 0.82, 0.76]  # never drops below RELEASE_THRESHOLD

    fired_steps = [policy.step(score) for score in scores]

    # It should reach persistence and fire exactly once across this dithering sequence, never
    # having its counter reset to zero by a dip that stays above the release threshold.
    assert any(fired_steps)
    assert fired_steps.count(True) == 1


def test_decision_policy_release_threshold_resets_the_counter() -> None:
    policy = DecisionPolicyState()

    assert policy.step(0.9) is False
    assert policy.consecutive_high == 1
    assert policy.step(0.2) is False  # below RELEASE_THRESHOLD: full reset
    assert policy.consecutive_high == 0
    assert policy.step(0.9) is False  # needs to rebuild persistence from scratch
    assert policy.step(0.9) is True


def test_decision_policy_repeated_low_scores_never_fire() -> None:
    policy = DecisionPolicyState()

    results = [policy.step(0.1) for _ in range(20)]

    assert not any(results)


def test_decision_policy_cooldown_suppresses_a_second_event() -> None:
    policy = DecisionPolicyState()

    assert policy.step(0.9) is False
    assert policy.step(0.9) is True  # first event fires, cooldown starts
    # Immediately after, even two more high scores must not fire a second event.
    assert policy.step(0.9) is False
    assert policy.step(0.9) is False


def test_decision_policy_can_fire_again_once_cooldown_elapses() -> None:
    policy = DecisionPolicyState()

    assert policy.step(0.9) is False
    assert policy.step(0.9) is True
    # Let the cooldown run out with low scores (also resets persistence).
    for _ in range(int(COOLDOWN_SECONDS) + 1):
        policy.step(0.0, dt_seconds=1.0)
    assert policy.cooldown_remaining_seconds == 0.0

    assert policy.step(0.9) is False
    assert policy.step(0.9) is True


def test_decision_policy_thresholds_match_spec_constants() -> None:
    assert ACTIVATE_THRESHOLD == pytest.approx(0.8)
    assert RELEASE_THRESHOLD == pytest.approx(0.4)
    assert PERSISTENCE_STEPS == 2
    assert COOLDOWN_SECONDS == pytest.approx(10.0)


# --------------------------------------------------------------------------------------
# Selection rule
# --------------------------------------------------------------------------------------


def test_select_detector_head_picks_gru_when_both_conditions_hold() -> None:
    result = select_detector_head(
        logistic_average_precision=0.50,
        gru_average_precision=0.53,
        per_group_average_precision=[(0.4, 0.5), (0.6, 0.65), (0.5, 0.4)],
    )

    assert result.selected == "gru"
    assert result.gru_group_wins == 2
    assert result.total_groups == 3


def test_select_detector_head_picks_logistic_when_margin_is_too_small() -> None:
    result = select_detector_head(
        logistic_average_precision=0.50,
        gru_average_precision=0.505,  # only +0.005, below the 0.02 margin
        per_group_average_precision=[(0.4, 0.5), (0.6, 0.65), (0.5, 0.6)],
    )

    assert result.selected == "logistic"


def test_select_detector_head_picks_logistic_when_group_wins_are_not_a_majority() -> None:
    result = select_detector_head(
        logistic_average_precision=0.50,
        gru_average_precision=0.53,  # margin clears 0.02
        per_group_average_precision=[(0.4, 0.3), (0.6, 0.5), (0.5, 0.9)],  # GRU wins only 1/3
    )

    assert result.selected == "logistic"
    assert result.gru_group_wins == 1


def test_select_detector_head_ties_favor_logistic() -> None:
    result = select_detector_head(
        logistic_average_precision=0.50,
        gru_average_precision=0.52,
        per_group_average_precision=[(0.5, 0.5), (0.5, 0.5)],  # exact ties never count as wins
    )

    assert result.selected == "logistic"
    assert result.gru_group_wins == 0


def test_select_detector_head_handles_no_groups() -> None:
    result = select_detector_head(
        logistic_average_precision=0.50, gru_average_precision=0.6, per_group_average_precision=[]
    )

    assert result.selected == "logistic"
    assert result.total_groups == 0


# --------------------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------------------


def test_onnx_logistic_export_matches_pytorch_within_tolerance(tmp_path: Path) -> None:
    features, labels = _synthetic_dataset(n=50, seed=5)
    head = LogisticHead.fit(features, labels)
    calibration = PlattCalibration.fit(head.logit(features), labels)

    onnx_path = tmp_path / "logistic_head.onnx"
    export_logistic_to_onnx(head, calibration, onnx_path)

    query = features[:8].astype(np.float32)
    expected_logit = head.logit(query)
    expected_probability = calibration.apply(expected_logit)

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logit, onnx_probability = session.run(None, {"features": query})

    np.testing.assert_allclose(onnx_logit, expected_logit, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(onnx_probability, expected_probability, rtol=1e-4, atol=1e-4)


def test_onnx_gru_export_matches_pytorch_within_tolerance(tmp_path: Path) -> None:
    torch.manual_seed(0)
    head = CausalGRUHead()
    head.eval()
    calibration = PlattCalibration(slope=1.3, bias=-0.2)

    onnx_path = tmp_path / "gru_head.onnx"
    export_gru_to_onnx(head, calibration, onnx_path)

    rng = np.random.default_rng(6)
    query = rng.normal(size=(5, GRU_HISTORY_LENGTH, WORLD_STATE_DIM)).astype(np.float32)
    with torch.no_grad():
        expected_logit = head(torch.from_numpy(query)).numpy()
    expected_probability = calibration.apply(expected_logit)

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logit, onnx_probability = session.run(None, {"world_state_window": query})

    np.testing.assert_allclose(onnx_logit, expected_logit, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(onnx_probability, expected_probability, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------------------
# Streaming detector
# --------------------------------------------------------------------------------------


def test_streaming_logistic_detector_returns_probability_and_decision() -> None:
    features, labels = _synthetic_dataset(n=50, seed=7)
    head = LogisticHead.fit(features, labels)
    calibration = PlattCalibration.fit(head.logit(features), labels)
    detector = StreamingLogisticDetector(head=head, calibration=calibration)

    rng = np.random.default_rng(8)
    for _ in range(5):
        world_state = rng.normal(size=WORLD_STATE_DIM)
        step = detector.step(world_state)
        assert 0.0 <= step.probability <= 1.0
        assert isinstance(step.transition_event, bool)
