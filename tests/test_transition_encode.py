"""Unit tests for the Stage 3 world-state encoding pipeline (issue #21).

Every test here is pure: synthetic tensors, synthetic manifests and boundaries, and fakes for the
frame reader and the encoder. None of this touches torch.hub, a real V-JEPA 2 checkpoint, a GPU, or
real video -- that is exercised separately by a real, `live`-marked smoke test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from blindsight.transition.corpus import ManifestClip, ProxyBoundary
from blindsight.transition.encode import (
    CacheMetadata,
    CostBudgetExceededError,
    ExtractionCostEstimate,
    ExtractionWorkItem,
    ProvenanceMismatchError,
    WINDOW_CONFIGS,
    assemble_run_record,
    boundary_context_range,
    cache_file_stem,
    center_crop,
    emission_timestamps,
    enforce_cost_budget,
    estimate_extraction_cost,
    merge_ranges,
    normalize_imagenet,
    plan_extraction_work_items,
    preprocess_frames,
    read_cache_metadata,
    resize_short_side,
    run_extraction_work_item,
    sanitize_identifier,
    trailing_window_frame_times,
    write_world_state_cache,
)


def _clip(
    *,
    corpus: str = "ego4d",
    source_video_id: str = "video-1",
    clip_id: str = "clip-1",
    split: str = "train",
) -> ManifestClip:
    return ManifestClip(
        corpus=corpus,  # type: ignore[arg-type]
        source_video_id=source_video_id,
        clip_id=clip_id,
        split=split,  # type: ignore[arg-type]
        evaluation_runs=("A",),
        boundary_count=1,
        resolution_status="clip-file",
        resolution_reason=None,
        extraction_strategy="download-540p-clip",
    )


def _boundary(
    *,
    corpus: str = "ego4d",
    source_video_id: str = "video-1",
    clip_id: str = "clip-1",
    boundary_time: float,
) -> ProxyBoundary:
    return ProxyBoundary(
        corpus=corpus,  # type: ignore[arg-type]
        source_video_id=source_video_id,
        clip_id=clip_id,
        boundary_time=boundary_time,
        prior_room_label="kitchen",
        prior_room_instance="1",
        next_room_label="bedroom",
        next_room_instance="1",
        family="indoor-to-indoor",
    )


# --------------------------------------------------------------------------------------------
# Window / timestamp math
# --------------------------------------------------------------------------------------------


class TestTrailingWindowCausality:
    @pytest.mark.parametrize("window_name", ["short", "long"])
    @pytest.mark.parametrize("emission_time", [0.0, 0.5, 1.0, 2.0, 5.0, 16.0, 100.0, 987.25])
    def test_every_sampled_frame_is_at_or_before_emission_time(
        self, window_name: str, emission_time: float
    ) -> None:
        window = WINDOW_CONFIGS[window_name]

        frame_times = trailing_window_frame_times(emission_time, window)

        assert len(frame_times) == window.frame_count
        assert all(t <= emission_time for t in frame_times)

    @pytest.mark.parametrize("window_name", ["short", "long"])
    def test_frame_times_are_non_decreasing(self, window_name: str) -> None:
        window = WINDOW_CONFIGS[window_name]

        frame_times = trailing_window_frame_times(50.0, window)

        assert frame_times == sorted(frame_times)

    def test_short_window_spacing_matches_30fps(self) -> None:
        window = WINDOW_CONFIGS["short"]

        frame_times = trailing_window_frame_times(50.0, window)

        assert frame_times[-1] == pytest.approx(50.0)
        assert frame_times[-2] == pytest.approx(50.0 - 1 / 30)

    def test_long_window_spacing_matches_4fps(self) -> None:
        window = WINDOW_CONFIGS["long"]

        frame_times = trailing_window_frame_times(50.0, window)

        assert frame_times[-1] == pytest.approx(50.0)
        assert frame_times[-2] == pytest.approx(50.0 - 1 / 4)

    def test_span_seconds_matches_specification(self) -> None:
        assert WINDOW_CONFIGS["short"].span_seconds == pytest.approx(64 / 30)
        assert WINDOW_CONFIGS["long"].span_seconds == pytest.approx(16.0)

    @pytest.mark.parametrize("window_name", ["short", "long"])
    def test_warm_up_near_clip_start_clamps_instead_of_fabricating(self, window_name: str) -> None:
        """A full window is not available near t=0. Every clamped sample must still be <= t, and
        the earliest available frame (t=0) must be the one that gets repeated."""

        window = WINDOW_CONFIGS[window_name]

        frame_times = trailing_window_frame_times(0.0, window)

        assert frame_times == [0.0] * window.frame_count

    def test_negative_emission_time_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            trailing_window_frame_times(-1.0, WINDOW_CONFIGS["short"])


class TestEmissionTimestamps:
    def test_one_hz_grid_over_a_range(self) -> None:
        result = emission_timestamps(10.0, 13.0)

        assert result == [10.0, 11.0, 12.0, 13.0]

    def test_zero_length_range_emits_one_timestamp(self) -> None:
        assert emission_timestamps(5.0, 5.0) == [5.0]

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError):
            emission_timestamps(5.0, 4.0)

    def test_custom_emission_rate(self) -> None:
        result = emission_timestamps(0.0, 1.0, emission_hz=2.0)

        assert result == pytest.approx([0.0, 0.5, 1.0])


# --------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------


class TestPreprocessing:
    def test_resize_short_side_to_438(self) -> None:
        frames = torch.rand(3, 3, 300, 500)

        resized = resize_short_side(frames)

        assert resized.shape[0] == 3
        assert min(resized.shape[-2:]) == 438

    def test_resize_short_side_handles_portrait_orientation(self) -> None:
        frames = torch.rand(2, 3, 500, 300)

        resized = resize_short_side(frames)

        assert resized.shape[-1] == 438

    def test_center_crop_produces_exact_size(self) -> None:
        frames = torch.rand(4, 3, 438, 600)

        cropped = center_crop(frames)

        assert cropped.shape[-2:] == (384, 384)

    def test_center_crop_rejects_too_small_input(self) -> None:
        with pytest.raises(ValueError):
            center_crop(torch.rand(1, 3, 300, 300))

    def test_normalize_imagenet_shifts_and_scales(self) -> None:
        frames = torch.full((1, 3, 4, 4), 0.5)

        normalized = normalize_imagenet(frames)

        expected = (0.5 - 0.456) / 0.224  # green channel
        assert normalized[0, 1, 0, 0].item() == pytest.approx(expected, abs=1e-4)

    def test_full_pipeline_shape(self) -> None:
        frames = torch.rand(64, 3, 480, 640)

        result = preprocess_frames(frames)

        assert result.shape == (64, 3, 384, 384)

    def test_full_pipeline_preserves_frame_count_for_odd_sizes(self) -> None:
        frames = torch.rand(5, 3, 90, 160)

        result = preprocess_frames(frames)

        assert result.shape == (5, 3, 384, 384)


# --------------------------------------------------------------------------------------------
# Extraction planning
# --------------------------------------------------------------------------------------------


class TestBoundaryContextRange:
    def test_range_is_symmetric_margin_around_boundary(self) -> None:
        result = boundary_context_range(
            100.0, guard_band_seconds=1.0, positive_seconds=4.0, clip_duration_seconds=1000.0
        )

        assert result.start_seconds == pytest.approx(100.0 - 13.0)
        assert result.end_seconds == pytest.approx(100.0 + 13.0)

    def test_range_clamps_to_clip_bounds(self) -> None:
        result = boundary_context_range(
            2.0, guard_band_seconds=1.0, positive_seconds=4.0, clip_duration_seconds=10.0
        )

        assert result.start_seconds == 0.0
        assert result.end_seconds == 10.0


class TestMergeRanges:
    def test_merges_overlapping_ranges(self) -> None:
        from blindsight.transition.encode import ExtractionRange

        merged = merge_ranges([ExtractionRange(0, 5), ExtractionRange(3, 8)])

        assert merged == [ExtractionRange(0, 8)]

    def test_keeps_disjoint_ranges_separate(self) -> None:
        from blindsight.transition.encode import ExtractionRange

        merged = merge_ranges([ExtractionRange(0, 5), ExtractionRange(100, 110)])

        assert merged == [ExtractionRange(0, 5), ExtractionRange(100, 110)]

    def test_empty_input(self) -> None:
        assert merge_ranges([]) == []


class TestSanitizeIdentifier:
    def test_replaces_slashes(self) -> None:
        assert sanitize_identifier("abc/def_012_139") == "abc-def_012_139"

    def test_leaves_safe_characters_alone(self) -> None:
        assert sanitize_identifier("abc-def_012.139") == "abc-def_012.139"


class TestPlanExtractionWorkItems:
    def test_heldout_clip_gets_one_full_clip_item(self) -> None:
        clip = _clip(split="heldout")

        items = plan_extraction_work_items(
            clip, [], clip_duration_seconds=10.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        assert len(items) == 1
        item = items[0]
        assert item.range_start_seconds == 0.0
        assert item.range_end_seconds == 10.0
        assert item.emission_timestamps == tuple(emission_timestamps(0.0, 10.0))
        assert item.file_stem == "clip-1"

    def test_test_split_clip_also_gets_full_clip_item(self) -> None:
        clip = _clip(split="test")

        items = plan_extraction_work_items(
            clip, [], clip_duration_seconds=6.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        assert len(items) == 1
        assert items[0].range_end_seconds == 6.0

    def test_not_selected_clip_plans_nothing(self) -> None:
        clip = _clip(split="not-selected")

        items = plan_extraction_work_items(
            clip, [], clip_duration_seconds=100.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        assert items == []

    def test_train_clip_with_no_boundaries_plans_nothing(self) -> None:
        clip = _clip(split="train")

        items = plan_extraction_work_items(
            clip, [], clip_duration_seconds=100.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        assert items == []

    def test_train_clip_with_one_boundary_gets_a_boundary_window(self) -> None:
        clip = _clip(split="train")
        boundary = _boundary(boundary_time=500.0)

        items = plan_extraction_work_items(
            clip,
            [boundary],
            clip_duration_seconds=1000.0,
            guard_band_seconds=1.0,
            positive_seconds=4.0,
            gru_history_seconds=8.0,
        )

        assert len(items) == 1
        item = items[0]
        assert item.file_stem == "clip-1__boundary0"
        assert item.range_start_seconds == pytest.approx(500.0 - 13.0)
        assert item.range_end_seconds == pytest.approx(500.0 + 13.0)

    def test_train_clip_with_close_boundaries_gets_one_merged_window(self) -> None:
        clip = _clip(split="train")
        boundaries = [_boundary(boundary_time=100.0), _boundary(boundary_time=110.0)]

        items = plan_extraction_work_items(
            clip, boundaries, clip_duration_seconds=1000.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        assert len(items) == 1
        assert items[0].range_start_seconds == pytest.approx(87.0)
        assert items[0].range_end_seconds == pytest.approx(123.0)

    def test_train_clip_with_far_apart_boundaries_gets_separate_windows(self) -> None:
        clip = _clip(split="train")
        boundaries = [_boundary(boundary_time=100.0), _boundary(boundary_time=800.0)]

        items = plan_extraction_work_items(
            clip, boundaries, clip_duration_seconds=1000.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        assert len(items) == 2
        assert items[0].file_stem == "clip-1__boundary0"
        assert items[1].file_stem == "clip-1__boundary1"
        # Each item's range covers only its own boundary, never the whole clip.
        assert items[0].range_end_seconds < items[1].range_start_seconds

    def test_boundaries_from_a_different_clip_are_ignored(self) -> None:
        clip = _clip(split="train", clip_id="clip-1")
        other_clip_boundary = _boundary(clip_id="clip-2", boundary_time=50.0)

        items = plan_extraction_work_items(
            clip,
            [other_clip_boundary],
            clip_duration_seconds=1000.0,
            guard_band_seconds=1.0,
            positive_seconds=4.0,
        )

        assert items == []

    def test_every_planned_timestamp_is_within_the_clip_and_causal_window_is_valid(self) -> None:
        """Every timestamp this plans must itself be a legal, causal emission point: the trailing
        window it implies never reaches past the timestamp itself (see TestTrailingWindowCausality)."""

        clip = _clip(split="heldout")

        items = plan_extraction_work_items(
            clip, [], clip_duration_seconds=25.0, guard_band_seconds=1.0, positive_seconds=4.0
        )

        for timestamp in items[0].emission_timestamps:
            assert 0.0 <= timestamp <= 25.0
            for window in WINDOW_CONFIGS.values():
                assert all(t <= timestamp for t in trailing_window_frame_times(timestamp, window))


class TestCostEstimate:
    def test_estimate_scales_with_planned_world_states(self) -> None:
        items = [
            _work_item(emission_timestamps=(0.0, 1.0, 2.0)),
            _work_item(clip_id="clip-2", file_stem="clip-2", emission_timestamps=(0.0, 1.0)),
        ]

        estimate = estimate_extraction_cost(items)

        assert estimate.work_item_count == 2
        assert estimate.world_state_count == 5
        assert estimate.estimated_gpu_seconds == pytest.approx(5 * 0.30)
        assert estimate.estimated_retained_bytes == 5 * (1024 * 4 + 8)

    def test_enforce_cost_budget_passes_within_budget(self) -> None:
        estimate = estimate_extraction_cost([_work_item(emission_timestamps=(0.0,))])

        enforce_cost_budget(estimate, max_gpu_seconds=10.0, max_retained_bytes=10_000)

    def test_enforce_cost_budget_rejects_gpu_seconds_over_budget(self) -> None:
        estimate = estimate_extraction_cost(
            [_work_item(emission_timestamps=tuple(float(i) for i in range(1000)))]
        )

        with pytest.raises(CostBudgetExceededError):
            enforce_cost_budget(estimate, max_gpu_seconds=1.0)

    def test_enforce_cost_budget_rejects_bytes_over_budget(self) -> None:
        estimate = estimate_extraction_cost(
            [_work_item(emission_timestamps=tuple(float(i) for i in range(1000)))]
        )

        with pytest.raises(CostBudgetExceededError):
            enforce_cost_budget(estimate, max_retained_bytes=1)

    def test_enforce_cost_budget_with_no_bounds_never_raises(self) -> None:
        estimate = estimate_extraction_cost(
            [_work_item(emission_timestamps=tuple(float(i) for i in range(1000)))]
        )

        enforce_cost_budget(estimate)


# --------------------------------------------------------------------------------------------
# Run record assembly: the measured wall clock around the fan-out (issue #21's final record).
# --------------------------------------------------------------------------------------------


def _batch_result(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {"clip_id": "clip-1", "status": "completed", "items": [], "results": []}
    base.update(overrides)
    return base


class TestAssembleRunRecord:
    def _assemble(
        self,
        batch_results: list[dict[str, object]],
        *,
        estimate: ExtractionCostEstimate | None = None,
    ) -> dict[str, object]:
        if estimate is None:
            estimate = estimate_extraction_cost([_work_item(emission_timestamps=(0.0, 1.0, 2.0))])
        return assemble_run_record(
            manifest_name="ego4d-only-v1",
            corpus="ego4d",
            window_config="short",
            estimate=estimate,
            batch_count=1,
            clip_count=1,
            fan_out=lambda: batch_results,
        )

    def test_success_record_has_non_negative_wall_seconds(self) -> None:
        record = self._assemble([_batch_result()])

        assert "wall_seconds" in record
        assert record["wall_seconds"] >= 0.0

    def test_failure_record_also_records_wall_seconds(self) -> None:
        record = self._assemble(
            [
                _batch_result(
                    status="completed",
                    results=[{"clip_id": "clip-1", "status": "failed", "error": "no source video found"}],
                )
            ]
        )

        assert "wall_seconds" in record
        assert record["wall_seconds"] >= 0.0
        assert record["failed_count"] == 1
        assert record["failed"] == [{"clip_id": "clip-1", "error": "no source video found"}]

    def test_wall_seconds_measures_the_fan_out_itself(self) -> None:
        def slow_fan_out() -> list[dict[str, object]]:
            time.sleep(0.05)
            return [_batch_result()]

        estimate = estimate_extraction_cost([_work_item(emission_timestamps=(0.0, 1.0, 2.0))])
        record = assemble_run_record(
            manifest_name="ego4d-only-v1",
            corpus="ego4d",
            window_config="short",
            estimate=estimate,
            batch_count=1,
            clip_count=1,
            fan_out=slow_fan_out,
        )

        assert record["wall_seconds"] >= 0.04

    def test_record_keeps_estimated_fields_unchanged(self) -> None:
        estimate = estimate_extraction_cost([_work_item(emission_timestamps=(0.0, 1.0, 2.0))])

        record = self._assemble([_batch_result()], estimate=estimate)

        assert record["estimated_gpu_seconds"] == estimate.estimated_gpu_seconds
        assert record["estimated_retained_bytes"] == estimate.estimated_retained_bytes


# --------------------------------------------------------------------------------------------
# Cache format contract
# --------------------------------------------------------------------------------------------


def _metadata(**overrides: object) -> CacheMetadata:
    base = dict(
        corpus="ego4d",
        source_video_id="video-1",
        clip_id="clip-1",
        split="heldout",
        window_config="short",
        manifest_name="ego4d-only-v1",
        guard_band_seconds=1.0,
        positive_seconds=4.0,
        encoder_commit="204698b45b3712590f06245fbfba32d3be539812",
        checkpoint_digest="deadbeef",
        range_start_seconds=0.0,
        range_end_seconds=10.0,
    )
    base.update(overrides)
    return CacheMetadata(**base)  # type: ignore[arg-type]


class TestCacheFormat:
    def test_writes_expected_array_names_and_dtypes(self, tmp_path: Path) -> None:
        world_states = np.zeros((3, 1024), dtype=np.float32)
        timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        npz_path, json_path = write_world_state_cache(
            tmp_path, "clip-1__short", world_states=world_states, timestamps=timestamps, metadata=_metadata()
        )

        loaded = np.load(npz_path)
        assert set(loaded.files) == {"world_states", "timestamps"}
        assert loaded["world_states"].shape == (3, 1024)
        assert loaded["world_states"].dtype == np.float32
        assert loaded["timestamps"].dtype == np.float64
        assert list(loaded["timestamps"]) == [0.0, 1.0, 2.0]

        metadata_on_disk = json.loads(json_path.read_text(encoding="utf-8"))
        for key in (
            "corpus",
            "source_video_id",
            "clip_id",
            "split",
            "window_config",
            "manifest_name",
            "guard_band_seconds",
            "positive_seconds",
            "encoder_commit",
            "checkpoint_digest",
            "range_start_seconds",
            "range_end_seconds",
        ):
            assert key in metadata_on_disk

    def test_rejects_wrong_world_state_width(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_world_state_cache(
                tmp_path,
                "bad",
                world_states=np.zeros((3, 7), dtype=np.float32),
                timestamps=np.array([0.0, 1.0, 2.0], dtype=np.float64),
                metadata=_metadata(),
            )

    def test_rejects_non_increasing_timestamps(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_world_state_cache(
                tmp_path,
                "bad",
                world_states=np.zeros((2, 1024), dtype=np.float32),
                timestamps=np.array([1.0, 1.0], dtype=np.float64),
                metadata=_metadata(),
            )

    def test_file_naming_pattern(self, tmp_path: Path) -> None:
        assert cache_file_stem("clip-1", "short") == "clip-1__short"
        assert cache_file_stem("clip-1__boundary0", "long") == "clip-1__boundary0__long"

    def test_read_cache_metadata_returns_none_when_missing(self, tmp_path: Path) -> None:
        assert read_cache_metadata(tmp_path / "missing.json") is None


# --------------------------------------------------------------------------------------------
# Work-item orchestration: idempotency and provenance, with fakes standing in for the real
# GPU-backed frame reader and encoder.
# --------------------------------------------------------------------------------------------


def _fake_read_frames(frame_times: list[float]) -> torch.Tensor:
    return torch.rand(len(frame_times), 3, 64, 64)


def _fake_encode_window(preprocessed: torch.Tensor) -> torch.Tensor:
    # Deterministic given the window's mean pixel value, so re-encoding the same input is stable.
    return torch.full((1024,), float(preprocessed.mean()))


def _work_item(**overrides: object) -> ExtractionWorkItem:
    base: dict[str, object] = dict(
        corpus="ego4d",
        source_video_id="video-1",
        clip_id="clip-1",
        split="heldout",
        range_start_seconds=0.0,
        range_end_seconds=2.0,
        emission_timestamps=(0.0, 1.0, 2.0),
        file_stem="clip-1",
    )
    base.update(overrides)
    return ExtractionWorkItem(**base)  # type: ignore[arg-type]


class TestRunExtractionWorkItem:
    def _run(self, item: ExtractionWorkItem, cache_dir: Path, **kwargs: object) -> dict[str, object]:
        defaults: dict[str, object] = dict(
            window=WINDOW_CONFIGS["short"],
            read_frames=_fake_read_frames,
            encode_window=_fake_encode_window,
            cache_dir=cache_dir,
            manifest_name="ego4d-only-v1",
            encoder_commit="204698b45b3712590f06245fbfba32d3be539812",
            checkpoint_digest="deadbeef",
            guard_band_seconds=1.0,
            positive_seconds=4.0,
        )
        defaults.update(kwargs)
        return run_extraction_work_item(item, **defaults)  # type: ignore[arg-type]

    def test_encodes_and_writes_a_valid_cache_pair(self, tmp_path: Path) -> None:
        item = _work_item()

        result = self._run(item, tmp_path)

        assert result["status"] == "encoded"
        loaded = np.load(tmp_path / "clip-1__short.npz")
        assert loaded["world_states"].shape == (3, 1024)
        assert list(loaded["timestamps"]) == [0.0, 1.0, 2.0]

    def test_repeating_a_completed_work_item_is_skipped(self, tmp_path: Path) -> None:
        item = _work_item()
        self._run(item, tmp_path)
        npz_path = tmp_path / "clip-1__short.npz"
        written_at = npz_path.stat().st_mtime_ns

        result = self._run(item, tmp_path)

        assert result["status"] == "skipped-idempotent"
        assert npz_path.stat().st_mtime_ns == written_at

    def test_provenance_mismatch_raises_instead_of_overwriting(self, tmp_path: Path) -> None:
        item = _work_item()
        self._run(item, tmp_path)

        with pytest.raises(ProvenanceMismatchError):
            self._run(item, tmp_path, manifest_name="a-different-manifest")

        # The original cache file must survive untouched.
        loaded = np.load(tmp_path / "clip-1__short.npz")
        assert loaded["world_states"].shape == (3, 1024)

    def test_missing_data_file_with_matching_metadata_re_encodes(self, tmp_path: Path) -> None:
        item = _work_item()
        self._run(item, tmp_path)
        (tmp_path / "clip-1__short.npz").unlink()

        result = self._run(item, tmp_path)

        assert result["status"] == "encoded"
        assert (tmp_path / "clip-1__short.npz").exists()

    def test_boundary_work_item_uses_range_relative_timestamps(self, tmp_path: Path) -> None:
        item = _work_item(
            split="train",
            range_start_seconds=100.0,
            range_end_seconds=102.0,
            emission_timestamps=(100.0, 101.0, 102.0),
            file_stem="clip-1__boundary0",
        )

        self._run(item, tmp_path)

        loaded = np.load(tmp_path / "clip-1__boundary0__short.npz")
        assert list(loaded["timestamps"]) == [0.0, 1.0, 2.0]
