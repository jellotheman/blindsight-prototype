"""Tests for train_transition_detector.py against synthetic cached-feature fixtures.

Real cached world-state files (produced by the sibling encoding pipeline) and real EgoEnv
annotation CSVs are not expected to exist in this checkout yet. Every fixture here is hand-built:
tiny `.npz`/`.json` cache pairs following the shared cache-format contract, and small RoomPred CSVs
following `blindsight/transition/corpus.py`'s existing loaders.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from blindsight.transition.corpus import IGNORE_LABEL, ProxyBoundary, RoomInterval
from blindsight.transition.features import WORLD_STATE_DIM
from train_transition_detector import (
    CachedClip,
    ClipFeatures,
    ManifestClipRef,
    cache_file_pairs_for_clip,
    cache_paths_for_clip,
    compute_group_average_precisions,
    evaluate_at_fixed_policy,
    label_cached_clip,
    load_cached_clip,
    load_manifest_clips,
    load_split_clip_features,
    run,
    sanitize_clip_id,
    stack_gru_rows,
    stack_labeled_rows,
)


def test_sanitize_clip_id_replaces_unsafe_characters() -> None:
    # Must match blindsight/transition/encode.py's sanitize_identifier exactly: unsafe characters
    # become "-", not "_" -- the two pipelines write and read the same cache directory.
    assert sanitize_clip_id("abc/def:ghi") == "abc-def-ghi"
    assert sanitize_clip_id("already-safe_123.mp4") == "already-safe_123.mp4"


def test_cache_paths_for_clip_matches_the_shared_contract_layout(tmp_path: Path) -> None:
    npz_path, json_path = cache_paths_for_clip(tmp_path, "ego4d-only-v1", "ego4d", "abc/def", "short")

    assert npz_path == tmp_path / "world-states" / "ego4d-only-v1" / "ego4d" / "abc-def__short.npz"
    assert json_path == tmp_path / "world-states" / "ego4d-only-v1" / "ego4d" / "abc-def__short.json"


def _write_cache_pair(
    directory: Path,
    *,
    corpus: str,
    source_video_id: str,
    clip_id: str,
    split: str,
    window_config: str,
    manifest_name: str,
    world_states: np.ndarray,
    timestamps: np.ndarray,
    range_start_seconds: float = 0.0,
    guard_band_seconds: float = 1.0,
    positive_seconds: float = 4.0,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{sanitize_clip_id(clip_id)}__{window_config}"
    npz_path = directory / f"{stem}.npz"
    json_path = directory / f"{stem}.json"
    np.savez(npz_path, world_states=world_states.astype(np.float32), timestamps=timestamps.astype(np.float64))
    json_path.write_text(
        json.dumps(
            {
                "corpus": corpus,
                "source_video_id": source_video_id,
                "clip_id": clip_id,
                "split": split,
                "window_config": window_config,
                "manifest_name": manifest_name,
                "guard_band_seconds": guard_band_seconds,
                "positive_seconds": positive_seconds,
                "encoder_commit": "deadbeef",
                "checkpoint_digest": "sha256:fake",
                "range_start_seconds": range_start_seconds,
                "range_end_seconds": range_start_seconds + float(timestamps[-1]),
            }
        )
    )
    return npz_path, json_path


def test_load_cached_clip_round_trips_arrays_and_metadata(tmp_path: Path) -> None:
    world_states = np.random.default_rng(0).normal(size=(6, WORLD_STATE_DIM))
    timestamps = np.arange(6, dtype=np.float64)
    npz_path, json_path = _write_cache_pair(
        tmp_path,
        corpus="ego4d",
        source_video_id="video-1",
        clip_id="clip-1",
        split="train",
        window_config="short",
        manifest_name="test-manifest",
        world_states=world_states,
        timestamps=timestamps,
    )

    cached = load_cached_clip(npz_path, json_path)

    assert cached.corpus == "ego4d"
    assert cached.clip_id == "clip-1"
    assert cached.split == "train"
    np.testing.assert_allclose(cached.world_states, world_states, atol=1e-5)
    np.testing.assert_allclose(cached.timestamps, timestamps)


def test_load_cached_clip_rejects_mismatched_shapes(tmp_path: Path) -> None:
    npz_path, json_path = _write_cache_pair(
        tmp_path,
        corpus="ego4d",
        source_video_id="video-1",
        clip_id="clip-1",
        split="train",
        window_config="short",
        manifest_name="test-manifest",
        world_states=np.zeros((5, WORLD_STATE_DIM)),
        timestamps=np.arange(4, dtype=np.float64),  # mismatched length
    )

    with pytest.raises(ValueError):
        load_cached_clip(npz_path, json_path)


def test_cache_file_pairs_for_clip_discovers_boundary_split_training_files(tmp_path: Path) -> None:
    # blindsight/transition/encode.py's plan_extraction_work_items gives a training clip with
    # multiple far-apart boundaries one file per boundary window (`<clip>__boundary<N>__<config>`),
    # never one whole-clip file. A loader that only checks the whole-clip name (the heldout/test
    # layout) would find nothing for every real training clip -- this is the regression this guards.
    directory = tmp_path / "world-states" / "test-manifest" / "ego4d"
    for index in range(3):
        _write_cache_pair(
            directory,
            corpus="ego4d",
            source_video_id="video-1",
            clip_id="clip-1",
            split="train",
            window_config="short",
            manifest_name="test-manifest",
            world_states=np.zeros((4, WORLD_STATE_DIM)),
            timestamps=np.arange(4, dtype=np.float64),
            range_start_seconds=float(index * 100),
        )
        # _write_cache_pair always names the file as the whole-clip layout; rename to the real
        # boundary-window layout encode.py actually produces so this test exercises real filenames.
        whole_npz = directory / "clip-1__short.npz"
        whole_json = directory / "clip-1__short.json"
        whole_npz.rename(directory / f"clip-1__boundary{index}__short.npz")
        whole_json.rename(directory / f"clip-1__boundary{index}__short.json")

    pairs = cache_file_pairs_for_clip(tmp_path, "test-manifest", "ego4d", "clip-1", "short")

    assert len(pairs) == 3
    assert [npz.name for npz, _ in pairs] == [
        "clip-1__boundary0__short.npz",
        "clip-1__boundary1__short.npz",
        "clip-1__boundary2__short.npz",
    ]


def test_load_split_clip_features_loads_every_boundary_window_for_a_training_clip(tmp_path: Path) -> None:
    directory = tmp_path / "world-states" / "test-manifest" / "ego4d"
    for index in range(2):
        _write_cache_pair(
            directory,
            corpus="ego4d",
            source_video_id="video-1",
            clip_id="clip-1",
            split="train",
            window_config="short",
            manifest_name="test-manifest",
            world_states=np.zeros((4, WORLD_STATE_DIM)),
            timestamps=np.arange(4, dtype=np.float64),
            range_start_seconds=float(index * 100),
        )
        (directory / "clip-1__short.npz").rename(directory / f"clip-1__boundary{index}__short.npz")
        (directory / "clip-1__short.json").rename(directory / f"clip-1__boundary{index}__short.json")
    # A heldout clip keeps the plain whole-clip layout.
    _write_cache_pair(
        directory,
        corpus="ego4d",
        source_video_id="video-2",
        clip_id="clip-2",
        split="heldout",
        window_config="short",
        manifest_name="test-manifest",
        world_states=np.zeros((4, WORLD_STATE_DIM)),
        timestamps=np.arange(4, dtype=np.float64),
    )
    manifest_clips = [
        ManifestClipRef(corpus="ego4d", source_video_id="video-1", clip_id="clip-1", split="train"),
        ManifestClipRef(corpus="ego4d", source_video_id="video-2", clip_id="clip-2", split="heldout"),
    ]

    loaded, missing = load_split_clip_features(
        manifest_clips,
        "train",
        cache_root=tmp_path,
        manifest_name="test-manifest",
        window_config="short",
        intervals=[],
        boundaries=[],
    )

    assert missing == []
    assert len(loaded) == 2  # both boundary-window files for clip-1, none of clip-2's


def test_label_cached_clip_shifts_into_the_clips_absolute_time_frame() -> None:
    # The cache covers a boundary window starting 10s into the clip; timestamps are local to that
    # range, so a boundary at absolute time 15 appears at local time 5.
    cached = CachedClip(
        corpus="ego4d",
        source_video_id="video-1",
        clip_id="clip-1",
        split="heldout",
        window_config="short",
        manifest_name="m",
        guard_band_seconds=1.0,
        positive_seconds=4.0,
        range_start_seconds=10.0,
        range_end_seconds=20.0,
        world_states=np.zeros((10, WORLD_STATE_DIM)),
        timestamps=np.arange(10, dtype=np.float64),
    )
    intervals = [
        RoomInterval("ego4d", "video-1", "clip-1", 0.0, 15.0, "kitchen", "1"),
        RoomInterval("ego4d", "video-1", "clip-1", 15.0, 40.0, "living_room", "1"),
    ]
    boundaries = [
        ProxyBoundary("ego4d", "video-1", "clip-1", 15.0, "kitchen", "1", "living_room", "1", "indoor-to-indoor")
    ]

    labels = label_cached_clip(cached, intervals, boundaries)

    # local time 5 (=absolute 15) through local time 9 (=absolute 19) fall inside [15, 19): positive.
    assert labels[5:9] == [1, 1, 1, 1]
    # local time 4 (=absolute 14) is inside the guard band before the boundary.
    assert labels[4] == IGNORE_LABEL


def test_stack_labeled_rows_drops_ignored_steps() -> None:
    clip = ClipFeatures(
        clip=CachedClip(
            corpus="ego4d",
            source_video_id="v",
            clip_id="c",
            split="train",
            window_config="short",
            manifest_name="m",
            guard_band_seconds=1.0,
            positive_seconds=4.0,
            range_start_seconds=0.0,
            range_end_seconds=3.0,
            world_states=np.zeros((3, WORLD_STATE_DIM)),
            timestamps=np.arange(3, dtype=np.float64),
        ),
        features=np.arange(3 * 3076, dtype=np.float64).reshape(3, 3076),
        labels=[0, IGNORE_LABEL, 1],
        boundaries=[],
    )

    features, labels = stack_labeled_rows([clip])

    assert features.shape == (2, 3076)
    assert list(labels) == [0, 1]


def test_stack_gru_rows_builds_causal_windows_and_drops_ignored_steps() -> None:
    features = np.zeros((4, 3076))
    features[:, :WORLD_STATE_DIM] = np.arange(4).reshape(4, 1)
    clip = ClipFeatures(
        clip=CachedClip(
            corpus="ego4d",
            source_video_id="v",
            clip_id="c",
            split="train",
            window_config="short",
            manifest_name="m",
            guard_band_seconds=1.0,
            positive_seconds=4.0,
            range_start_seconds=0.0,
            range_end_seconds=4.0,
            world_states=np.zeros((4, WORLD_STATE_DIM)),
            timestamps=np.arange(4, dtype=np.float64),
        ),
        features=features,
        labels=[0, IGNORE_LABEL, 1, 0],
        boundaries=[],
    )

    windows, labels = stack_gru_rows([clip])

    assert windows.shape[0] == 3  # 4 steps minus 1 ignored
    assert list(labels) == [0, 1, 0]


def test_evaluate_at_fixed_policy_reports_recall_delay_and_false_triggers() -> None:
    """A hand-crafted scenario: one detected boundary, one missed boundary, one false trigger.

    Boundaries sit at t=15 and t=25 (rather than closer together) specifically so the 10-second
    cooldown from the deliberate early false trigger (t=0-1) has fully cleared well before the
    real detection window opens at t=15 -- the two events must not interact.
    """

    timestamps = np.arange(30, dtype=np.float64)
    cached = CachedClip(
        corpus="ego4d",
        source_video_id="video-1",
        clip_id="clip-1",
        split="test",
        window_config="short",
        manifest_name="m",
        guard_band_seconds=1.0,
        positive_seconds=4.0,
        range_start_seconds=0.0,
        range_end_seconds=29.0,
        world_states=np.zeros((30, WORLD_STATE_DIM)),
        timestamps=timestamps,
    )
    boundaries = [
        ProxyBoundary("ego4d", "video-1", "clip-1", 15.0, "kitchen", "1", "living_room", "1", "indoor-to-indoor"),
        ProxyBoundary("ego4d", "video-1", "clip-1", 25.0, "kitchen", "1", "bathroom", "1", "indoor-to-indoor"),
    ]
    intervals = [RoomInterval("ego4d", "video-1", "clip-1", 0.0, 30.0, "kitchen", "1")]
    labels = [0] * 30
    for boundary_time in (15.0, 25.0):
        for t in range(int(boundary_time), int(boundary_time) + 4):
            labels[t] = 1
        labels[int(boundary_time) - 1] = IGNORE_LABEL
        if int(boundary_time) + 4 < len(labels):
            labels[int(boundary_time) + 4] = IGNORE_LABEL

    clip = ClipFeatures(clip=cached, features=np.zeros((30, 3076)), labels=labels, boundaries=boundaries)

    probabilities = np.full(30, 0.1)
    # A spurious event during clean negative time, far before either boundary: fires at t=1, and
    # its 10-second cooldown fully elapses well before t=15.
    probabilities[0] = 0.9
    probabilities[1] = 0.9
    # Detect the first boundary with two consecutive high scores inside its positive window.
    probabilities[15] = 0.9
    probabilities[16] = 0.9
    # Leave the second boundary (25-28) entirely undetected.

    def probability_fn(cf: ClipFeatures) -> np.ndarray:
        return probabilities

    metrics = evaluate_at_fixed_policy([clip], probability_fn)

    assert metrics.total_boundary_count == 2
    assert metrics.detected_boundary_count == 1
    assert metrics.recall_at_budget == pytest.approx(0.5)
    assert metrics.median_detection_delay_seconds == pytest.approx(1.0)  # fired at t=16, boundary at t=15
    assert metrics.recall_by_family["indoor-to-indoor"] == pytest.approx(0.5)
    assert metrics.achieved_false_trigger_rate_per_10min > 0.0


def test_compute_group_average_precisions_groups_by_source_video() -> None:
    def make_clip(source_video_id: str, labels: list[int], scores: list[float]) -> ClipFeatures:
        features = np.zeros((len(labels), 3076))
        return ClipFeatures(
            clip=CachedClip(
                corpus="ego4d",
                source_video_id=source_video_id,
                clip_id=f"{source_video_id}-clip",
                split="heldout",
                window_config="short",
                manifest_name="m",
                guard_band_seconds=1.0,
                positive_seconds=4.0,
                range_start_seconds=0.0,
                range_end_seconds=float(len(labels)),
                world_states=np.zeros((len(labels), WORLD_STATE_DIM)),
                timestamps=np.arange(len(labels), dtype=np.float64),
            ),
            features=features,
            labels=labels,
            boundaries=[],
        )

    clip_a = make_clip("video-1", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    clip_b = make_clip("video-2", [0, 0, 1, 1], [0.6, 0.1, 0.9, 0.9])

    def logistic_fn(cf: ClipFeatures) -> np.ndarray:
        return np.array([0.1, 0.9, 0.2, 0.8]) if cf.clip.source_video_id == "video-1" else np.array([0.1, 0.1, 0.9, 0.9])

    def gru_fn(cf: ClipFeatures) -> np.ndarray:
        return np.array([0.5, 0.5, 0.5, 0.5])

    groups = compute_group_average_precisions([clip_a, clip_b], logistic_fn, gru_fn)

    assert len(groups) == 2


def _write_roompred_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_uid", "clip_uid", "start_time", "end_time", "label", "instance"])
        writer.writeheader()
        writer.writerows(rows)


def test_run_end_to_end_on_synthetic_fixtures(tmp_path: Path) -> None:
    """A full smoke test of the CLI pipeline: manifest + cache + annotations -> report + ONNX files."""

    manifest_name = "test-manifest"
    window_config = "short"
    cache_root = tmp_path / "cache"
    annotations_root = tmp_path / "annotations"
    output_dir = tmp_path / "output"
    annotations_root.mkdir()

    clip_specs = [
        ("video-1", "clip-1", "train"),
        ("video-2", "clip-2", "train"),
        ("video-3", "clip-3", "heldout"),
        ("video-4", "clip-4", "heldout"),
        ("video-5", "clip-5", "test"),
    ]

    rows = []
    manifest_clips = []
    rng = np.random.default_rng(42)
    for source_video_id, clip_id, split in clip_specs:
        steps = 40
        rows.append(
            {
                "video_uid": source_video_id,
                "clip_uid": clip_id,
                "start_time": "0",
                "end_time": "15",
                "label": "kitchen",
                "instance": "1",
            }
        )
        rows.append(
            {
                "video_uid": source_video_id,
                "clip_uid": clip_id,
                "start_time": "15",
                "end_time": str(steps),
                "label": "living_room",
                "instance": "1",
            }
        )
        manifest_clips.append(
            {
                "corpus": "ego4d",
                "source_video_id": source_video_id,
                "clip_id": clip_id,
                "split": split,
                "evaluation_runs": ["A"],
                "boundary_count": 1,
                "resolution_status": "clip-file",
                "resolution_reason": None,
                "extraction_strategy": "download-540p-clip",
                "source_start_frame": None,
                "source_end_frame": None,
            }
        )
        world_states = rng.normal(size=(steps, WORLD_STATE_DIM))
        timestamps = np.arange(steps, dtype=np.float64)
        _write_cache_pair(
            cache_root / "world-states" / manifest_name / "ego4d",
            corpus="ego4d",
            source_video_id=source_video_id,
            clip_id=clip_id,
            split=split,
            window_config=window_config,
            manifest_name=manifest_name,
            world_states=world_states,
            timestamps=timestamps,
        )

    _write_roompred_csv(annotations_root / "ego4d_roompred_train.csv", rows)
    # The loader requires both corpora present, even though housetours has zero usable rows here.
    with (annotations_root / "housetours_roompred_train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip_uid", "start_time", "end_time", "label"])
        writer.writeheader()

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"clips": manifest_clips}))

    args = type(
        "Args",
        (),
        {
            "manifest_path": manifest_path,
            "manifest_name": manifest_name,
            "cache_root": cache_root,
            "annotations_root": annotations_root,
            "window_config": window_config,
            "output_dir": output_dir,
        },
    )()

    report = run(args)

    assert report["run"] == "A"
    assert report["counts"]["train_clips"] == 2
    assert report["counts"]["heldout_clips"] == 2
    assert report["counts"]["test_clips"] == 1
    assert report["selection"]["selected"] in {"logistic", "gru"}
    assert (output_dir / "logistic_head.onnx").exists()
    assert (output_dir / "gru_head.onnx").exists()
    assert (output_dir / "evaluation-report.json").exists()

    persisted = json.loads((output_dir / "evaluation-report.json").read_text())
    assert persisted["window_config"] == window_config


def test_load_manifest_clips_reads_split_and_identity(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clips": [
                    {"corpus": "ego4d", "source_video_id": "v1", "clip_id": "c1", "split": "train"},
                    {"corpus": "ego4d", "source_video_id": "v2", "clip_id": "c2", "split": "heldout"},
                ]
            }
        )
    )

    clips = load_manifest_clips(manifest_path)

    assert len(clips) == 2
    assert clips[0].split == "train"
    assert clips[1].clip_id == "c2"
