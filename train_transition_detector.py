"""Train, calibrate, select, and evaluate the Stage 3 transition detector (issue #22).

Per ADR-0004, only **Run A** applies: train on Ego4D ``train``, fix the decision-policy operating
point on the disjoint Ego4D ``heldout`` group, and report the final threshold-transfer measurement
on the disjoint Ego4D ``test`` group. Runs B and C need HouseTours source video that does not exist
locally or on the Modal volume, so this script does not build or report them.

Inputs, all taken as parameters (never hardcoded), per the frozen-manifest contract:

- ``--manifest-path``: path to a frozen corpus manifest JSON written by
  ``blindsight.transition.corpus.write_frozen_corpus_artifacts`` (e.g. ``ego4d-only-v1`` or the
  smaller ``ego4d-30gb-v1``). Only the ``clips`` list's ``corpus``/``source_video_id``/``clip_id``/
  ``split`` fields are consulted here.
- ``--manifest-name``: the name embedded in the cache layout
  (``world-states/<manifest_name>/<corpus>/...``). This is independent of the manifest *path*
  because the encoding pipeline names cache directories by manifest name, not by file location.
- ``--cache-root``: root directory holding the ``world-states/`` tree of cached ``.npz``/``.json``
  pairs produced by the encoding pipeline (see the module docstring of this repository's shared
  cache-format contract). A missing cache file for a manifest clip is not an error here: it is
  recorded as a named "missing" clip in the report, per the requirement that the report "names
  missing or failed traces and explains their treatment."
- ``--annotations-root``: root directory holding the published EgoEnv RoomPred CSVs. This script
  re-derives room intervals and proxy boundaries from them via
  ``blindsight.transition.corpus.load_egoenv_roompred_directory`` and ``build_boundary_table``,
  because the frozen manifest records only clip identity and a boundary *count*, not boundary
  *times* — the actual interval/boundary timestamps that ``label_time_steps`` needs come from the
  same annotation source the corpus builder itself reads.
- ``--window-config``: ``short`` or ``long``, selecting which cached window configuration to load.
- ``--output-dir``: where the ONNX exports and the evaluation report JSON are written.

Real cached world-state files and real EgoEnv annotation CSVs are not expected to exist yet in a
fresh checkout — a sibling effort produces the cache, and the annotation archive is a separate
download. This module is exercised in tests with small synthetic ``.npz``/``.json`` fixtures and
hand-built ``RoomInterval``/``ProxyBoundary`` rows; the real end-to-end run happens once all of
issue #22's siblings (#20, #21) have produced real artifacts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

from blindsight.transition.corpus import (
    IGNORE_LABEL,
    ProxyBoundary,
    RoomInterval,
    build_boundary_table,
    label_time_steps,
    load_egoenv_roompred_directory,
)
from blindsight.transition.detector import (
    FEATURE_DIM,
    GRU_HISTORY_LENGTH,
    CausalGRUHead,
    DecisionPolicyState,
    HeadSelectionResult,
    LogisticHead,
    PlattCalibration,
    build_gru_windows,
    export_gru_to_onnx,
    export_logistic_to_onnx,
    fit_gru_head,
    gru_logits,
    select_detector_head,
)
from blindsight.transition.features import WORLD_STATE_DIM, compute_features_offline

RECALL_FLOOR = 0.25
DETECTION_DELAY_BUDGET_SECONDS = 4.0
FALSE_TRIGGER_BUDGET_PER_10_MIN = 1.0

_UNSAFE_CLIP_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def sanitize_clip_id(clip_id: str) -> str:
    """Mirror the filesystem-safe clip-id sanitizer the encoding pipeline applies to cache filenames.

    This module only reads the shared cache; it does not own the sanitizer. Any character outside
    ``[A-Za-z0-9_.-]`` becomes a single underscore, which is the conservative, unambiguous rule
    described in the shared cache-format contract.
    """

    return _UNSAFE_CLIP_ID_CHARS.sub("-", clip_id)


def cache_paths_for_clip(
    cache_root: Path, manifest_name: str, corpus: str, clip_id: str, window_config: str
) -> tuple[Path, Path]:
    """Return the ``(npz_path, json_path)`` pair for one ``(clip, window_config)`` per the contract:
    ``world-states/<manifest_name>/<corpus>/<clip_id-sanitized>__<window_config>.npz`` (+ ``.json``).

    This is the whole-clip layout the encoding pipeline uses for ``heldout``/``test`` clips. A
    ``train`` clip can instead be split across several disjoint boundary-window files (see
    ``cache_file_pairs_for_clip``); this function does not discover those.
    """

    directory = cache_root / "world-states" / manifest_name / corpus
    stem = f"{sanitize_clip_id(clip_id)}__{window_config}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def cache_file_pairs_for_clip(
    cache_root: Path, manifest_name: str, corpus: str, clip_id: str, window_config: str
) -> list[tuple[Path, Path]]:
    """Return every ``(npz_path, json_path)`` cache pair the encoding pipeline wrote for this clip.

    ``blindsight/transition/encode.py``'s ``plan_extraction_work_items`` gives a ``heldout``/``test``
    clip exactly one whole-clip file (``<clip>__<window_config>.npz``, the layout
    ``cache_paths_for_clip`` checks), but a ``train`` clip one or more disjoint boundary-window files
    named ``<clip>__boundary<index>__<window_config>.npz`` — the boundary-only encoding strategy from
    the spec's "Compute cost" section. Both layouts can exist on disk; this checks for both rather
    than assuming one, and never merges results across pairs (each covers an independent time range,
    so features/labels are built per pair, never bridging the gap between two boundary windows).
    """

    directory = cache_root / "world-states" / manifest_name / corpus
    sanitized = sanitize_clip_id(clip_id)
    pairs: list[tuple[Path, Path]] = []
    whole_npz, whole_json = cache_paths_for_clip(cache_root, manifest_name, corpus, clip_id, window_config)
    if whole_npz.exists() and whole_json.exists():
        pairs.append((whole_npz, whole_json))
    for npz_path in sorted(directory.glob(f"{sanitized}__boundary*__{window_config}.npz")):
        json_path = npz_path.with_suffix(".json")
        if json_path.exists():
            pairs.append((npz_path, json_path))
    return pairs


@dataclass(frozen=True)
class ManifestClipRef:
    """The subset of a frozen manifest's per-clip record this script needs."""

    corpus: str
    source_video_id: str
    clip_id: str
    split: str


def load_manifest_clips(manifest_path: Path) -> list[ManifestClipRef]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        ManifestClipRef(
            corpus=entry["corpus"],
            source_video_id=entry["source_video_id"],
            clip_id=entry["clip_id"],
            split=entry["split"],
        )
        for entry in data["clips"]
    ]


@dataclass(frozen=True)
class CachedClip:
    """One decoded ``(npz, json)`` cache pair, per the shared cache-format contract."""

    corpus: str
    source_video_id: str
    clip_id: str
    split: str
    window_config: str
    manifest_name: str
    guard_band_seconds: float
    positive_seconds: float
    range_start_seconds: float
    range_end_seconds: float
    world_states: np.ndarray
    timestamps: np.ndarray


def load_cached_clip(npz_path: Path, json_path: Path) -> CachedClip:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as data:
        world_states = np.asarray(data["world_states"], dtype=np.float32)
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    if world_states.ndim != 2 or world_states.shape[1] != WORLD_STATE_DIM:
        raise ValueError(f"{npz_path}: world_states must have shape (T, {WORLD_STATE_DIM}), got {world_states.shape}")
    if timestamps.shape != (world_states.shape[0],):
        raise ValueError(f"{npz_path}: timestamps shape {timestamps.shape} does not match world_states")
    return CachedClip(
        corpus=meta["corpus"],
        source_video_id=meta["source_video_id"],
        clip_id=meta["clip_id"],
        split=meta["split"],
        window_config=meta["window_config"],
        manifest_name=meta["manifest_name"],
        guard_band_seconds=float(meta["guard_band_seconds"]),
        positive_seconds=float(meta["positive_seconds"]),
        range_start_seconds=float(meta["range_start_seconds"]),
        range_end_seconds=float(meta["range_end_seconds"]),
        world_states=world_states,
        timestamps=timestamps,
    )


@dataclass(frozen=True)
class ClipFeatures:
    """One cached clip's causal features, per-step labels, and the boundaries inside it."""

    clip: CachedClip
    features: np.ndarray
    labels: list[int]
    boundaries: list[ProxyBoundary]


def label_cached_clip(
    cached: CachedClip, intervals: Sequence[RoomInterval], boundaries: Sequence[ProxyBoundary]
) -> list[int]:
    """Label each cached time step with ``corpus.label_time_steps``, in the clip's absolute time frame.

    Cached ``timestamps`` are seconds from ``range_start_seconds`` — the start of the *encoded*
    range, which the Ego4D boundary-window training strategy makes a sub-span of the whole clip.
    Room intervals and proxy boundaries are expressed in clip-absolute seconds, so this shifts the
    cached timestamps forward by ``range_start_seconds`` before calling the shared labeling rule.
    """

    clip_intervals = [i for i in intervals if i.corpus == cached.corpus and i.clip_id == cached.clip_id]
    clip_boundaries = [b for b in boundaries if b.corpus == cached.corpus and b.clip_id == cached.clip_id]
    absolute_times = [float(t) + cached.range_start_seconds for t in cached.timestamps.tolist()]
    return label_time_steps(
        absolute_times,
        clip_intervals,
        clip_boundaries,
        guard_band_seconds=cached.guard_band_seconds,
        positive_seconds=cached.positive_seconds,
    )


def build_clip_features(
    npz_path: Path, json_path: Path, intervals: Sequence[RoomInterval], boundaries: Sequence[ProxyBoundary]
) -> ClipFeatures:
    cached = load_cached_clip(npz_path, json_path)
    features = compute_features_offline(cached.world_states)
    labels = label_cached_clip(cached, intervals, boundaries)
    clip_boundaries = [b for b in boundaries if b.corpus == cached.corpus and b.clip_id == cached.clip_id]
    return ClipFeatures(clip=cached, features=features, labels=labels, boundaries=clip_boundaries)


def load_split_clip_features(
    manifest_clips: Sequence[ManifestClipRef],
    split: str,
    *,
    cache_root: Path,
    manifest_name: str,
    window_config: str,
    intervals: Sequence[RoomInterval],
    boundaries: Sequence[ProxyBoundary],
) -> tuple[list[ClipFeatures], list[str]]:
    """Load every manifest clip in ``split`` whose cache exists; name every one that does not."""

    loaded: list[ClipFeatures] = []
    missing: list[str] = []
    for ref in manifest_clips:
        if ref.split != split:
            continue
        pairs = cache_file_pairs_for_clip(cache_root, manifest_name, ref.corpus, ref.clip_id, window_config)
        if not pairs:
            missing.append(ref.clip_id)
            continue
        for npz_path, json_path in pairs:
            loaded.append(build_clip_features(npz_path, json_path, intervals, boundaries))
    return loaded, missing


def stack_labeled_rows(clip_features: Sequence[ClipFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every non-ignored feature row and label across a list of clips."""

    features_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for clip in clip_features:
        labels_array = np.array(clip.labels)
        mask = labels_array != IGNORE_LABEL
        features_parts.append(clip.features[mask])
        labels_parts.append(labels_array[mask])
    if not features_parts:
        return np.zeros((0, FEATURE_DIM)), np.zeros((0,))
    return np.concatenate(features_parts, axis=0), np.concatenate(labels_parts, axis=0)


def stack_gru_rows(clip_features: Sequence[ClipFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """Build causal GRU windows per clip (never across a clip boundary), then drop ignored rows."""

    windows_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for clip in clip_features:
        normalized = clip.features[:, :WORLD_STATE_DIM]
        windows = build_gru_windows(normalized)
        labels_array = np.array(clip.labels)
        mask = labels_array != IGNORE_LABEL
        windows_parts.append(windows[mask])
        labels_parts.append(labels_array[mask])
    if not windows_parts:
        return np.zeros((0, GRU_HISTORY_LENGTH, WORLD_STATE_DIM)), np.zeros((0,))
    return np.concatenate(windows_parts, axis=0), np.concatenate(labels_parts, axis=0)


ProbabilityFn = Callable[[ClipFeatures], np.ndarray]


def logistic_probability_fn(head: LogisticHead, calibration: PlattCalibration) -> ProbabilityFn:
    def score(clip: ClipFeatures) -> np.ndarray:
        return calibration.apply(head.logit(clip.features))

    return score


def gru_probability_fn(head: CausalGRUHead, calibration: PlattCalibration) -> ProbabilityFn:
    def score(clip: ClipFeatures) -> np.ndarray:
        windows = build_gru_windows(clip.features[:, :WORLD_STATE_DIM])
        return calibration.apply(gru_logits(head, windows))

    return score


def compute_group_average_precisions(
    clip_features: Sequence[ClipFeatures],
    logistic_fn: ProbabilityFn,
    gru_fn: ProbabilityFn,
) -> list[tuple[float, float]]:
    """One ``(logistic_ap, gru_ap)`` pair per held-out group (grouped by source video).

    Grouping by source video, not by clip, matches the evaluation protocol's requirement to keep
    "complete clips and source groups intact": several clips can come from one source video, and
    treating each as an independent vote would let one video's idiosyncrasies cast several votes. A
    group with no positive rows (or none at all) has no well-defined average precision and is
    excluded from the vote, not scored as a win for either head.
    """

    by_group: dict[str, list[ClipFeatures]] = defaultdict(list)
    for clip in clip_features:
        by_group[clip.clip.source_video_id].append(clip)

    results: list[tuple[float, float]] = []
    for group_clips in by_group.values():
        labels_parts: list[np.ndarray] = []
        logistic_parts: list[np.ndarray] = []
        gru_parts: list[np.ndarray] = []
        for clip in group_clips:
            labels_array = np.array(clip.labels)
            mask = labels_array != IGNORE_LABEL
            if not mask.any():
                continue
            labels_parts.append(labels_array[mask])
            logistic_parts.append(logistic_fn(clip)[mask])
            gru_parts.append(gru_fn(clip)[mask])
        if not labels_parts:
            continue
        labels_concat = np.concatenate(labels_parts)
        if len(np.unique(labels_concat)) < 2:
            continue
        logistic_ap = average_precision_score(labels_concat, np.concatenate(logistic_parts))
        gru_ap = average_precision_score(labels_concat, np.concatenate(gru_parts))
        results.append((float(logistic_ap), float(gru_ap)))
    return results


@dataclass(frozen=True)
class EvaluationMetrics:
    """The Run-A reported values from issue #22, for one head at the fixed decision policy."""

    average_precision: float
    brier_score: float
    recall_at_budget: float
    achieved_false_trigger_rate_per_10min: float
    meets_false_trigger_budget: bool
    median_detection_delay_seconds: float | None
    recall_by_family: dict[str, float]
    boundary_count_by_family: dict[str, int]
    detected_boundary_count: int
    total_boundary_count: int
    clip_count: int
    positive_step_count: int
    ignored_step_count: int
    negative_step_count: int


def evaluate_at_fixed_policy(
    clip_features: Sequence[ClipFeatures], probability_fn: ProbabilityFn
) -> EvaluationMetrics:
    """Run the fixed decision policy over every step of every clip and report the Run-A values.

    The false-trigger rate is measured from *every* time step, not a sampled negative set, per the
    specification: the deployed detector scores every step, so the measured rate must come from
    every step too. A boundary counts as detected when a transition event fires at any point in its
    ``[boundary_time, boundary_time + positive_seconds)`` window; the delay is the time from the
    boundary to the first such event.
    """

    all_labels: list[int] = []
    all_probabilities: list[float] = []
    detected_by_family: dict[str, int] = defaultdict(int)
    total_by_family: dict[str, int] = defaultdict(int)
    delays: list[float] = []
    negative_seconds = 0.0
    false_trigger_events = 0
    positive_steps = ignored_steps = negative_steps = 0

    for clip in clip_features:
        probabilities = probability_fn(clip)
        if probabilities.shape[0] != len(clip.labels):
            raise ValueError("probability_fn must return exactly one probability per time step")
        timestamps = clip.clip.timestamps
        policy = DecisionPolicyState()
        events: list[tuple[float, bool]] = []
        previous_time: float | None = None
        for index, label in enumerate(clip.labels):
            absolute_time = float(timestamps[index]) + clip.clip.range_start_seconds
            dt_seconds = 1.0 if previous_time is None else max(absolute_time - previous_time, 0.0)
            previous_time = absolute_time
            probability = float(probabilities[index])
            fired = policy.step(probability, dt_seconds)
            events.append((absolute_time, fired))
            if label == IGNORE_LABEL:
                ignored_steps += 1
                continue
            all_labels.append(label)
            all_probabilities.append(probability)
            if label == 1:
                positive_steps += 1
            else:
                negative_steps += 1
                negative_seconds += dt_seconds
                if fired:
                    false_trigger_events += 1

        for boundary in clip.boundaries:
            total_by_family[boundary.family] += 1
            window_start = boundary.boundary_time
            window_end = boundary.boundary_time + clip.clip.positive_seconds
            fire_times = [t for t, fired in events if fired and window_start <= t < window_end]
            if fire_times:
                detected_by_family[boundary.family] += 1
                delays.append(fire_times[0] - boundary.boundary_time)

    total_boundaries = sum(total_by_family.values())
    detected_boundaries = sum(detected_by_family.values())
    recall_by_family = {
        family: (detected_by_family.get(family, 0) / count if count else 0.0)
        for family, count in total_by_family.items()
    }
    negative_minutes = negative_seconds / 60.0
    achieved_rate = (false_trigger_events / negative_minutes * 10.0) if negative_minutes > 0 else 0.0
    labels_array = np.array(all_labels)
    probabilities_array = np.array(all_probabilities)
    has_both_classes = len(np.unique(labels_array)) > 1
    average_precision = float(average_precision_score(labels_array, probabilities_array)) if has_both_classes else float("nan")
    brier_score = float(np.mean((probabilities_array - labels_array) ** 2)) if labels_array.size else float("nan")

    return EvaluationMetrics(
        average_precision=average_precision,
        brier_score=brier_score,
        recall_at_budget=(detected_boundaries / total_boundaries) if total_boundaries else 0.0,
        achieved_false_trigger_rate_per_10min=achieved_rate,
        meets_false_trigger_budget=achieved_rate <= FALSE_TRIGGER_BUDGET_PER_10_MIN,
        median_detection_delay_seconds=(float(np.median(delays)) if delays else None),
        recall_by_family=recall_by_family,
        boundary_count_by_family=dict(total_by_family),
        detected_boundary_count=detected_boundaries,
        total_boundary_count=total_boundaries,
        clip_count=len(clip_features),
        positive_step_count=positive_steps,
        ignored_step_count=ignored_steps,
        negative_step_count=negative_steps,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", type=Path, required=True, help="Path to the frozen corpus manifest JSON.")
    parser.add_argument("--manifest-name", type=str, required=True, help="Manifest name used in the cache layout.")
    parser.add_argument("--cache-root", type=Path, required=True, help="Root of the cached world-states/ tree.")
    parser.add_argument("--annotations-root", type=Path, required=True, help="Root of the EgoEnv RoomPred CSVs.")
    parser.add_argument("--window-config", choices=["short", "long"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write ONNX exports and the report.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_clips = load_manifest_clips(args.manifest_path)
    intervals = load_egoenv_roompred_directory(args.annotations_root)
    boundaries, _zero_length_rows = build_boundary_table(intervals)

    load_kwargs = dict(
        cache_root=args.cache_root,
        manifest_name=args.manifest_name,
        window_config=args.window_config,
        intervals=intervals,
        boundaries=boundaries,
    )
    train_clips, missing_train = load_split_clip_features(manifest_clips, "train", **load_kwargs)
    heldout_clips, missing_heldout = load_split_clip_features(manifest_clips, "heldout", **load_kwargs)
    test_clips, missing_test = load_split_clip_features(manifest_clips, "test", **load_kwargs)

    if not train_clips:
        raise ValueError("No cached training clips were found; cannot fit either head.")

    train_features, train_labels = stack_labeled_rows(train_clips)
    logistic_head = LogisticHead.fit(train_features, train_labels)

    train_windows, train_window_labels = stack_gru_rows(train_clips)
    gru_head = fit_gru_head(train_windows, train_window_labels)

    heldout_features, heldout_labels = stack_labeled_rows(heldout_clips)
    heldout_logistic_logits = logistic_head.logit(heldout_features)
    logistic_calibration = PlattCalibration.fit(heldout_logistic_logits, heldout_labels)

    heldout_windows, heldout_window_labels = stack_gru_rows(heldout_clips)
    heldout_gru_logits = gru_logits(gru_head, heldout_windows)
    gru_calibration = PlattCalibration.fit(heldout_gru_logits, heldout_window_labels)

    logistic_score = logistic_probability_fn(logistic_head, logistic_calibration)
    gru_score = gru_probability_fn(gru_head, gru_calibration)

    heldout_logistic_probability = logistic_calibration.apply(heldout_logistic_logits)
    heldout_gru_probability = gru_calibration.apply(heldout_gru_logits)
    heldout_logistic_ap = (
        float(average_precision_score(heldout_labels, heldout_logistic_probability))
        if len(np.unique(heldout_labels)) > 1
        else float("nan")
    )
    heldout_gru_ap = (
        float(average_precision_score(heldout_window_labels, heldout_gru_probability))
        if len(np.unique(heldout_window_labels)) > 1
        else float("nan")
    )
    group_average_precisions = compute_group_average_precisions(heldout_clips, logistic_score, gru_score)
    selection = select_detector_head(heldout_logistic_ap, heldout_gru_ap, group_average_precisions)

    logistic_test_metrics = evaluate_at_fixed_policy(test_clips, logistic_score)
    gru_test_metrics = evaluate_at_fixed_policy(test_clips, gru_score)
    selected_test_metrics = logistic_test_metrics if selection.selected == "logistic" else gru_test_metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_logistic_to_onnx(logistic_head, logistic_calibration, args.output_dir / "logistic_head.onnx")
    export_gru_to_onnx(gru_head, gru_calibration, args.output_dir / "gru_head.onnx")

    kill_criterion_triggered = selected_test_metrics.recall_at_budget < RECALL_FLOOR
    meets_delay_budget = (
        selected_test_metrics.median_detection_delay_seconds is not None
        and selected_test_metrics.median_detection_delay_seconds <= DETECTION_DELAY_BUDGET_SECONDS
    )

    report: dict[str, object] = {
        "run": "A",
        "manifest_path": str(args.manifest_path),
        "manifest_name": args.manifest_name,
        "window_config": args.window_config,
        "selection": asdict(selection),
        "heldout": {
            "logistic_average_precision": heldout_logistic_ap,
            "gru_average_precision": heldout_gru_ap,
            "held_out_group_count": len(group_average_precisions),
            "clip_count": len(heldout_clips),
        },
        "test": {
            "logistic": asdict(logistic_test_metrics),
            "gru": asdict(gru_test_metrics),
            "selected_head": selection.selected,
        },
        "counts": {
            "train_clips": len(train_clips),
            "heldout_clips": len(heldout_clips),
            "test_clips": len(test_clips),
            "missing_train_clips": missing_train,
            "missing_heldout_clips": missing_heldout,
            "missing_test_clips": missing_test,
        },
        "operating_point": {
            "recall_floor": RECALL_FLOOR,
            "detection_delay_budget_seconds": DETECTION_DELAY_BUDGET_SECONDS,
            "false_trigger_budget_per_10_min": FALSE_TRIGGER_BUDGET_PER_10_MIN,
        },
        "kill_criterion_triggered": kill_criterion_triggered,
        "meets_delay_budget": meets_delay_budget,
    }
    (args.output_dir / "evaluation-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
