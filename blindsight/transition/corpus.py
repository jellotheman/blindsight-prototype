"""Normalize EgoEnv room intervals into reproducible proxy boundaries."""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Literal, Mapping

CorpusName = Literal["ego4d", "housetours"]
BoundaryFamily = Literal[
    "indoor-to-indoor", "threshold-cross", "indoor-to-outdoor", "outdoor-to-outdoor"
]
ResolutionStatus = Literal["clip-file", "parent-cut", "source-video", "unresolved", "failed"]
SplitName = Literal["train", "heldout", "test", "not-selected", "unresolved", "unavailable"]
IGNORE_LABEL = -1

# This is the union of the published EgoEnv room vocabulary.  The zone map is intentionally data,
# rather than an inference from spelling, so every family assignment is reproducible.
ROOM_LABEL_ZONES: dict[str, str] = {
    "attic": "indoor",
    "basement": "indoor",
    "bathroom": "indoor",
    "bedroom": "indoor",
    "closet": "indoor",
    "corridor/hallway": "indoor",
    "dining_room": "indoor",
    "garage / shed": "indoor",
    "gym": "indoor",
    "kitchen": "indoor",
    "living_room": "indoor",
    "office / home_office": "indoor",
    "recreation_room (billiards room / play room)": "indoor",
    "storage / laundry / utility room": "indoor",
    "balcony": "outdoor",
    "driveway": "outdoor",
    "lawn/yard/garden": "outdoor",
    "porch": "outdoor",
    "swimming_pool": "outdoor",
    "front_door/entrance": "threshold",
    "staircase": "threshold",
}


@dataclass(frozen=True)
class RoomInterval:
    """One room-visit interval from either published EgoEnv label set."""

    corpus: CorpusName
    source_video_id: str
    clip_id: str
    start_time: float
    end_time: float
    room_label: str
    room_instance: str


@dataclass(frozen=True)
class ProxyBoundary:
    corpus: CorpusName
    source_video_id: str
    clip_id: str
    boundary_time: float
    prior_room_label: str
    prior_room_instance: str
    next_room_label: str
    next_room_instance: str
    family: BoundaryFamily


@dataclass(frozen=True)
class HouseToursClipReference:
    """The public-video source and 2-fps source-frame cut encoded by an EgoEnv clip id."""

    source_video_id: str
    source_start_frame: int
    source_end_frame: int


@dataclass(frozen=True)
class ManifestClip:
    """One complete source clip and its immutable corpus-selection decision."""

    corpus: CorpusName
    source_video_id: str
    clip_id: str
    split: SplitName
    evaluation_runs: tuple[str, ...]
    boundary_count: int
    resolution_status: ResolutionStatus
    resolution_reason: str | None
    extraction_strategy: str
    source_start_frame: int | None = None
    source_end_frame: int | None = None


@dataclass(frozen=True)
class FrozenCorpusManifest:
    """The data-only record that later extraction and training runs must consume unchanged."""

    schema_version: int
    guard_band_seconds: float
    positive_seconds: float
    random_seed: int
    selection_procedure: str
    ego4d_data_version: str
    ego4d_cli_version: str
    clips: tuple[ManifestClip, ...]

    @property
    def unresolved_ego4d(self) -> tuple[ManifestClip, ...]:
        return tuple(
            clip
            for clip in self.clips
            if clip.corpus == "ego4d" and clip.resolution_status == "unresolved"
        )

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["clips"] = [asdict(clip) for clip in self.clips]
        value["unresolved_ego4d_count"] = len(self.unresolved_ego4d)
        return value

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _zone(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_")
    try:
        return ROOM_LABEL_ZONES[normalized]
    except KeyError as exc:
        raise ValueError(f"Room label {label!r} is missing from ROOM_LABEL_ZONES.") from exc


def classify_boundary(prior_label: str, next_label: str) -> BoundaryFamily:
    """Return the declared room-zone family for one proxy boundary."""

    prior_zone, next_zone = _zone(prior_label), _zone(next_label)
    if "threshold" in {prior_zone, next_zone}:
        return "threshold-cross"
    if prior_zone == "outdoor" and next_zone == "outdoor":
        return "outdoor-to-outdoor"
    if "outdoor" in {prior_zone, next_zone}:
        return "indoor-to-outdoor"
    return "indoor-to-indoor"


def parse_housetours_clip_id(clip_id: str) -> HouseToursClipReference:
    """Parse ``<video-id>_<start-frame>_<end-frame>`` without confusing frames with seconds."""

    try:
        source_video_id, start_text, end_text = clip_id.rsplit("_", 2)
        start_frame, end_frame = int(start_text), int(end_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid HouseTours clip identifier {clip_id!r}.") from exc
    if not source_video_id or start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"Invalid HouseTours clip identifier {clip_id!r}.")
    return HouseToursClipReference(source_video_id, start_frame, end_frame)


def load_roompred_csv(path: Path, *, corpus: CorpusName) -> list[RoomInterval]:
    """Read either published RoomPred CSV layout into one normalized interval representation."""

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    fields = set(rows[0])
    required = (
        {"video_uid", "clip_uid", "start_time", "end_time", "label", "instance"}
        if corpus == "ego4d"
        else {"clip_uid", "start_time", "end_time", "label"}
    )
    if not required.issubset(fields):
        raise ValueError(f"{path} does not have the required {corpus} RoomPred columns.")
    intervals: list[RoomInterval] = []
    for row in rows:
        clip_id = row["clip_uid"].strip()
        if corpus == "ego4d":
            source_video_id = row["video_uid"].strip()
            room_instance = row["instance"].strip()
        else:
            source_video_id = parse_housetours_clip_id(clip_id).source_video_id
            room_instance = ""
        intervals.append(
            RoomInterval(
                corpus=corpus,
                source_video_id=source_video_id,
                clip_id=clip_id,
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                room_label=row["label"].strip(),
                room_instance=room_instance,
            )
        )
    return intervals


def load_egoenv_roompred_directory(root: Path) -> list[RoomInterval]:
    """Read every published RoomPred split from both EgoEnv corpora.

    The caller may point at the extracted archive root or at a parent directory that contains it.
    Labels are checked immediately against the frozen union map, before a manifest can be made.
    """

    intervals: list[RoomInterval] = []
    for corpus in ("ego4d", "housetours"):
        paths = sorted(root.rglob(f"{corpus}_roompred_*.csv"))
        if not paths:
            raise ValueError(f"No {corpus} RoomPred CSV files found below {root}.")
        for path in paths:
            intervals.extend(load_roompred_csv(path, corpus=corpus))
    unknown_labels = sorted({item.room_label for item in intervals if item.room_label not in ROOM_LABEL_ZONES})
    if unknown_labels:
        raise ValueError(f"EgoEnv contains labels missing from ROOM_LABEL_ZONES: {unknown_labels!r}")
    return intervals


def _extraction_strategy(corpus: CorpusName, resolution_status: ResolutionStatus) -> str:
    if resolution_status == "unresolved":
        return "unresolved"
    if resolution_status == "failed":
        return "not-available"
    if corpus == "housetours":
        return "download-source-video-and-cut-2fps-range"
    if resolution_status == "clip-file":
        return "download-540p-clip"
    return "download-540p-parent-and-cut"


def build_frozen_manifest(
    intervals: list[RoomInterval],
    boundaries: list[ProxyBoundary],
    *,
    random_seed: int,
    heldout_counts: Mapping[CorpusName, int],
    train_counts: Mapping[CorpusName, int],
    ego4d_data_version: str,
    ego4d_cli_version: str,
    guard_band_seconds: float,
    resolution_by_clip: Mapping[tuple[CorpusName, str], ResolutionStatus],
    resolution_reasons: Mapping[tuple[CorpusName, str], str] | None = None,
    test_counts: Mapping[CorpusName, int] | None = None,
) -> FrozenCorpusManifest:
    """Freeze whole-clip corpus splits before any feature extraction occurs.

    ``heldout`` fixes the decision-policy operating point; the optional, disjoint ``test`` group
    gives the final threshold-transfer measurement, per the Stage 3 evaluation protocol's rule that
    the group that fixes the operating point cannot also provide the transfer measurement. Both are
    uniform-random samples over eligible clips, drawn from one seeded generator so the split is
    reproducible; training selection is independently sorted by proxy-boundary density. Failed and
    unresolved identifiers remain visible but cannot silently reduce any selected split.
    """

    if guard_band_seconds < 0:
        raise ValueError("guard_band_seconds cannot be negative")
    if not ego4d_data_version or not ego4d_cli_version:
        raise ValueError("Ego4D data and CLI versions must both be pinned")
    source_by_key: dict[tuple[CorpusName, str], str] = {}
    for interval in intervals:
        key = (interval.corpus, interval.clip_id)
        existing = source_by_key.setdefault(key, interval.source_video_id)
        if existing != interval.source_video_id:
            raise ValueError(f"Clip {key!r} resolves to multiple source videos.")
    boundary_counts: dict[tuple[CorpusName, str], int] = defaultdict(int)
    for boundary in boundaries:
        boundary_counts[(boundary.corpus, boundary.clip_id)] += 1

    split_by_key: dict[tuple[CorpusName, str], SplitName] = {}
    for corpus in ("ego4d", "housetours"):
        eligible = sorted(
            key
            for key in source_by_key
            if key[0] == corpus
            and resolution_by_clip.get(key) in {"clip-file", "parent-cut", "source-video"}
        )
        requested_heldout = heldout_counts.get(corpus, 0)
        if requested_heldout > len(eligible):
            raise ValueError(
                f"Requested {requested_heldout} held-out {corpus} clips, only {len(eligible)} resolve."
            )
        rng = random.Random(random_seed)
        heldout = set(rng.sample(eligible, requested_heldout))
        after_heldout = [key for key in eligible if key not in heldout]
        requested_test = (test_counts or {}).get(corpus, 0)
        if requested_test > len(after_heldout):
            raise ValueError(
                f"Requested {requested_test} test {corpus} clips, only {len(after_heldout)} remain "
                "after the held-out draw."
            )
        test = set(rng.sample(after_heldout, requested_test))
        remaining = [key for key in after_heldout if key not in test]
        requested_train = train_counts.get(corpus, len(remaining))
        if requested_train > len(remaining):
            raise ValueError(
                f"Requested {requested_train} training {corpus} clips, only {len(remaining)} remain."
            )
        train = set(sorted(remaining, key=lambda key: (-boundary_counts[key], key[1]))[:requested_train])
        for key in source_by_key:
            if key[0] != corpus:
                continue
            status = resolution_by_clip.get(key, "unresolved")
            if status == "unresolved":
                split_by_key[key] = "unresolved"
            elif status == "failed":
                split_by_key[key] = "unavailable"
            elif key in heldout:
                split_by_key[key] = "heldout"
            elif key in test:
                split_by_key[key] = "test"
            elif key in train:
                split_by_key[key] = "train"
            else:
                split_by_key[key] = "not-selected"

    entries: list[ManifestClip] = []
    for corpus, clip_id in sorted(source_by_key):
        status = resolution_by_clip.get((corpus, clip_id), "unresolved")
        frame_range = parse_housetours_clip_id(clip_id) if corpus == "housetours" else None
        entries.append(
            ManifestClip(
                corpus=corpus,
                source_video_id=source_by_key[(corpus, clip_id)],
                clip_id=clip_id,
                split=split_by_key[(corpus, clip_id)],
                evaluation_runs=("A", "C") if corpus == "ego4d" else ("B",),
                boundary_count=boundary_counts[(corpus, clip_id)],
                resolution_status=status,
                resolution_reason=(resolution_reasons or {}).get((corpus, clip_id)),
                extraction_strategy=_extraction_strategy(corpus, status),
                source_start_frame=frame_range.source_start_frame if frame_range else None,
                source_end_frame=frame_range.source_end_frame if frame_range else None,
            )
        )
    return FrozenCorpusManifest(
        schema_version=1,
        guard_band_seconds=guard_band_seconds,
        positive_seconds=4.0,
        random_seed=random_seed,
        selection_procedure=(
            "Held-out clips are sampled uniformly without replacement from resolved clips; "
            "training clips are selected by descending proxy-boundary count and clip identifier."
        ),
        ego4d_data_version=ego4d_data_version,
        ego4d_cli_version=ego4d_cli_version,
        clips=tuple(entries),
    )


def build_corpus_report(
    intervals: list[RoomInterval],
    boundaries: list[ProxyBoundary],
    *,
    zero_length_rows: int,
    manifest: FrozenCorpusManifest,
    specification_counts: Mapping[CorpusName, Mapping[str, int]],
    difference_explanations: Mapping[tuple[CorpusName, str], str] | None = None,
) -> dict[str, object]:
    """Count the frozen corpus and reject unexplained deviations from its declared specification."""

    observed_zero_length = sum(interval.end_time == interval.start_time for interval in intervals)
    if observed_zero_length != zero_length_rows:
        raise ValueError("zero_length_rows does not match the normalized source rows")
    clips_by_key = {(clip.corpus, clip.clip_id): clip for clip in manifest.clips}
    corpora: dict[str, dict[str, object]] = {}
    splits: dict[str, dict[str, object]] = {}
    for corpus in ("ego4d", "housetours"):
        source_intervals = [item for item in intervals if item.corpus == corpus]
        source_boundaries = [item for item in boundaries if item.corpus == corpus]
        families = {
            family: sum(item.family == family for item in source_boundaries)
            for family in (
                "indoor-to-indoor",
                "threshold-cross",
                "indoor-to-outdoor",
                "outdoor-to-outdoor",
            )
        }
        corpora[corpus] = {
            "room_intervals": len(source_intervals),
            "clips": len({item.clip_id for item in source_intervals}),
            "source_videos": len({item.source_video_id for item in source_intervals}),
            "proxy_transitions": len(source_boundaries),
            "zero_length_rows": sum(item.end_time == item.start_time for item in source_intervals),
            "boundary_families": families,
        }
        for split in ("train", "heldout", "test", "not-selected", "unresolved", "unavailable"):
            selected = [clip for clip in manifest.clips if clip.corpus == corpus and clip.split == split]
            if not selected:
                continue
            selected_keys = {(clip.corpus, clip.clip_id) for clip in selected}
            selected_boundaries = [
                boundary
                for boundary in source_boundaries
                if (boundary.corpus, boundary.clip_id) in selected_keys
            ]
            splits[f"{corpus}:{split}"] = {
                "clips": len(selected),
                "source_videos": len({clip.source_video_id for clip in selected}),
                "proxy_transitions": len(selected_boundaries),
                "boundary_families": {
                    family: sum(boundary.family == family for boundary in selected_boundaries)
                    for family in families
                },
            }

    comparison: dict[str, dict[str, dict[str, object]]] = {}
    explanations = difference_explanations or {}
    for corpus, expected_metrics in specification_counts.items():
        actual_metrics = corpora[corpus]
        corpus_comparison: dict[str, dict[str, object]] = {}
        for metric, expected in expected_metrics.items():
            observed = actual_metrics.get(metric)
            if not isinstance(observed, int):
                raise ValueError(f"Specification metric {corpus}.{metric} is not a scalar count.")
            difference = observed - expected
            explanation = explanations.get((corpus, metric))
            if difference and not explanation:
                raise ValueError(f"Specification difference for {corpus}.{metric} needs an explanation.")
            corpus_comparison[metric] = {
                "expected": expected,
                "observed": observed,
                "difference": difference,
                "explanation": explanation,
            }
        comparison[corpus] = corpus_comparison
    return {
        "corpora": corpora,
        "splits": splits,
        "specification_comparison": comparison,
        "unresolved_ego4d_identifiers": [
            {"video_uid": clip.source_video_id, "clip_uid": clip.clip_id, "reason": clip.resolution_reason}
            for clip in manifest.unresolved_ego4d
        ],
    }


def write_frozen_corpus_artifacts(
    destination: Path,
    intervals: list[RoomInterval],
    boundaries: list[ProxyBoundary],
    *,
    random_seed: int,
    heldout_counts: Mapping[CorpusName, int],
    train_counts: Mapping[CorpusName, int],
    ego4d_data_version: str,
    ego4d_cli_version: str,
    guard_band_seconds: float,
    resolution_by_clip: Mapping[tuple[CorpusName, str], ResolutionStatus],
    specification_counts: Mapping[CorpusName, Mapping[str, int]],
    resolution_reasons: Mapping[tuple[CorpusName, str], str] | None = None,
    difference_explanations: Mapping[tuple[CorpusName, str], str] | None = None,
    test_counts: Mapping[CorpusName, int] | None = None,
) -> tuple[FrozenCorpusManifest, dict[str, object]]:
    """Write the manifest and report together, refusing to alter an existing frozen artifact."""

    manifest_path = destination / "manifest.json"
    report_path = destination / "corpus-report.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError("Frozen corpus artifacts already exist and cannot be overwritten.")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = build_frozen_manifest(
        intervals,
        boundaries,
        random_seed=random_seed,
        heldout_counts=heldout_counts,
        train_counts=train_counts,
        ego4d_data_version=ego4d_data_version,
        ego4d_cli_version=ego4d_cli_version,
        guard_band_seconds=guard_band_seconds,
        resolution_by_clip=resolution_by_clip,
        resolution_reasons=resolution_reasons,
        test_counts=test_counts,
    )
    zero_length_rows = sum(item.end_time == item.start_time for item in intervals)
    report = build_corpus_report(
        intervals,
        boundaries,
        zero_length_rows=zero_length_rows,
        manifest=manifest,
        specification_counts=specification_counts,
        difference_explanations=difference_explanations,
    )
    manifest.save(manifest_path)
    try:
        with report_path.open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest, report


def build_boundary_table(intervals: list[RoomInterval]) -> tuple[list[ProxyBoundary], int]:
    """Create proxy boundaries only from adjacent, changed, non-empty room intervals.

    A clip stays inside its corpus namespace, since the two annotations can reuse identifiers.
    The return value separately records ignored zero-duration source rows for the corpus report.
    """

    negative_intervals = [interval for interval in intervals if interval.end_time < interval.start_time]
    if negative_intervals:
        raise ValueError("Room intervals cannot end before they start.")
    zero_length_rows = sum(interval.end_time == interval.start_time for interval in intervals)
    by_clip: dict[tuple[CorpusName, str, str], list[RoomInterval]] = defaultdict(list)
    for interval in intervals:
        if interval.end_time == interval.start_time:
            continue
        by_clip[(interval.corpus, interval.source_video_id, interval.clip_id)].append(interval)

    boundaries: list[ProxyBoundary] = []
    for key in sorted(by_clip):
        ordered = sorted(
            by_clip[key], key=lambda item: (item.start_time, item.end_time, item.room_label, item.room_instance)
        )
        for prior, following in zip(ordered, ordered[1:], strict=False):
            changed = (prior.room_label, prior.room_instance) != (
                following.room_label,
                following.room_instance,
            )
            if not changed:
                continue
            boundaries.append(
                ProxyBoundary(
                    corpus=following.corpus,
                    source_video_id=following.source_video_id,
                    clip_id=following.clip_id,
                    boundary_time=following.start_time,
                    prior_room_label=prior.room_label,
                    prior_room_instance=prior.room_instance,
                    next_room_label=following.room_label,
                    next_room_instance=following.room_instance,
                    family=classify_boundary(prior.room_label, following.room_label),
                )
            )
    return boundaries, zero_length_rows


def label_time_steps(
    time_steps: list[float],
    intervals: list[RoomInterval],
    boundaries: list[ProxyBoundary],
    *,
    guard_band_seconds: float,
    positive_seconds: float = 4.0,
) -> list[int]:
    """Label each scored step from the declared room-interval proxy convention.

    Unannotated steps are ignored.  Within annotated room intervals, ordinary steps are negative;
    the two guard bands are ignored and the four seconds after a boundary are positive.  Positive
    evidence takes precedence if sparse annotations make two declared ranges overlap.
    """

    if guard_band_seconds < 0:
        raise ValueError("guard_band_seconds cannot be negative")
    if positive_seconds <= 0:
        raise ValueError("positive_seconds must be positive")
    scored = [any(interval.start_time <= time < interval.end_time for interval in intervals) for time in time_steps]
    labels = [0 if active else IGNORE_LABEL for active in scored]
    for boundary in boundaries:
        guard_start = boundary.boundary_time - guard_band_seconds
        guard_end = boundary.boundary_time + positive_seconds + guard_band_seconds
        positive_end = boundary.boundary_time + positive_seconds
        for index, time in enumerate(time_steps):
            if labels[index] == IGNORE_LABEL:
                continue
            if guard_start <= time < boundary.boundary_time or positive_end <= time < guard_end:
                labels[index] = IGNORE_LABEL
    for boundary in boundaries:
        positive_end = boundary.boundary_time + positive_seconds
        for index, time in enumerate(time_steps):
            if scored[index] and boundary.boundary_time <= time < positive_end:
                labels[index] = 1
    return labels
