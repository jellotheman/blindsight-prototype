"""Modal encoding pipeline for the Stage 3 causal world-state cache (issue #21).

This is a **sibling** file to ``modal_transition.py``, not an addition to it. ``docs/spec/phase-3-
transition.md``'s "Code layout" section names ``modal_transition.py`` as the one Modal application
file for this package; this pass deliberately deviates into its own file to avoid a live-file
collision with a concurrent agent who owns ``modal_transition.py`` for corpus acquisition. It
declares its own ``modal.App`` (``blindsight-transition-encode``) and reuses the existing data
volume and Ego4D secret only by name, exactly as the corpus-acquisition app does. Reconciling the
two files into one is left to a follow-up pass, as directed.

Like ``modal_transition.py``, this file never imports the Stage 0/1 ASGI application and the ASGI
application never imports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from blindsight.transition.corpus import (
    CorpusName,
    ManifestClip,
    ProxyBoundary,
    build_boundary_table,
    load_egoenv_roompred_directory,
)
from blindsight.transition.encode import (
    VJEPA2_CHECKPOINT_URL,
    VJEPA2_COMMIT,
    VJEPA2WorldStateEncoder,
    WINDOW_CONFIGS,
    ExtractionWorkItem,
    ProvenanceMismatchError,
    WindowConfigName,
    assemble_run_record,
    cache_relative_directory,
    enforce_cost_budget,
    estimate_extraction_cost,
    plan_extraction_work_items,
    run_extraction_work_item,
    sha256_file,
)

APP_NAME = "blindsight-transition-encode"
DATA_VOLUME_NAME = "blindsight-transition-data"
EGO4D_SECRET_NAME = "blindsight-ego4d-aws"
DATA_MOUNT = Path("/data")
CHECKPOINT_DIRECTORY = DATA_MOUNT / "models"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
# Declared for parity with modal_transition.py's convention and for any future discovery step that
# needs authenticated S3 access; no function below currently reads AWS credentials directly, since
# encoding only reads video that acquisition has already staged on the shared volume.
ego4d_secret = modal.Secret.from_name(EGO4D_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.0",
        "numpy>=1.26",
        "timm>=1.0",
        "einops>=0.8",
        "decord>=0.6.0",
    )
    .add_local_python_source("blindsight")
)


# --------------------------------------------------------------------------------------------
# Manifest / annotation loading (read-only consumers of the shared corpus artifacts)
# --------------------------------------------------------------------------------------------


def _load_manifest_clips(manifest_path: Path) -> list[ManifestClip]:
    """Reconstruct :class:`ManifestClip` rows from the frozen manifest JSON that
    ``freeze_transition_corpus`` (in ``modal_transition.py``) already writes to the volume."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = []
    for row in payload["clips"]:
        clips.append(
            ManifestClip(
                corpus=row["corpus"],
                source_video_id=row["source_video_id"],
                clip_id=row["clip_id"],
                split=row["split"],
                evaluation_runs=tuple(row["evaluation_runs"]),
                boundary_count=row["boundary_count"],
                resolution_status=row["resolution_status"],
                resolution_reason=row["resolution_reason"],
                extraction_strategy=row["extraction_strategy"],
                source_start_frame=row.get("source_start_frame"),
                source_end_frame=row.get("source_end_frame"),
            )
        )
    return clips


def _load_boundaries(corpus: CorpusName) -> list[ProxyBoundary]:
    """Recompute proxy boundaries from the shared EgoEnv annotation copy that
    ``publish_egoenv_annotations`` (in ``modal_transition.py``) already writes to the volume.

    Boundaries are cheap to recompute (pure function over the annotation CSVs already on the
    volume) and are not part of the frozen manifest schema, so this is the intended read path
    rather than a duplicated cache.
    """

    intervals = load_egoenv_roompred_directory(DATA_MOUNT / "egoenv" / "annotations")
    boundaries, _zero_length_rows = build_boundary_table(intervals)
    return [boundary for boundary in boundaries if boundary.corpus == corpus]


def _manifest_path(manifest_name: str) -> Path:
    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise ValueError("manifest_name must be one path component.")
    return DATA_MOUNT / "frozen-corpora" / manifest_name / "manifest.json"


# --------------------------------------------------------------------------------------------
# Source video discovery -- the acquisition agent's on-volume layout is not finalized, so this
# searches by identifier instead of assuming a fixed path.
# --------------------------------------------------------------------------------------------


def discover_source_video(identifiers: list[str], *, search_root: Path = DATA_MOUNT) -> Path:
    """Find the one video file on the volume matching any of the given identifiers.

    Tries the clip identifier first (direct Ego4D clip files and HouseTours cuts are typically
    named after the clip), then the source video identifier (parent-cut Ego4D files are named
    after the parent ``video_540ss`` id). Raises with a listing of what *is* present so a failure
    is diagnosable without a second round-trip, per the issue's "discovers... fail with a clear,
    listable error if nothing is found."
    """

    for identifier in identifiers:
        if not identifier:
            continue
        candidates = sorted(
            path
            for path in search_root.rglob(f"*{identifier}*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous source video for identifier {identifier!r}: "
                f"{[str(c) for c in candidates]}"
            )
    present = sorted(
        str(path.relative_to(search_root))
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )[:50]
    raise FileNotFoundError(
        f"No source video found under {search_root} for identifiers {identifiers!r}. "
        f"First 50 video files present: {present}"
    )


# --------------------------------------------------------------------------------------------
# Checkpoint provisioning: download once, cache on the volume, reuse across containers.
# --------------------------------------------------------------------------------------------


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=1_800)
def ensure_checkpoint_cached() -> dict[str, object]:
    """Modal entrypoint for pre-warming the shared checkpoint cache; ``encode_clip_batch`` also
    calls the underlying local helper directly so a batch run never depends on this having run
    first."""

    path = ensure_checkpoint()
    return {"checkpoint_path": str(path), "checkpoint_digest": sha256_file(path)}


def ensure_checkpoint(*, checkpoint_url: str = VJEPA2_CHECKPOINT_URL) -> Path:
    """Return a local path to the pinned V-JEPA 2.1 checkpoint, downloading it to the shared
    volume once if it is not already present there."""

    from urllib.request import urlopen

    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    filename = checkpoint_url.rsplit("/", 1)[-1]
    path = CHECKPOINT_DIRECTORY / filename
    if not path.exists():
        tmp_path = path.with_suffix(path.suffix + ".partial")
        with urlopen(checkpoint_url, timeout=1800) as response, tmp_path.open("wb") as output:
            while chunk := response.read(16 * 1024 * 1024):
                output.write(chunk)
        tmp_path.rename(path)
        data_volume.commit()
    return path


# --------------------------------------------------------------------------------------------
# Frame reading (decord-backed; the one video-decode-dependent function in this file)
# --------------------------------------------------------------------------------------------


def _make_frame_reader(video_path: Path):  # type: ignore[no-untyped-def]
    """Build a ``read_frames`` callable (see ``encode.FrameWindowReader``) backed by one decord
    ``VideoReader`` over ``video_path``, reused across every window this work item needs."""

    import decord  # type: ignore[import-not-found]
    import torch

    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(str(video_path), num_threads=1)
    fps = reader.get_avg_fps()
    last_index = len(reader) - 1

    def read_frames(frame_times: list[float]) -> "torch.Tensor":
        indices = [min(last_index, max(0, round(t * fps))) for t in frame_times]
        batch = reader.get_batch(indices)  # [T, H, W, C] uint8
        frames = batch.permute(0, 3, 1, 2).to(torch.float32) / 255.0
        return frames

    return read_frames


# --------------------------------------------------------------------------------------------
# Modal functions
# --------------------------------------------------------------------------------------------


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=1_800)
def plan_corpus_encoding(
    manifest_name: str, corpus: CorpusName, window_config: WindowConfigName
) -> dict[str, object]:
    """Report the planned work items and their estimated cost for one corpus and window
    configuration, without encoding anything. This is the number the issue's "Cost and storage
    controls" section requires be read and approved before a real extraction run starts."""

    manifest_clips = [clip for clip in _load_manifest_clips(_manifest_path(manifest_name)) if clip.corpus == corpus]
    boundaries = _load_boundaries(corpus)
    guard_band_seconds, positive_seconds = _manifest_label_parameters(manifest_name)
    all_items: list[ExtractionWorkItem] = []
    durations_unknown: list[str] = []
    for clip in manifest_clips:
        if clip.split not in ("train", "heldout", "test"):
            continue
        try:
            duration = _clip_duration_seconds(clip)
        except FileNotFoundError:
            durations_unknown.append(clip.clip_id)
            continue
        all_items.extend(
            plan_extraction_work_items(
                clip,
                boundaries,
                clip_duration_seconds=duration,
                guard_band_seconds=guard_band_seconds,
                positive_seconds=positive_seconds,
            )
        )
    estimate = estimate_extraction_cost(all_items)
    return {
        "manifest_name": manifest_name,
        "corpus": corpus,
        "window_config": window_config,
        "clip_count": len(manifest_clips),
        "work_item_count": estimate.work_item_count,
        "world_state_count": estimate.world_state_count,
        "estimated_gpu_seconds": estimate.estimated_gpu_seconds,
        "estimated_retained_bytes": estimate.estimated_retained_bytes,
        "clips_with_undiscoverable_video": durations_unknown,
    }


def _manifest_label_parameters(manifest_name: str) -> tuple[float, float]:
    payload = json.loads(_manifest_path(manifest_name).read_text(encoding="utf-8"))
    return float(payload["guard_band_seconds"]), float(payload["positive_seconds"])


def _clip_duration_seconds(clip: ManifestClip) -> float:
    """The full duration of whatever source video file discovery finds for this clip.

    This is a documented simplification: for an Ego4D ``parent-cut`` clip, the discovered
    ``video_540ss`` file is the whole parent recording, not just this EgoEnv clip's span, because
    the frozen manifest schema does not currently carry the clip's start/end offset inside that
    parent (see ``blindsight/transition/corpus.py``'s ``ManifestClip`` -- only HouseTours rows
    carry ``source_start_frame``/``source_end_frame``). Treating the whole parent file as available
    duration only ever widens a work item's causal-safe range; it cannot fabricate a future frame
    or otherwise break causality, so it is a conservative placeholder pending a finalized
    Ego4D clip-offset field.
    """

    import decord  # type: ignore[import-not-found]

    identifiers = [clip.clip_id, clip.source_video_id]
    video_path = discover_source_video(identifiers)
    reader = decord.VideoReader(str(video_path))
    return len(reader) / reader.get_avg_fps()


@app.function(
    image=image,
    gpu="A100",
    secrets=[ego4d_secret],
    volumes={str(DATA_MOUNT): data_volume},
    timeout=6 * 3_600,
)
def encode_clip_batch(
    manifest_name: str, corpus: CorpusName, window_config: WindowConfigName, clip_ids: list[str]
) -> dict[str, object]:
    """Encode every planned work item for a batch of clips, for one window configuration.

    Loads the manifest, the shared boundary table, and the pinned encoder once per container, then
    processes ``clip_ids`` sequentially within this container. The caller (``run_encoding``) fans
    out many such batches in parallel across containers via ``.map()``.
    """

    manifest_clips = {clip.clip_id: clip for clip in _load_manifest_clips(_manifest_path(manifest_name))}
    boundaries = _load_boundaries(corpus)
    guard_band_seconds, positive_seconds = _manifest_label_parameters(manifest_name)
    checkpoint_path = ensure_checkpoint()
    checkpoint_digest = sha256_file(checkpoint_path)
    encoder = VJEPA2WorldStateEncoder(checkpoint_path)
    window = WINDOW_CONFIGS[window_config]

    results: list[dict[str, object]] = []
    for clip_id in clip_ids:
        clip = manifest_clips.get(clip_id)
        if clip is None:
            results.append({"clip_id": clip_id, "status": "failed", "error": "not in manifest"})
            continue
        try:
            video_path = discover_source_video([clip.clip_id, clip.source_video_id])
            duration = _clip_duration_seconds(clip)
            items = plan_extraction_work_items(
                clip,
                boundaries,
                clip_duration_seconds=duration,
                guard_band_seconds=guard_band_seconds,
                positive_seconds=positive_seconds,
            )
            read_frames = _make_frame_reader(video_path)
            cache_dir = DATA_MOUNT / cache_relative_directory(manifest_name, corpus)
            item_results = []
            for item in items:
                item_results.append(
                    run_extraction_work_item(
                        item,
                        window=window,
                        read_frames=read_frames,
                        encode_window=encoder.encode_window,
                        cache_dir=cache_dir,
                        manifest_name=manifest_name,
                        encoder_commit=encoder.encoder_commit,
                        checkpoint_digest=checkpoint_digest,
                        guard_band_seconds=guard_band_seconds,
                        positive_seconds=positive_seconds,
                    )
                )
            results.append({"clip_id": clip_id, "status": "completed", "items": item_results})
        except ProvenanceMismatchError as exc:
            results.append({"clip_id": clip_id, "status": "failed", "error": str(exc)})
        except FileNotFoundError as exc:
            results.append({"clip_id": clip_id, "status": "failed", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - record every planned item's outcome, never crash the batch
            results.append({"clip_id": clip_id, "status": "failed", "error": repr(exc)})
    data_volume.commit()
    return {
        "manifest_name": manifest_name,
        "corpus": corpus,
        "window_config": window_config,
        "checkpoint_digest": checkpoint_digest,
        "encoder_commit": VJEPA2_COMMIT,
        "results": results,
    }


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=3_600)
def run_encoding(
    manifest_name: str,
    corpus: CorpusName,
    window_config: WindowConfigName,
    *,
    batch_size: int = 4,
    max_gpu_seconds: float | None = None,
    max_retained_bytes: int | None = None,
) -> dict[str, object]:
    """Fan out clip-batch encoding across parallel Modal GPU containers for one corpus and window
    configuration, after checking the planned run against an accepted cost budget."""

    manifest_clips = [clip for clip in _load_manifest_clips(_manifest_path(manifest_name)) if clip.corpus == corpus]
    boundaries = _load_boundaries(corpus)
    guard_band_seconds, positive_seconds = _manifest_label_parameters(manifest_name)
    eligible = [clip for clip in manifest_clips if clip.split in ("train", "heldout", "test")]

    all_items: list[ExtractionWorkItem] = []
    for clip in eligible:
        try:
            duration = _clip_duration_seconds(clip)
        except FileNotFoundError:
            continue
        all_items.extend(
            plan_extraction_work_items(
                clip,
                boundaries,
                clip_duration_seconds=duration,
                guard_band_seconds=guard_band_seconds,
                positive_seconds=positive_seconds,
            )
        )
    estimate = estimate_extraction_cost(all_items)
    enforce_cost_budget(estimate, max_gpu_seconds=max_gpu_seconds, max_retained_bytes=max_retained_bytes)

    clip_ids = [clip.clip_id for clip in eligible]
    batches = [clip_ids[i : i + batch_size] for i in range(0, len(clip_ids), batch_size)]
    return assemble_run_record(
        manifest_name=manifest_name,
        corpus=corpus,
        window_config=window_config,
        estimate=estimate,
        batch_count=len(batches),
        clip_count=len(clip_ids),
        fan_out=lambda: encode_clip_batch.starmap(
            [(manifest_name, corpus, window_config, batch) for batch in batches]
        ),
    )


@app.local_entrypoint()
def main(
    action: str = "plan",
    manifest_name: str = "",
    corpus: str = "ego4d",
    window_config: str = "short",
    batch_size: int = 4,
    max_gpu_seconds: float = 0.0,
    max_retained_bytes: int = 0,
    clip_ids: str = "",
) -> None:
    if action == "plan":
        result = plan_corpus_encoding.remote(manifest_name, corpus, window_config)  # type: ignore[arg-type]
    elif action == "run":
        result = run_encoding.remote(
            manifest_name,
            corpus,  # type: ignore[arg-type]
            window_config,  # type: ignore[arg-type]
            batch_size=batch_size,
            max_gpu_seconds=max_gpu_seconds or None,
            max_retained_bytes=max_retained_bytes or None,
        )
    elif action == "ensure-checkpoint":
        result = ensure_checkpoint_cached.remote()
    elif action == "encode-clips":
        # CLI-only convenience wrapper around encode_clip_batch: the CLI cannot parse a bare
        # list[str] parameter, so this takes a comma-separated string instead and splits it here.
        # This exists so a small, verifiable real encode can be run as one literal `modal run`
        # command (see docs/spec/phase-3-transition.md's "verify one clip before a large plan"
        # convention) without inventing a second code path for the actual encode logic.
        parsed_clip_ids = [value for value in clip_ids.split(",") if value]
        if not parsed_clip_ids:
            raise ValueError("action=encode-clips requires --clip-ids as a comma-separated list.")
        result = encode_clip_batch.remote(manifest_name, corpus, window_config, parsed_clip_ids)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown action {action!r}.")
    print(json.dumps(result, indent=2, sort_keys=True))
