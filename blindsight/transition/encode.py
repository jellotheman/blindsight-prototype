"""Causal world-state encoding for Stage 3 (see ``docs/spec/phase-3-transition.md``, issue #21).

This module has two halves with very different testability.

**Pure logic** (window/timestamp math, preprocessing tensor ops, extraction planning, cache I/O,
and work-item orchestration) needs no GPU, no network, and no real video. Every function down to
``run_extraction_work_item`` is unit-tested with synthetic tensors and fakes.

**The V-JEPA 2.1 encoder wrapper** (:class:`VJEPA2WorldStateEncoder`) is the one GPU/network-
dependent seam: it clones the pinned upstream commit through ``torch.hub`` and loads the pinned
checkpoint. It is exercised by a real smoke encode, not by the unit-test suite.

The Modal wrapper in ``modal_transition_encode.py`` supplies the concrete frame reader (video
decode) and calls into ``run_extraction_work_item`` here so the orchestration logic -- idempotency,
provenance checks, the cache-format contract -- is identical in tests and in production.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from blindsight.transition.corpus import CorpusName, ManifestClip, ProxyBoundary, SplitName

WindowConfigName = Literal["short", "long"]

EMISSION_HZ = 1.0
GRU_HISTORY_SECONDS = 8.0


# --------------------------------------------------------------------------------------------
# Window configurations
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowConfig:
    """One frame-sampling configuration. ``span_seconds`` is the nominal duration the spec quotes
    (``frame_count / sample_rate_hz``), not the exact first-to-last-frame span, which is one sample
    interval shorter."""

    name: WindowConfigName
    sample_rate_hz: float
    frame_count: int

    @property
    def span_seconds(self) -> float:
        return self.frame_count / self.sample_rate_hz


WINDOW_CONFIGS: dict[WindowConfigName, WindowConfig] = {
    "short": WindowConfig("short", sample_rate_hz=30.0, frame_count=64),
    "long": WindowConfig("long", sample_rate_hz=4.0, frame_count=64),
}


def emission_timestamps(
    range_start_seconds: float, range_end_seconds: float, *, emission_hz: float = EMISSION_HZ
) -> list[float]:
    """1 Hz absolute clip-time timestamps covering ``[range_start, range_end]`` inclusive.

    Both endpoints are included when they land on the sampling grid, so a full-clip range always
    emits a timestamp at the clip's final whole second.
    """

    if emission_hz <= 0:
        raise ValueError("emission_hz must be positive")
    if range_end_seconds < range_start_seconds:
        raise ValueError("range_end_seconds cannot be before range_start_seconds")
    step = 1.0 / emission_hz
    span = range_end_seconds - range_start_seconds
    count = int(span / step + 1e-9) + 1
    return [range_start_seconds + index * step for index in range(count)]


def trailing_window_frame_times(emission_time_seconds: float, window: WindowConfig) -> list[float]:
    """The ascending, causal source-frame timestamps for one trailing window ending at ``t``.

    Frame ``k`` (counting back from the most recent) sits at ``t - k / sample_rate_hz``. Near the
    start of a clip this goes negative -- a full window is not yet available. Rather than fabricate
    padding frames (which would inject content that never existed), every negative sample time is
    clamped to 0.0, so the earliest available frame is repeated instead. This keeps the window
    strictly causal: every returned time is still ``<= t`` (clamping only ever raises a negative
    value up to 0, and ``t >= 0`` always holds for an emission timestamp).
    """

    if emission_time_seconds < 0:
        raise ValueError("emission_time_seconds cannot be negative")
    spacing = 1.0 / window.sample_rate_hz
    raw_descending = [emission_time_seconds - step * spacing for step in range(window.frame_count)]
    return [max(0.0, value) for value in reversed(raw_descending)]


# --------------------------------------------------------------------------------------------
# Preprocessing (pure tensor ops)
# --------------------------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SHORT_SIDE = 438
CROP_SIZE = 384


def resize_short_side(frames: torch.Tensor, *, short_side: int = RESIZE_SHORT_SIDE) -> torch.Tensor:
    """Resize a ``[..., C, H, W]`` frame batch so its shorter spatial side equals ``short_side``."""

    if frames.ndim < 3:
        raise ValueError("frames must have shape [..., C, H, W]")
    *leading, channels, height, width = frames.shape
    if height <= width:
        new_height = short_side
        new_width = max(1, round(width * short_side / height))
    else:
        new_width = short_side
        new_height = max(1, round(height * short_side / width))
    flat = frames.reshape(-1, channels, height, width).to(torch.float32)
    resized = F.interpolate(
        flat, size=(new_height, new_width), mode="bilinear", align_corners=False, antialias=True
    )
    return resized.reshape(*leading, channels, new_height, new_width)


def center_crop(frames: torch.Tensor, *, size: int = CROP_SIZE) -> torch.Tensor:
    """Take a centered ``size`` by ``size`` crop from a ``[..., C, H, W]`` frame batch."""

    *_, height, width = frames.shape
    if height < size or width < size:
        raise ValueError(f"Cannot center-crop to {size}: frame is {height}x{width}")
    top = (height - size) // 2
    left = (width - size) // 2
    return frames[..., top : top + size, left : left + size]


def normalize_imagenet(frames: torch.Tensor) -> torch.Tensor:
    """Apply the ImageNet mean/std normalization to a ``[..., 3, H, W]`` tensor scaled to ``[0, 1]``."""

    if frames.shape[-3] != 3:
        raise ValueError("normalize_imagenet expects 3 channels")
    mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype, device=frames.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=frames.dtype, device=frames.device).view(3, 1, 1)
    return (frames - mean) / std


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """Full V-JEPA 2.1 preprocessing: resize short side to 438px, center-crop 384, normalize.

    ``frames`` is ``[T, 3, H, W]``, float, scaled to ``[0, 1]``. Returns ``[T, 3, 384, 384]``.
    """

    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must be [T, 3, H, W]")
    resized = resize_short_side(frames)
    cropped = center_crop(resized)
    return normalize_imagenet(cropped)


# --------------------------------------------------------------------------------------------
# Extraction planning
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionRange:
    """One contiguous span of clip time, in absolute clip-time seconds."""

    start_seconds: float
    end_seconds: float


def boundary_context_range(
    boundary_time_seconds: float,
    *,
    guard_band_seconds: float,
    positive_seconds: float,
    clip_duration_seconds: float,
    gru_history_seconds: float = GRU_HISTORY_SECONDS,
) -> ExtractionRange:
    """The context window one training boundary needs, clamped to the clip's own duration.

    Margin = ``guard_band_seconds + positive_seconds + gru_history_seconds`` on each side. The
    guard band and the positive range are exactly the labeled span around the boundary (see
    ``corpus.label_time_steps``); the extra ``gru_history_seconds`` (documented default: 8 seconds,
    matching the causal GRU head's 8-step history in ``docs/spec/phase-3-transition.md``, "Detector")
    gives the recurrent detector enough causal lead-in that its hidden state is not empty by the
    time labeled scoring starts. This is a first cut, not a tuned constant -- a wider margin only
    costs a little more encode time, so it errs generous.
    """

    if clip_duration_seconds < 0:
        raise ValueError("clip_duration_seconds cannot be negative")
    margin = guard_band_seconds + positive_seconds + gru_history_seconds
    start = max(0.0, boundary_time_seconds - margin)
    end = min(clip_duration_seconds, boundary_time_seconds + margin)
    return ExtractionRange(start, end)


def merge_ranges(ranges: Sequence[ExtractionRange]) -> list[ExtractionRange]:
    """Merge overlapping or touching ranges so nearby boundaries share one encoding pass."""

    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: (item.start_seconds, item.end_seconds))
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if current.start_seconds <= last.end_seconds:
            if current.end_seconds > last.end_seconds:
                merged[-1] = ExtractionRange(last.start_seconds, current.end_seconds)
        else:
            merged.append(current)
    return merged


def sanitize_identifier(value: str) -> str:
    """Replace filesystem-unsafe characters (notably ``/``) so an identifier is a safe path segment."""

    return re.sub(r"[^A-Za-z0-9_.-]", "-", value)


@dataclass(frozen=True)
class ExtractionWorkItem:
    """One ``(clip, window_config, contiguous range)`` unit. Each work item becomes exactly one
    cache file pair.

    ``file_stem`` disambiguates multiple work items from the same clip: a held-out or test clip
    always has exactly one item, whose stem is the clip's own sanitized identifier. A training clip
    can need several disjoint boundary windows (the spec's compute-cost section requires encoding
    only "a window around each boundary", not the whole clip, and boundaries in a ~16-minute Ego4D
    clip can sit far enough apart that their context windows never merge into one span) -- each such
    window gets its own file, named ``<clip>__boundary<index>``, where ``index`` is the position of
    that merged range in ascending start-time order (stable and reproducible across runs). The
    ``range_start_seconds``/``range_end_seconds`` cache-metadata fields always describe this one
    item's own range, never the whole clip, for a training item.
    """

    corpus: CorpusName
    source_video_id: str
    clip_id: str
    split: SplitName
    range_start_seconds: float
    range_end_seconds: float
    emission_timestamps: tuple[float, ...]
    file_stem: str


def plan_extraction_work_items(
    clip: ManifestClip,
    boundaries: Sequence[ProxyBoundary],
    *,
    clip_duration_seconds: float,
    guard_band_seconds: float,
    positive_seconds: float,
    gru_history_seconds: float = GRU_HISTORY_SECONDS,
    emission_hz: float = EMISSION_HZ,
) -> list[ExtractionWorkItem]:
    """The extraction work items one manifest clip needs, independent of window configuration.

    Held-out and test clips always get exactly one item covering the whole clip -- "Held-out clips:
    encode the full clip, always" (phase-3-transition.md, "Compute cost"), because the false-trigger
    measurement needs continuous ordinary video. Training clips get one item per merged boundary
    context range. A clip outside ``{train, heldout, test}`` (``not-selected``, ``unresolved``,
    ``unavailable``) plans nothing.
    """

    if clip.split in ("heldout", "test"):
        timestamps = emission_timestamps(0.0, clip_duration_seconds, emission_hz=emission_hz)
        return [
            ExtractionWorkItem(
                corpus=clip.corpus,
                source_video_id=clip.source_video_id,
                clip_id=clip.clip_id,
                split=clip.split,
                range_start_seconds=0.0,
                range_end_seconds=clip_duration_seconds,
                emission_timestamps=tuple(timestamps),
                file_stem=sanitize_identifier(clip.clip_id),
            )
        ]
    if clip.split != "train":
        return []
    clip_boundaries = [
        boundary
        for boundary in boundaries
        if boundary.corpus == clip.corpus and boundary.clip_id == clip.clip_id
    ]
    if not clip_boundaries:
        return []
    ranges = [
        boundary_context_range(
            boundary.boundary_time,
            guard_band_seconds=guard_band_seconds,
            positive_seconds=positive_seconds,
            clip_duration_seconds=clip_duration_seconds,
            gru_history_seconds=gru_history_seconds,
        )
        for boundary in clip_boundaries
    ]
    merged = merge_ranges(ranges)
    items = []
    for index, span in enumerate(merged):
        timestamps = emission_timestamps(span.start_seconds, span.end_seconds, emission_hz=emission_hz)
        items.append(
            ExtractionWorkItem(
                corpus=clip.corpus,
                source_video_id=clip.source_video_id,
                clip_id=clip.clip_id,
                split=clip.split,
                range_start_seconds=span.start_seconds,
                range_end_seconds=span.end_seconds,
                emission_timestamps=tuple(timestamps),
                file_stem=f"{sanitize_identifier(clip.clip_id)}__boundary{index}",
            )
        )
    return items


# --------------------------------------------------------------------------------------------
# Cost and storage controls
# --------------------------------------------------------------------------------------------

GPU_SECONDS_PER_WINDOW = 0.30
"""A100-40GB, bfloat16 autocast, FlashAttention, uint8 frame transfer (phase-3-transition.md,
"Compute cost"). There is no further large saving in the encoder itself, so this is the right unit
to budget against."""

_WORLD_STATE_BYTES = 1024 * 4  # one float32 world state
_TIMESTAMP_BYTES = 8  # one float64 timestamp


@dataclass(frozen=True)
class ExtractionCostEstimate:
    work_item_count: int
    world_state_count: int
    estimated_gpu_seconds: float
    estimated_retained_bytes: int


def estimate_extraction_cost(work_items: Sequence[ExtractionWorkItem]) -> ExtractionCostEstimate:
    """Estimate GPU time and retained bytes for a planned batch of work items, before any of it
    runs. This is the number the issue's "Cost and storage controls" section requires be read and
    checked against an accepted plan before extraction starts."""

    world_state_count = sum(len(item.emission_timestamps) for item in work_items)
    return ExtractionCostEstimate(
        work_item_count=len(work_items),
        world_state_count=world_state_count,
        estimated_gpu_seconds=world_state_count * GPU_SECONDS_PER_WINDOW,
        estimated_retained_bytes=world_state_count * (_WORLD_STATE_BYTES + _TIMESTAMP_BYTES),
    )


class CostBudgetExceededError(RuntimeError):
    """Raised when a planned extraction run exceeds its accepted GPU-time or storage guard."""


def enforce_cost_budget(
    estimate: ExtractionCostEstimate,
    *,
    max_gpu_seconds: float | None = None,
    max_retained_bytes: int | None = None,
) -> None:
    """Reject a run that exceeds a declared guard without a new approved plan.

    Either bound is optional so a caller can enforce just one axis; passing neither always passes.
    """

    if max_gpu_seconds is not None and estimate.estimated_gpu_seconds > max_gpu_seconds:
        raise CostBudgetExceededError(
            f"Estimated {estimate.estimated_gpu_seconds:.1f} GPU-seconds exceeds the accepted "
            f"budget of {max_gpu_seconds:.1f}. Approve a new plan before running this extraction."
        )
    if max_retained_bytes is not None and estimate.estimated_retained_bytes > max_retained_bytes:
        raise CostBudgetExceededError(
            f"Estimated {estimate.estimated_retained_bytes} retained bytes exceeds the accepted "
            f"budget of {max_retained_bytes}. Approve a new plan before running this extraction."
        )


# --------------------------------------------------------------------------------------------
# Cache format contract
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheMetadata:
    """The sibling JSON metadata file for one cache entry. Field set and names are the shared
    cache-format contract that the detector-training pipeline depends on byte-for-byte."""

    corpus: str
    source_video_id: str
    clip_id: str
    split: str
    window_config: str
    manifest_name: str
    guard_band_seconds: float
    positive_seconds: float
    encoder_commit: str
    checkpoint_digest: str
    range_start_seconds: float
    range_end_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ProvenanceMismatchError(RuntimeError):
    """Raised when a cache entry already exists on disk with different provenance metadata."""


def cache_file_stem(work_item_stem: str, window_config: WindowConfigName) -> str:
    return f"{work_item_stem}__{window_config}"


def cache_relative_directory(manifest_name: str, corpus: CorpusName) -> Path:
    """The path layout the contract fixes: ``world-states/<manifest_name>/<corpus>/``."""

    return Path("world-states") / manifest_name / corpus


def _validate_trace(world_states: np.ndarray, timestamps: np.ndarray) -> None:
    if world_states.ndim != 2 or world_states.shape[1] != 1024:
        raise ValueError(f"world_states must be [T, 1024], got {world_states.shape}")
    if world_states.dtype != np.float32:
        raise ValueError(f"world_states must be float32, got {world_states.dtype}")
    if timestamps.ndim != 1 or timestamps.shape[0] != world_states.shape[0]:
        raise ValueError("timestamps must be one value for each world state")
    if timestamps.dtype != np.float64:
        raise ValueError(f"timestamps must be float64, got {timestamps.dtype}")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing")


def write_world_state_cache(
    directory: Path,
    stem: str,
    *,
    world_states: np.ndarray,
    timestamps: np.ndarray,
    metadata: CacheMetadata,
) -> tuple[Path, Path]:
    """Write one cache file pair per the shared contract: ``.npz`` with exactly ``world_states``
    and ``timestamps``, plus a sibling ``.json`` metadata file."""

    _validate_trace(world_states, timestamps)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{stem}.npz"
    json_path = directory / f"{stem}.json"
    np.savez(str(npz_path), world_states=world_states, timestamps=timestamps)
    json_path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return npz_path, json_path


def read_cache_metadata(json_path: Path) -> dict[str, object] | None:
    if not json_path.exists():
        return None
    value: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
    return value


# --------------------------------------------------------------------------------------------
# Work-item orchestration (GPU-independent: the frame reader and encoder are injected)
# --------------------------------------------------------------------------------------------

FrameWindowReader = Callable[[Sequence[float]], torch.Tensor]
"""Given ascending, causal absolute clip-time frame timestamps, returns raw ``[T, 3, H, W]``
frames scaled to ``[0, 1]``, one frame per requested timestamp, in the same order."""

WindowEncoder = Callable[[torch.Tensor], torch.Tensor]
"""Given a preprocessed ``[T, 3, 384, 384]`` window, returns a ``[1024]`` float32 world state."""


def run_extraction_work_item(
    item: ExtractionWorkItem,
    *,
    window: WindowConfig,
    read_frames: FrameWindowReader,
    encode_window: WindowEncoder,
    cache_dir: Path,
    manifest_name: str,
    encoder_commit: str,
    checkpoint_digest: str,
    guard_band_seconds: float,
    positive_seconds: float,
) -> dict[str, object]:
    """Encode one work item end to end and write its cache pair, idempotently.

    A completed work item whose existing metadata matches this call's provenance exactly is skipped
    without re-encoding. An existing cache entry whose metadata differs raises
    :class:`ProvenanceMismatchError` rather than being silently overwritten. This function contains
    every GPU-independent behavior the acceptance criteria name (idempotency, provenance safety, the
    cache-format contract); only ``read_frames`` and ``encode_window`` touch real video or the real
    model, so this whole path is exercised by fakes in the unit-test suite.
    """

    stem = cache_file_stem(item.file_stem, window.name)
    npz_path = cache_dir / f"{stem}.npz"
    json_path = cache_dir / f"{stem}.json"
    metadata = CacheMetadata(
        corpus=item.corpus,
        source_video_id=item.source_video_id,
        clip_id=item.clip_id,
        split=item.split,
        window_config=window.name,
        manifest_name=manifest_name,
        guard_band_seconds=guard_band_seconds,
        positive_seconds=positive_seconds,
        encoder_commit=encoder_commit,
        checkpoint_digest=checkpoint_digest,
        range_start_seconds=item.range_start_seconds,
        range_end_seconds=item.range_end_seconds,
    )
    existing = read_cache_metadata(json_path)
    if existing is not None:
        if existing != metadata.to_dict():
            raise ProvenanceMismatchError(
                f"{json_path} already exists with different provenance: {existing} != {metadata.to_dict()}"
            )
        if npz_path.exists():
            return {"stem": stem, "status": "skipped-idempotent", "world_state_count": len(item.emission_timestamps)}
        # Metadata survived without its data file (an interrupted prior run) -- fall through and
        # re-encode so the pair is complete again.

    world_states: list[torch.Tensor] = []
    for timestamp in item.emission_timestamps:
        frame_times = trailing_window_frame_times(timestamp, window)
        if any(frame_time > timestamp for frame_time in frame_times):
            raise AssertionError("trailing_window_frame_times produced a future frame")
        raw_frames = read_frames(frame_times)
        preprocessed = preprocess_frames(raw_frames)
        world_states.append(encode_window(preprocessed).to(torch.float32))

    world_states_arr = torch.stack(world_states).cpu().numpy().astype(np.float32)
    timestamps_arr = np.array(
        [timestamp - item.range_start_seconds for timestamp in item.emission_timestamps], dtype=np.float64
    )
    write_world_state_cache(
        cache_dir, stem, world_states=world_states_arr, timestamps=timestamps_arr, metadata=metadata
    )
    return {"stem": stem, "status": "encoded", "world_state_count": len(item.emission_timestamps)}


# --------------------------------------------------------------------------------------------
# V-JEPA 2.1 encoder wrapper (GPU/network-dependent; not exercised by the unit-test suite)
# --------------------------------------------------------------------------------------------

VJEPA2_REPO = "facebookresearch/vjepa2"
VJEPA2_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA2_HUB_ENTRYPOINT = "vjepa2_1_vit_large_384"
VJEPA2_CHECKPOINT_FILENAME = "vjepa2_1_vitl_dist_vitG_384.pt"
VJEPA2_CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/vjepa2/{VJEPA2_CHECKPOINT_FILENAME}"
VJEPA2_CHECKPOINT_STATE_DICT_KEY = "ema_encoder"
VJEPA2_EMBED_DIM = 1024


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_backbone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the ``module.``/``backbone.`` prefixes the released checkpoint stores its weights
    under -- the same cleaning step ``facebookresearch/vjepa2``'s own ``hubconf`` path applies at
    the pinned commit, so the official checkpoint loads with ``strict=True``."""

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def mean_pool_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``[B, N, D]`` (or ``[N, D]``) patch tokens over the patch dimension, as float32."""

    if tokens.ndim == 2:
        return tokens.mean(dim=0).to(torch.float32)
    return tokens.mean(dim=1).to(torch.float32)


class VJEPA2WorldStateEncoder:
    """Loads the pinned V-JEPA 2.1 distilled ViT-L/384 encoder and mean-pools its patch tokens.

    This is the one GPU/network-dependent class in the module: constructing it clones the pinned
    ``facebookresearch/vjepa2`` commit through ``torch.hub`` and loads a local checkpoint file.
    Every other function above is unit-tested without it; this class is exercised by a real smoke
    encode instead (see ``tests/test_transition_encode.py``'s ``live``-marked test, skipped by
    default).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        repo: str = VJEPA2_REPO,
        commit: str = VJEPA2_COMMIT,
        hub_entrypoint: str = VJEPA2_HUB_ENTRYPOINT,
        state_dict_key: str = VJEPA2_CHECKPOINT_STATE_DICT_KEY,
    ) -> None:
        self.device = device
        self.encoder_commit = commit
        self.checkpoint_digest = sha256_file(checkpoint_path)
        encoder, _predictor = torch.hub.load(
            f"{repo}:{commit}", hub_entrypoint, pretrained=False, trust_repo=True
        )
        # weights_only=False: the official checkpoint is a plain state-dict-of-dicts from a trusted,
        # MIT-licensed first-party source, not an arbitrary pickle from the network. Do not pass
        # mmap=True here: it was verified (2026-08-30) to segfault on at least one real Windows /
        # Python 3.14 / torch-cu128 combination when materializing tensors out of this exact
        # checkpoint, while a plain (non-mmap) load of the same file succeeds cleanly on a standard
        # Python 3.12 / torch-cpu build. A full, non-mmap load is the portable choice.
        raw_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_state = _clean_backbone_state_dict(raw_state_dict[state_dict_key])
        encoder.load_state_dict(encoder_state, strict=True)
        encoder.eval()
        self._encoder = encoder.to(device)

    @torch.no_grad()
    def encode_window(self, frames_chw: torch.Tensor) -> torch.Tensor:
        """``frames_chw``: a preprocessed ``[T, 3, 384, 384]`` window. Returns a ``[1024]`` float32
        world state: the encoder's patch tokens, mean-pooled, under bfloat16 autocast."""

        if frames_chw.ndim != 4 or frames_chw.shape[1] != 3:
            raise ValueError("frames_chw must be [T, 3, H, W]")
        video = frames_chw.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # [1, 3, T, H, W]
        autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
            tokens = self._encoder(video)
        return mean_pool_patch_tokens(tokens).squeeze(0).to("cpu", torch.float32)
