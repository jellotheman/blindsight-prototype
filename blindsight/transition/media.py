"""Media framing for the Stage 3 transition worker.

The worker consumes queued transition chunks that are not independent video files: browser
MediaRecorder emits streaming-mode WebM/fMP4 fragments, and an individual fragment is not
standalone decodable. This module is the seam between the chunk queue and frame extraction.
It splices a contiguous chunk prefix into one byte stream, repairs it exactly as Stage 0
repairs live captures (copy-remux already-H.264 MP4; transcode WebM and every other
codec/container to H.264 MP4), and validates decodability before any GPU spend. Container
metadata reported by MediaRecorder is never trusted; decodability is measured with
ffmpeg/ffprobe against the bytes themselves.

Public surface: ``probe_decodable`` and ``splice_to_decodable_span``. Both return structured
results and never raise on unusable media, so the production worker can call them unchanged
in a later Stage 3 ticket.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PROBE_TIMEOUT_SECONDS = 30
_REPAIR_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of probing raw media bytes; ``decodable`` requires one decoded frame."""

    decodable: bool
    codec: str | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SpliceResult:
    """Outcome of splicing contiguous chunks; ``content`` is None when the splice failed."""

    content: bytes | None
    media_type: str
    duration_seconds: float | None = None
    frame_count: int | None = None
    reason: str | None = None


def _suffix_for(media_type: str) -> str:
    if media_type == "video/quicktime":
        return ".mov"
    if media_type == "video/mp4":
        return ".mp4"
    return ".webm"


def _probe_codec_name(path: Path) -> str | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    codec = result.stdout.decode().strip()
    return codec if result.returncode == 0 and codec else None


def _ffprobe_bytes(path: Path) -> ProbeResult:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return ProbeResult(decodable=False, reason="ffprobe is not available.")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,nb_read_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return ProbeResult(decodable=False, reason=f"ffprobe could not run: {error}")
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        return ProbeResult(decodable=False, reason=detail or "ffprobe reported an error.")
    try:
        report = json.loads(result.stdout.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ProbeResult(decodable=False, reason="The ffprobe output was not readable.")
    streams = report.get("streams") or []
    if not streams:
        return ProbeResult(decodable=False, reason="No video stream is present.")
    stream = streams[0]
    raw_frames = stream.get("nb_read_frames")
    try:
        frame_count = int(raw_frames) if raw_frames is not None else None
    except (TypeError, ValueError):
        frame_count = None
    if frame_count is None or frame_count <= 0:
        return ProbeResult(
            decodable=False,
            codec=stream.get("codec_name"),
            reason="The container parsed but no frame decoded.",
        )
    raw_duration = report.get("format", {}).get("duration")
    try:
        duration_seconds = float(raw_duration) if raw_duration is not None else None
    except (TypeError, ValueError):
        duration_seconds = None
    return ProbeResult(
        decodable=True,
        codec=stream.get("codec_name"),
        duration_seconds=duration_seconds,
        frame_count=frame_count,
    )


def probe_decodable(data: bytes, media_type: str) -> ProbeResult:
    """Probe raw media bytes: decodable means the container parses and at least one frame decodes."""
    if not data:
        return ProbeResult(decodable=False, reason="The media bytes are empty.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"input{_suffix_for(media_type)}"
        path.write_bytes(data)
        return _ffprobe_bytes(path)


def splice_to_decodable_span(
    chunks: list[tuple[bytes, str]], media_type: str
) -> SpliceResult:
    """Splice contiguous chunks into one repaired, validated H.264 MP4 span.

    The chunks are concatenated in order (they form one streaming byte sequence), repaired
    with ffmpeg -- copy-remux when the spliced stream already probes as H.264 MP4, otherwise
    transcoded to H.264 MP4 as Stage 0 does -- and then probed before returning. Any unusable
    input returns a structured failure; this function never raises and never trusts container
    metadata.
    """
    if not chunks:
        return SpliceResult(content=None, media_type=media_type, reason="No chunks were supplied.")
    joined = b"".join(content for content, _ in chunks)
    if not joined:
        return SpliceResult(
            content=None, media_type=media_type, reason="The chunk bytes are empty."
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return SpliceResult(content=None, media_type=media_type, reason="ffmpeg is not available.")
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = Path(tmp_dir) / f"spliced{_suffix_for(media_type)}"
        target = Path(tmp_dir) / "repaired.mp4"
        source.write_bytes(joined)
        if media_type == "video/mp4" and _probe_codec_name(source) == "h264":
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(target),
            ]
        else:
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-an",
                str(target),
            ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=_REPAIR_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return SpliceResult(
                content=None, media_type=media_type, reason=f"ffmpeg could not run: {error}"
            )
        if result.returncode != 0 or not target.exists():
            detail = result.stderr.decode(errors="replace").strip()[-500:]
            return SpliceResult(
                content=None,
                media_type=media_type,
                reason=detail or "ffmpeg could not repair the spliced media.",
            )
        probe = _ffprobe_bytes(target)
        if not probe.decodable:
            return SpliceResult(content=None, media_type=media_type, reason=probe.reason)
        return SpliceResult(
            content=target.read_bytes(),
            media_type="video/mp4",
            duration_seconds=probe.duration_seconds,
            frame_count=probe.frame_count,
        )
