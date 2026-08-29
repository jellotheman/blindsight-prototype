"""Captured-view media validation before any provider spend."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .providers import CaptureEvidence


class MediaValidator(Protocol):
    def is_decodable(self, evidence: CaptureEvidence) -> bool: ...


class MediaRemuxer(Protocol):
    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence: ...


class PassthroughMediaRemuxer:
    """No-op remuxer for excerpt evidence and test doubles that need no repair."""

    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence:
        return evidence


class FfmpegChunkRemuxer:
    """Repairs a live capture assembled by raw concatenation of streamed chunks.

    `MediaRecorder` writes WebM/Matroska in streaming mode and never seeks back to patch the
    segment Duration/seek metadata once recording stops. `ffprobe`'s codec/dimension check
    accepts the result, but Reka's ingestion rejects it outright as invalid video metadata.
    Copy-remuxing through `ffmpeg` into a freshly-seekable file lets the muxer write the
    metadata that streaming mode omitted, without re-encoding.
    """

    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return evidence
        suffix = ".webm" if evidence.media_type == "video/webm" else ".mp4"
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / f"input{suffix}"
            target = Path(tmp_dir) / f"output{suffix}"
            source.write_bytes(evidence.content)
            try:
                result = subprocess.run(
                    [ffmpeg, "-y", "-i", str(source), "-c", "copy", str(target)],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return evidence
            if result.returncode != 0 or not target.exists():
                return evidence
            return replace(evidence, content=target.read_bytes())


class FfprobeMediaValidator:
    def is_decodable(self, evidence: CaptureEvidence) -> bool:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return False
        suffix = ".mp4" if evidence.media_type == "video/mp4" else ".webm"
        if evidence.media_type == "image/jpeg":
            suffix = ".jpg"
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(evidence.content)
                path = Path(handle.name)
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return False
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
