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
    segment Duration/seek metadata once recording stops, and browsers record it as VP8/VP9.
    `ffprobe`'s codec/dimension check accepts the result, but Reka's ingestion cannot decode
    that container/codec: after parsing the clip (``Expected 6 frames``) its decoder yields no
    frames (``got 0``). Transcoding the live WebM to H.264 MP4 -- the container/codec pair the
    preloaded excerpt path already hands Reka successfully -- fixes both defects at once. An
    already-MP4 live capture (e.g. Safari) only needs the streaming metadata repaired, so it is
    copy-remuxed without a lossy re-encode.
    """

    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return evidence
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_suffix = ".webm" if evidence.media_type == "video/webm" else ".mp4"
            source = Path(tmp_dir) / f"input{source_suffix}"
            target = Path(tmp_dir) / "output.mp4"
            source.write_bytes(evidence.content)
            if evidence.media_type == "video/webm":
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
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(target),
                ]
                media_type = "video/mp4"
            else:
                command = [ffmpeg, "-y", "-i", str(source), "-c", "copy", str(target)]
                media_type = evidence.media_type
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return evidence
            if result.returncode != 0 or not target.exists():
                return evidence
            return replace(evidence, content=target.read_bytes(), media_type=media_type)


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
