"""Provider-facing media repair for live captures.

The reference client records VP9/VP8-in-WebM with MediaRecorder and patches the container
duration in the browser, but Reka's ingestion still yields zero decoded frames from that
bitstream (`Expected 6 frames, got 0`) while it processes H.264 MP4 -- the excerpt path --
without complaint. The remuxer is the server-side seam that converts an assembled live WebM
into the container/codec pair every provider attempt is known to accept, before any provider
spend and before the clip is retained as evidence.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .providers import CaptureEvidence


class MediaRemuxer(Protocol):
    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence: ...


class PassthroughMediaRemuxer:
    """No-op remuxer for excerpt evidence and test doubles that need no repair."""

    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence:
        return evidence


class FfmpegChunkRemuxer:
    """Transcodes a live WebM capture to H.264 MP4 before any provider call.

    MediaRecorder streams WebM whose VP8/VP9 bitstream Reka's frame extraction cannot decode
    (`Expected 6 frames, got 0`) even once the container duration is patched. H.264 MP4 is the
    container/codec pair the excerpt path already hands Reka successfully, so a live WebM is
    transcoded to it; an already-MP4 live capture (e.g. Safari) only needs the streaming
    metadata normalized, so it is copy-remuxed without a lossy re-encode. When ffmpeg is not
    installed the evidence passes through unchanged: local development without media tools
    degrades to provider-side failure rather than blocking the pipeline here.
    """

    def remux(self, evidence: CaptureEvidence) -> CaptureEvidence:
        if evidence.media_type not in {"video/webm", "video/mp4"}:
            return evidence
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
