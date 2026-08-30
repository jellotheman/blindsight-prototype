"""Captured-view media validation before any provider spend."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from .providers import CaptureEvidence


class MediaValidator(Protocol):
    def is_decodable(self, evidence: CaptureEvidence) -> bool: ...


def probe_video_codec(path: Path) -> str | None:
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
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    codec = result.stdout.decode().strip()
    return codec if result.returncode == 0 and codec else None


class FfprobeMediaValidator:
    def is_decodable(self, evidence: CaptureEvidence) -> bool:
        suffix = ".mp4" if evidence.media_type == "video/mp4" else ".webm"
        if evidence.media_type == "image/jpeg":
            suffix = ".jpg"
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(evidence.content)
                path = Path(handle.name)
            return probe_video_codec(path) is not None
        except (OSError, subprocess.SubprocessError):
            return False
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
