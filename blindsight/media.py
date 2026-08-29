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
