"""Unit tests for the server-side media-repair seam in `blindsight.remux`."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from blindsight.providers import CaptureEvidence
from blindsight.remux import FfmpegChunkRemuxer, PassthroughMediaRemuxer


def _ffprobe_duration(ffprobe: str, content: bytes) -> float | None:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as handle:
        handle.write(content)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            timeout=30,
            check=False,
        )
        text = result.stdout.decode().strip()
        return float(text) if text and text != "N/A" else None
    finally:
        path.unlink(missing_ok=True)


def _ffprobe_video_codec(ffprobe: str, content: bytes) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(content)
        path = Path(handle.name)
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
        text = result.stdout.decode().strip()
        return text if text else None
    finally:
        path.unlink(missing_ok=True)


def _require_ffmpeg_with_libx264() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to exercise FfmpegChunkRemuxer")
    encoders = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, timeout=30
    ).stdout
    if encoders.count(b"libx264") == 0:
        pytest.skip("libx264 is required to exercise FfmpegChunkRemuxer's transcode path")
    return ffmpeg


def _synthetic_webm(ffmpeg: str) -> bytes:
    return subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=5",
            "-c:v",
            "libvpx",
            "-b:v",
            "150k",
            "-f",
            "webm",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout


def _synthetic_mp4(ffmpeg: str) -> bytes:
    # The mp4 muxer's faststart remux needs a seekable output, so this writes to a temp file
    # rather than piping to stdout.
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = Path(tmp_dir) / "source.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=64x64:rate=5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        return target.read_bytes()


def test_webm_capture_is_transcoded_to_h264_mp4() -> None:
    ffmpeg = _require_ffmpeg_with_libx264()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe is required to verify the transcode")
    webm = _synthetic_webm(ffmpeg)

    remuxed = FfmpegChunkRemuxer().remux(CaptureEvidence(content=webm, media_type="video/webm"))

    assert remuxed.media_type == "video/mp4"
    assert _ffprobe_video_codec(ffprobe, remuxed.content) == "h264"


def test_mp4_capture_is_copy_remuxed_without_reencode() -> None:
    ffmpeg = _require_ffmpeg_with_libx264()
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe is required to verify the copy-remux")
    mp4 = _synthetic_mp4(ffmpeg)

    remuxed = FfmpegChunkRemuxer().remux(CaptureEvidence(content=mp4, media_type="video/mp4"))

    assert remuxed.media_type == "video/mp4"
    assert _ffprobe_video_codec(ffprobe, remuxed.content) == "h264"


def test_ffmpeg_failure_passes_evidence_through_unchanged() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required to exercise the failure path")
    evidence = CaptureEvidence(content=b"not a real video file", media_type="video/webm")

    remuxed = FfmpegChunkRemuxer().remux(evidence)

    assert remuxed is evidence


def test_missing_ffmpeg_passes_evidence_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    evidence = CaptureEvidence(content=b"irrelevant", media_type="video/webm")

    remuxed = FfmpegChunkRemuxer().remux(evidence)

    assert remuxed is evidence


def test_non_video_media_type_passes_through_without_invoking_ffmpeg() -> None:
    evidence = CaptureEvidence(content=b"{}", media_type="application/json")

    remuxed = FfmpegChunkRemuxer().remux(evidence)

    assert remuxed is evidence


def test_passthrough_remuxer_never_modifies_evidence() -> None:
    evidence = CaptureEvidence(content=b"anything", media_type="video/webm")

    assert PassthroughMediaRemuxer().remux(evidence) is evidence
