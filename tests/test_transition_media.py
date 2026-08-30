"""Media framing against real ffmpeg fixtures, skipped entirely when ffmpeg is absent."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from blindsight.transition.media import probe_decodable, splice_to_decodable_span

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required on PATH to build media fixtures at test time.",
)


def run_ffmpeg(arguments: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *arguments],
        check=True,
        capture_output=True,
        timeout=120,
    )


def fragment_boundaries(data: bytes, marker: bytes, child: bytes) -> list[int]:
    offsets = []
    position = data.find(marker)
    while position != -1:
        if data[position + 8 : position + 12] == child:
            offsets.append(position)
        position = data.find(marker, position + 1)
    return offsets


def split_fragmented_mp4(data: bytes) -> list[tuple[bytes, str]]:
    # A fragmented MP4 is [ftyp][moov]([moof][mdat])...; moof boxes always open with mfhd.
    moof_offsets = fragment_boundaries(data, b"moof", b"mfhd")
    assert len(moof_offsets) >= 2, "the fixture must contain at least two media fragments"
    boundaries = [0, *moof_offsets[1:], len(data)]
    return [
        (data[start:end], "video/mp4") for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _vint_length(data: bytes, offset: int) -> int:
    first = data[offset]
    if first == 0:
        return 8
    length = 1
    mask = 0x80
    while not first & mask:
        length += 1
        mask >>= 1
    return length


def split_webm_clusters(data: bytes) -> list[tuple[bytes, str]]:
    # MediaRecorder timeslices begin at Matroska Cluster elements (0x1F43B675); the cluster
    # size vint is followed by the Timecode element (0xE7).
    cluster_offsets = []
    position = data.find(b"\x1f\x43\xb6\x75")
    while position != -1:
        if data[position + 4 + _vint_length(data, position + 4)] == 0xE7:
            cluster_offsets.append(position)
        position = data.find(b"\x1f\x43\xb6\x75", position + 1)
    assert len(cluster_offsets) >= 2, "the fixture must contain at least two clusters"
    boundaries = [0, *cluster_offsets[1:], len(data)]
    return [
        (data[start:end], "video/webm") for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


@pytest.fixture(scope="module")
def fragmented_mp4_chunks(tmp_path_factory) -> list[tuple[bytes, str]]:
    base = tmp_path_factory.mktemp("fmp4")
    target = base / "capture.mp4"
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=128x96:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "10",
            "-movflags",
            "+frag_keyframe+empty_moov",
            str(target),
        ]
    )
    return split_fragmented_mp4(target.read_bytes())


@pytest.fixture(scope="module")
def webm_cluster_chunks(tmp_path_factory) -> list[tuple[bytes, str]]:
    base = tmp_path_factory.mktemp("webm")
    target = base / "capture.webm"
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=128x96:rate=10",
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "8",
            "-cluster_time_limit",
            "500",
            str(target),
        ]
    )
    return split_webm_clusters(target.read_bytes())


def test_a_bare_fragment_without_its_init_segment_is_not_decodable(
    fragmented_mp4_chunks: list[tuple[bytes, str]],
) -> None:
    result = probe_decodable(*fragmented_mp4_chunks[1])
    assert result.decodable is False
    assert result.reason


def test_an_individual_fragment_covers_only_part_of_the_span(
    fragmented_mp4_chunks: list[tuple[bytes, str]],
) -> None:
    result = probe_decodable(*fragmented_mp4_chunks[0])
    assert not result.decodable or (result.frame_count or 0) < 20


def test_splicing_fragmented_mp4_chunks_yields_one_decodable_h264_span(
    fragmented_mp4_chunks: list[tuple[bytes, str]],
) -> None:
    result = splice_to_decodable_span(fragmented_mp4_chunks, "video/mp4")
    assert result.content is not None
    assert result.reason is None
    assert result.media_type == "video/mp4"
    assert result.frame_count is not None and result.frame_count >= 15
    assert result.duration_seconds is not None and 1.5 <= result.duration_seconds <= 3.0
    probe = probe_decodable(result.content, result.media_type)
    assert probe.decodable is True
    assert probe.codec == "h264"


def test_splicing_webm_timeslice_chunks_transcodes_to_a_decodable_h264_span(
    webm_cluster_chunks: list[tuple[bytes, str]],
) -> None:
    assert probe_decodable(*webm_cluster_chunks[1]).decodable is False
    result = splice_to_decodable_span(webm_cluster_chunks, "video/webm")
    assert result.content is not None
    assert result.reason is None
    assert result.media_type == "video/mp4"
    assert result.frame_count is not None and result.frame_count >= 15
    assert result.duration_seconds is not None and 1.5 <= result.duration_seconds <= 3.0
    probe = probe_decodable(result.content, result.media_type)
    assert probe.decodable is True
    assert probe.codec == "h264"


def test_probing_unusable_media_yields_a_structured_failure() -> None:
    garbage = probe_decodable(b"this is not video at all" * 8, "video/webm")
    assert garbage.decodable is False
    assert garbage.reason

    empty = probe_decodable(b"", "video/mp4")
    assert empty.decodable is False
    assert empty.reason


def test_splicing_unusable_media_yields_a_structured_failure_not_an_exception(
    fragmented_mp4_chunks: list[tuple[bytes, str]],
) -> None:
    none_supplied = splice_to_decodable_span([], "video/mp4")
    assert none_supplied.content is None
    assert none_supplied.reason

    garbage = splice_to_decodable_span([(b"still not video", "video/webm")], "video/webm")
    assert garbage.content is None
    assert garbage.reason

    header_only = (fragmented_mp4_chunks[0][0][:64], "video/mp4")
    truncated = splice_to_decodable_span([header_only], "video/mp4")
    assert truncated.content is None
    assert truncated.reason
