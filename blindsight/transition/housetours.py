"""Reproducible HouseTours source-video planning and outcome retention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

from .corpus import RoomInterval

ProbeStatus = Literal["pending", "successful", "dead", "private", "blocked", "failed"]


@dataclass(frozen=True)
class HouseToursSourceRequest:
    source_video_id: str
    clip_ids: tuple[str, ...]


@dataclass(frozen=True)
class HouseToursProbeRecord:
    source_video_id: str
    clip_ids: tuple[str, ...]
    status: ProbeStatus
    reason: str | None


def build_source_requests(intervals: list[RoomInterval]) -> list[HouseToursSourceRequest]:
    """Group every requested HouseTours cut by its public source video without dropping any clip."""

    grouped: dict[str, set[str]] = {}
    for interval in intervals:
        if interval.corpus != "housetours":
            continue
        grouped.setdefault(interval.source_video_id, set()).add(interval.clip_id)
    return [
        HouseToursSourceRequest(source_video_id, tuple(sorted(clip_ids)))
        for source_video_id, clip_ids in sorted(grouped.items())
    ]


def record_probe_outcomes(
    requests: list[HouseToursSourceRequest],
    outcomes: Mapping[str, tuple[ProbeStatus, str | None]],
) -> list[HouseToursProbeRecord]:
    """Make every selected source's outcome visible, including unavailable public-video IDs."""

    return [
        HouseToursProbeRecord(
            source_video_id=request.source_video_id,
            clip_ids=request.clip_ids,
            status=outcomes.get(request.source_video_id, ("pending", None))[0],
            reason=outcomes.get(request.source_video_id, ("pending", None))[1],
        )
        for request in requests
    ]


def select_source_requests(
    requests: list[HouseToursSourceRequest], *, target_source_count: int, request_surplus: int
) -> list[HouseToursSourceRequest]:
    """Select a deterministic candidate pool that is larger than the requested successful target."""

    if target_source_count <= 0 or request_surplus <= 0:
        raise ValueError("HouseTours target and request surplus must both be positive.")
    candidate_count = target_source_count + request_surplus
    if candidate_count > len(requests):
        raise ValueError(f"Requested {candidate_count} HouseTours sources, only {len(requests)} are available.")
    return sorted(requests, key=lambda request: (-len(request.clip_ids), request.source_video_id))[
        :candidate_count
    ]


def _probe_status(returncode: int, output: str) -> ProbeStatus:
    if returncode == 0:
        return "successful"
    normalized = output.lower()
    if "private" in normalized:
        return "private"
    if "403" in normalized or "blocked" in normalized or "geo" in normalized:
        return "blocked"
    if "unavailable" in normalized or "not available" in normalized or "removed" in normalized:
        return "dead"
    return "failed"


def probe_source_requests(
    requests: list[HouseToursSourceRequest], runner: Callable[[str], tuple[int, str]]
) -> list[HouseToursProbeRecord]:
    """Run the public-source probe and retain every result, including tool failures."""

    records: list[HouseToursProbeRecord] = []
    for request in requests:
        returncode, output = runner(request.source_video_id)
        status = _probe_status(returncode, output)
        records.append(
            HouseToursProbeRecord(
                source_video_id=request.source_video_id,
                clip_ids=request.clip_ids,
                status=status,
                reason=None if status == "successful" else output[-2_000:],
            )
        )
    return records
