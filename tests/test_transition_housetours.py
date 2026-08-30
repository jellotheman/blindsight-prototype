from blindsight.transition.corpus import RoomInterval
from blindsight.transition.housetours import (
    build_source_requests,
    probe_source_requests,
    record_probe_outcomes,
    select_source_requests,
)


def test_housetours_source_failures_remain_in_the_record_and_do_not_drop_requested_clips() -> None:
    intervals = [
        RoomInterval("housetours", "video-ok", "video-ok_1_100", 0, 4, "kitchen", ""),
        RoomInterval("housetours", "video-blocked", "video-blocked_1_100", 0, 4, "bedroom", ""),
    ]

    requests = build_source_requests(intervals)
    records = record_probe_outcomes(
        requests,
        {"video-ok": ("successful", None), "video-blocked": ("blocked", "HTTP 403")},
    )

    assert [(record.source_video_id, record.status, record.clip_ids) for record in records] == [
        ("video-blocked", "blocked", ("video-blocked_1_100",)),
        ("video-ok", "successful", ("video-ok_1_100",)),
    ]
    assert records[0].reason == "HTTP 403"


def test_probe_classifies_real_tool_failures_and_requests_more_sources_than_the_target() -> None:
    requests = [
        *build_source_requests(
            [
                RoomInterval("housetours", "first", "first_1_100", 0, 4, "kitchen", ""),
                RoomInterval("housetours", "second", "second_1_100", 0, 4, "kitchen", ""),
                RoomInterval("housetours", "third", "third_1_100", 0, 4, "kitchen", ""),
            ]
        )
    ]

    selected = select_source_requests(requests, target_source_count=2, request_surplus=1)
    records = probe_source_requests(
        selected,
        lambda source_id: {
            "first": (0, "source id"),
            "second": (1, "ERROR: Private video"),
            "third": (1, "ERROR: Video unavailable"),
        }[source_id],
    )

    assert len(selected) == 3
    assert [(record.source_video_id, record.status) for record in records] == [
        ("first", "successful"),
        ("second", "private"),
        ("third", "dead"),
    ]
