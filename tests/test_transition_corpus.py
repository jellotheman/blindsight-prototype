import pytest

from blindsight.transition.corpus import (
    IGNORE_LABEL,
    ProxyBoundary,
    RoomInterval,
    build_boundary_table,
    build_frozen_manifest,
    build_corpus_report,
    write_frozen_corpus_artifacts,
    label_time_steps,
    load_egoenv_roompred_directory,
    load_roompred_csv,
    parse_housetours_clip_id,
)


def test_build_boundary_table_normalizes_both_corpora_and_classifies_families() -> None:
    intervals = [
        RoomInterval("ego4d", "video-1", "clip-1", 8, 12, "kitchen", "2"),
        RoomInterval("ego4d", "video-1", "clip-1", 12, 16, "lawn/yard/garden", "1"),
        RoomInterval("ego4d", "video-1", "clip-1", 16, 20, "balcony", "1"),
        RoomInterval("ego4d", "video-1", "clip-1", 4, 8, "front_door/entrance", "1"),
        RoomInterval("ego4d", "video-1", "clip-1", 0, 4, "kitchen", "1"),
        RoomInterval("housetours", "yt-1", "yt-1:0-40", 0, 4, "bedroom", "1"),
        RoomInterval("housetours", "yt-1", "yt-1:0-40", 4, 8, "bedroom", "2"),
        RoomInterval("housetours", "yt-1", "yt-1:0-40", 8, 12, "kitchen", "2"),
        RoomInterval("housetours", "yt-1", "yt-1:0-40", 12, 12, "kitchen", "3"),
    ]

    table, zero_length_rows = build_boundary_table(intervals)

    assert zero_length_rows == 1
    assert [(row.corpus, row.boundary_time, row.family) for row in table] == [
        ("ego4d", 4.0, "threshold-cross"),
        ("ego4d", 8.0, "threshold-cross"),
        ("ego4d", 12.0, "indoor-to-outdoor"),
        ("ego4d", 16.0, "outdoor-to-outdoor"),
        ("housetours", 4.0, "indoor-to-indoor"),
        ("housetours", 8.0, "indoor-to-indoor"),
    ]


def test_label_time_steps_marks_positive_guard_band_and_every_other_scored_step_negative() -> None:
    boundary = ProxyBoundary(
        "ego4d", "video-1", "clip-1", 10, "kitchen", "1", "bedroom", "1", "indoor-to-indoor"
    )
    interval = RoomInterval("ego4d", "video-1", "clip-1", 0, 20, "kitchen", "1")

    labels = label_time_steps(
        [8.5, 9.0, 9.9, 10.0, 13.9, 14.0, 14.9, 15.0, 19.0],
        [interval],
        [boundary],
        guard_band_seconds=1.0,
    )

    assert labels == [0, IGNORE_LABEL, IGNORE_LABEL, 1, 1, IGNORE_LABEL, IGNORE_LABEL, 0, 0]


def test_label_time_steps_never_labels_an_unannotated_gap_positive() -> None:
    boundary = ProxyBoundary(
        "ego4d", "video-1", "clip-1", 10, "kitchen", "1", "bedroom", "1", "indoor-to-indoor"
    )
    interval = RoomInterval("ego4d", "video-1", "clip-1", 0, 11, "kitchen", "1")

    labels = label_time_steps([10.0, 11.0, 13.0], [interval], [boundary], guard_band_seconds=1.0)

    assert labels == [1, IGNORE_LABEL, IGNORE_LABEL]


def test_loader_preserves_each_corpus_identifier_and_housetours_frame_range(tmp_path) -> None:
    ego4d = tmp_path / "ego4d_roompred_train.csv"
    ego4d.write_text(
        "video_uid,clip_uid,start_time,end_time,label,instance\n"
        "video-1,clip-1,0,4,kitchen,0\n",
        encoding="utf-8",
    )
    housetours = tmp_path / "housetours_roompred_train.csv"
    housetours.write_text(
        "clip_uid,start_time,end_time,label\nabc_DEF-12_1_139,0,4,corridor/hallway\n",
        encoding="utf-8",
    )

    ego_intervals = load_roompred_csv(ego4d, corpus="ego4d")
    house_intervals = load_roompred_csv(housetours, corpus="housetours")
    source = parse_housetours_clip_id(house_intervals[0].clip_id)

    assert ego_intervals[0].source_video_id == "video-1"
    assert (house_intervals[0].source_video_id, house_intervals[0].room_instance) == (
        "abc_DEF-12",
        "",
    )
    assert (source.source_start_frame, source.source_end_frame) == (1, 139)


def test_annotation_loader_reads_both_roompred_corpora_and_rejects_an_unmapped_label(tmp_path) -> None:
    roompred = tmp_path / "annotations" / "roompred"
    roompred.mkdir(parents=True)
    (roompred / "ego4d_roompred_train.csv").write_text(
        "video_uid,clip_uid,start_time,end_time,label,instance\nev,ec,0,4,kitchen,0\n",
        encoding="utf-8",
    )
    (roompred / "housetours_roompred_train.csv").write_text(
        "clip_uid,start_time,end_time,label\nhv_1_100,0,4,swimming_pool\n",
        encoding="utf-8",
    )

    intervals = load_egoenv_roompred_directory(tmp_path)

    assert [(interval.corpus, interval.room_label) for interval in intervals] == [
        ("ego4d", "kitchen"),
        ("housetours", "swimming_pool"),
    ]


def test_frozen_manifest_records_guard_band_complete_clip_splits_and_resolution_strategy() -> None:
    intervals = [
        RoomInterval("ego4d", "ev-1", "ec-1", 0, 4, "kitchen", "1"),
        RoomInterval("ego4d", "ev-1", "ec-1", 4, 8, "bedroom", "1"),
        RoomInterval("ego4d", "ev-2", "ec-2", 0, 4, "kitchen", "1"),
        RoomInterval("ego4d", "ev-2", "ec-2", 4, 8, "bedroom", "1"),
        RoomInterval("ego4d", "ev-3", "ec-3", 0, 4, "kitchen", "1"),
        RoomInterval("ego4d", "ev-3", "ec-3", 4, 8, "bedroom", "1"),
        RoomInterval("housetours", "hv-1", "hv-1_1_100", 0, 4, "kitchen", ""),
        RoomInterval("housetours", "hv-1", "hv-1_1_100", 4, 8, "bedroom", ""),
        RoomInterval("housetours", "hv-2", "hv-2_1_100", 0, 4, "kitchen", ""),
        RoomInterval("housetours", "hv-2", "hv-2_1_100", 4, 8, "bedroom", ""),
        RoomInterval("housetours", "hv-3", "hv-3_1_100", 0, 4, "kitchen", ""),
        RoomInterval("housetours", "hv-3", "hv-3_1_100", 4, 8, "bedroom", ""),
    ]
    boundaries, _ = build_boundary_table(intervals)

    manifest = build_frozen_manifest(
        intervals,
        boundaries,
        random_seed=7,
        heldout_counts={"ego4d": 1, "housetours": 1},
        train_counts={"ego4d": 1, "housetours": 1},
        ego4d_data_version="v2_1",
        ego4d_cli_version="1.7.3",
        guard_band_seconds=1.0,
        resolution_by_clip={
            ("ego4d", "ec-1"): "clip-file",
            ("ego4d", "ec-2"): "parent-cut",
            ("ego4d", "ec-3"): "unresolved",
            ("housetours", "hv-1_1_100"): "source-video",
            ("housetours", "hv-2_1_100"): "failed",
            ("housetours", "hv-3_1_100"): "source-video",
        },
    )

    by_clip = {(entry.corpus, entry.clip_id): entry for entry in manifest.clips}
    assert manifest.guard_band_seconds == 1.0
    assert manifest.positive_seconds == 4.0
    assert manifest.random_seed == 7
    assert by_clip[("ego4d", "ec-3")].split == "unresolved"
    assert by_clip[("ego4d", "ec-2")].extraction_strategy == "download-540p-parent-and-cut"
    assert by_clip[("housetours", "hv-2_1_100")].split == "unavailable"
    assert {entry.split for entry in manifest.clips if entry.corpus == "ego4d"} >= {
        "train",
        "heldout",
        "unresolved",
    }
    assert {entry.split for entry in manifest.clips if entry.corpus == "housetours"} >= {
        "train",
        "heldout",
        "unavailable",
    }


def test_corpus_report_requires_an_explanation_when_counts_differ_from_the_specification() -> None:
    interval = RoomInterval("ego4d", "video-1", "clip-1", 0, 4, "kitchen", "1")
    manifest = build_frozen_manifest(
        [interval],
        [],
        random_seed=7,
        heldout_counts={"ego4d": 0, "housetours": 0},
        train_counts={"ego4d": 1, "housetours": 0},
        ego4d_data_version="v2_1",
        ego4d_cli_version="1.7.3",
        guard_band_seconds=1.0,
        resolution_by_clip={("ego4d", "clip-1"): "clip-file"},
    )

    try:
        build_corpus_report(
            [interval],
            [],
            zero_length_rows=0,
            manifest=manifest,
            specification_counts={"ego4d": {"room_intervals": 2}},
        )
    except ValueError as error:
        assert "explanation" in str(error)
    else:
        raise AssertionError("A mismatched specification count must be explained.")

    report = build_corpus_report(
        [interval],
        [],
        zero_length_rows=0,
        manifest=manifest,
        specification_counts={"ego4d": {"room_intervals": 2}},
        difference_explanations={
            ("ego4d", "room_intervals"): "The unit fixture intentionally has one row."
        },
    )

    comparison = report["specification_comparison"]["ego4d"]["room_intervals"]
    assert comparison == {"expected": 2, "observed": 1, "difference": -1, "explanation": "The unit fixture intentionally has one row."}


def test_frozen_artifacts_are_written_once_with_the_manifest_and_report_together(tmp_path) -> None:
    interval = RoomInterval("ego4d", "video-1", "clip-1", 0, 4, "kitchen", "1")
    destination = tmp_path / "frozen"

    manifest, report = write_frozen_corpus_artifacts(
        destination,
        [interval],
        [],
        random_seed=7,
        heldout_counts={"ego4d": 0, "housetours": 0},
        train_counts={"ego4d": 1, "housetours": 0},
        ego4d_data_version="v2_1",
        ego4d_cli_version="1.7.3",
        guard_band_seconds=1.0,
        resolution_by_clip={("ego4d", "clip-1"): "clip-file"},
        specification_counts={"ego4d": {"room_intervals": 1}},
    )

    assert manifest.guard_band_seconds == 1.0
    assert report["corpora"]["ego4d"]["room_intervals"] == 1
    assert (destination / "manifest.json").is_file()
    assert (destination / "corpus-report.json").is_file()
    with pytest.raises(FileExistsError):
        write_frozen_corpus_artifacts(
            destination,
            [interval],
            [],
            random_seed=7,
            heldout_counts={"ego4d": 0, "housetours": 0},
            train_counts={"ego4d": 1, "housetours": 0},
            ego4d_data_version="v2_1",
            ego4d_cli_version="1.7.3",
            guard_band_seconds=2.0,
            resolution_by_clip={("ego4d", "clip-1"): "clip-file"},
            specification_counts={"ego4d": {"room_intervals": 1}},
        )
