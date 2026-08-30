from pathlib import Path

from blindsight.transition.acquisition import EGO4D_CLI_VERSION, ego4d_command


def test_ego4d_preview_uses_the_official_filtered_540p_command_without_downloading_media() -> None:
    command = ego4d_command(
        output_directory=Path("/data/ego4d-preview"),
        uid_file=Path("/data/plans/ego4d-uids.txt"),
        data_version="v2_1",
        aws_profile_name="ego4d-modal",
        approve_download=False,
    )

    assert EGO4D_CLI_VERSION == "1.7.3"
    assert command == [
        "ego4d",
        "--output_directory",
        "/data/ego4d-preview",
        "--version",
        "v2_1",
        "--datasets",
        "video_540ss",
        "--video_uid_file",
        "/data/plans/ego4d-uids.txt",
        "--aws_profile_name",
        "ego4d-modal",
    ]
    assert "-y" not in command
    assert "--skip-s3-checks" not in command
