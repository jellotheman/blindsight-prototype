"""Pinned, reviewable command construction for transition-corpus acquisition."""

from __future__ import annotations

from pathlib import Path

EGO4D_CLI_VERSION = "1.7.3"
EGO4D_DATASET = "video_540ss"
EGO4D_DIRECT_CLIP_DATASET = "clips"


def ego4d_command(
    *,
    output_directory: Path,
    uid_file: Path,
    data_version: str,
    aws_profile_name: str,
    approve_download: bool,
    dataset: str = EGO4D_DATASET,
) -> list[str]:
    """Build the official selective-download command.

    The normal preview deliberately has no automatic confirmation and retains S3 checks, because
    that is the CLI path that prints an exact estimated byte count before media transfer begins.
    """

    if not data_version or not aws_profile_name or not dataset:
        raise ValueError("Ego4D data version, dataset, and AWS profile name are required.")
    command = [
        "ego4d",
        "--output_directory",
        output_directory.as_posix(),
        "--version",
        data_version,
        "--datasets",
        dataset,
        "--video_uid_file",
        uid_file.as_posix(),
        "--aws_profile_name",
        aws_profile_name,
    ]
    if approve_download:
        command.append("-y")
    return command
