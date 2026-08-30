"""Modal acquisition checks for the Stage 3 transition corpus.

This is separate from ``modal_app.py``: it never imports the Stage 0/1 ASGI application and keeps
licensed source media on the private dataset Volume.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import csv
from dataclasses import asdict
from typing import Any

import modal

from blindsight.transition.acquisition import EGO4D_CLI_VERSION, EGO4D_DIRECT_CLIP_DATASET, ego4d_command

APP_NAME = "blindsight-transition-research"
DATA_VOLUME_NAME = "blindsight-transition-data"
EGO4D_SECRET_NAME = "blindsight-ego4d-aws"
DATA_MOUNT = Path("/data")
AWS_PROFILE_NAME = "ego4d-modal"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
ego4d_secret = modal.Secret.from_name(EGO4D_SECRET_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"ego4d=={EGO4D_CLI_VERSION}", "yt-dlp==2025.8.27")
    .add_local_python_source("blindsight")
)


def _write_aws_profile(directory: Path) -> tuple[Path, Path]:
    """Write secret values only to the function's ephemeral filesystem."""

    import os

    credentials = directory / "credentials"
    config = directory / "config"
    credentials.write_text(
        f"[{AWS_PROFILE_NAME}]\n"
        f"aws_access_key_id={os.environ['AWS_ACCESS_KEY_ID']}\n"
        f"aws_secret_access_key={os.environ['AWS_SECRET_ACCESS_KEY']}\n",
        encoding="utf-8",
    )
    config.write_text(
        f"[profile {AWS_PROFILE_NAME}]\n"
        f"region={os.environ.get('AWS_REGION', 'us-west-1')}\n",
        encoding="utf-8",
    )
    return credentials, config


def _safe_output_tail(output: str) -> str:
    """Keep enough CLI evidence for a run record without returning a potentially huge log."""

    return output[-12_000:]


def _reported_size(output: str) -> str | None:
    patterns = (
        r"(?:total|estimated|download)\s+(?:size|volume)[^\n]*?([0-9][0-9,\.]*\s*(?:KiB|MiB|GiB|TiB|KB|MB|GB|TB))",
        r"([0-9][0-9,\.]*\s*(?:KiB|MiB|GiB|TiB|KB|MB|GB|TB))[^\n]*(?:download|total|estimated)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _run_cli(command: list[str], *, approve_download: bool, subprocess_timeout: int = 900) -> tuple[int, str]:
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        credentials, config = _write_aws_profile(directory)
        environment = {
            **os.environ,
            "AWS_SHARED_CREDENTIALS_FILE": str(credentials),
            "AWS_CONFIG_FILE": str(config),
        }
        result = subprocess.run(
            command,
            input=None if approve_download else "n\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=subprocess_timeout,
            env=environment,
        )
    return result.returncode, result.stdout + result.stderr


@app.function(image=image, secrets=[ego4d_secret], timeout=120)
def secret_status() -> dict[str, object]:
    """Verify secret fields without exposing their values."""

    import os

    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
    return {"secret": EGO4D_SECRET_NAME, "fields_present": {key: bool(os.environ.get(key)) for key in required}}


@app.function(image=image, secrets=[ego4d_secret], timeout=180)
def list_ego4d_datasets() -> dict[str, object]:
    """Use the official CLI's catalogue operation before choosing a download source."""

    returncode, output = _run_cli(
        ["ego4d", "--list-datasets", "--aws_profile_name", AWS_PROFILE_NAME], approve_download=False
    )
    return {"returncode": returncode, "output_tail": _safe_output_tail(output)}


@app.function(
    image=image,
    secrets=[ego4d_secret],
    volumes={str(DATA_MOUNT): data_volume},
    timeout=900,
)
def preview_ego4d_download(
    video_or_clip_uids: list[str],
    *,
    data_version: str,
    estimated_retained_bytes: int,
    estimated_compute_usd: float,
) -> dict[str, object]:
    """Record the CLI confirmation estimate, then decline before any source-media transfer."""

    if not video_or_clip_uids:
        raise ValueError("The Ego4D preview needs at least one annotated identifier.")
    plans = DATA_MOUNT / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    uid_file = plans / "ego4d-uids.txt"
    uid_file.write_text("\n".join(sorted(set(video_or_clip_uids))) + "\n", encoding="utf-8")
    command = ego4d_command(
        output_directory=DATA_MOUNT / "ego4d-preview",
        uid_file=uid_file,
        data_version=data_version,
        aws_profile_name=AWS_PROFILE_NAME,
        approve_download=False,
    )
    returncode, output = _run_cli(command, approve_download=False)
    record = {
        "ego4d_cli_version": EGO4D_CLI_VERSION,
        "data_version": data_version,
        "dataset": "video_540ss",
        "identifier_count": len(set(video_or_clip_uids)),
        "estimated_retained_bytes": estimated_retained_bytes,
        "estimated_compute_usd": estimated_compute_usd,
        "reported_size": _reported_size(output),
        "returncode": returncode,
        "download_started": False,
        "output_tail": _safe_output_tail(output),
    }
    (plans / "ego4d-preview.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return record


@app.function(
    image=image,
    secrets=[ego4d_secret],
    volumes={str(DATA_MOUNT): data_volume},
    timeout=1_800,
)
def verify_one_ego4d_clip(
    clip_uid: str, *, data_version: str, dataset: str = EGO4D_DIRECT_CLIP_DATASET
) -> dict[str, object]:
    """Download exactly one requested source file end to end before a large plan is approved."""

    if not clip_uid.strip():
        raise ValueError("A single Ego4D clip UID is required.")
    verification_root = DATA_MOUNT / "ego4d-verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    uid_file = verification_root / "one-clip-uid.txt"
    uid_file.write_text(clip_uid.strip() + "\n", encoding="utf-8")
    command = ego4d_command(
        output_directory=verification_root,
        uid_file=uid_file,
        data_version=data_version,
        aws_profile_name=AWS_PROFILE_NAME,
        approve_download=True,
        dataset=dataset,
    )
    returncode, output = _run_cli(command, approve_download=True)
    media_files = [
        {"path": str(path.relative_to(verification_root)), "bytes": path.stat().st_size}
        for path in verification_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv"}
    ]
    record = {
        "ego4d_cli_version": EGO4D_CLI_VERSION,
        "data_version": data_version,
        "dataset": dataset,
        "clip_uid": clip_uid,
        "returncode": returncode,
        "completed": returncode == 0 and bool(media_files),
        "media_files": media_files,
        "output_tail": _safe_output_tail(output),
    }
    (verification_root / "verification.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return record


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=120)
def ego4d_metadata_summary() -> dict[str, object]:
    """Inspect the downloaded metadata shape without copying it or source media outside Modal."""

    path = DATA_MOUNT / "ego4d-verification" / "ego4d.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        sections: dict[str, dict[str, object]] = {}
        for name, section in value.items():
            if isinstance(section, list):
                sample = section[0] if section else None
                sections[name] = {
                    "type": "list",
                    "items": len(section),
                    "sample_keys": sorted(sample)[:30] if isinstance(sample, dict) else [],
                }
            else:
                sections[name] = {"type": type(section).__name__}
        return {
            "path": str(path),
            "top_level": "dict",
            "top_level_keys": sorted(value)[:20],
            "sections": sections,
        }
    if isinstance(value, list):
        sample = value[0] if value else None
        return {
            "path": str(path),
            "top_level": "list",
            "items": len(value),
            "sample_keys": sorted(sample)[:30] if isinstance(sample, dict) else [],
        }
    raise ValueError("Unexpected Ego4D metadata JSON shape.")


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=120)
def ego4d_dataset_manifest_summary(dataset: str = EGO4D_DIRECT_CLIP_DATASET) -> dict[str, object]:
    """Inspect a CLI manifest shape so identifier resolution follows the official file exactly."""

    path = DATA_MOUNT / "ego4d-verification" / "v2" / dataset / "manifest.csv"
    if not path.exists():
        root = DATA_MOUNT / "ego4d-verification"
        return {
            "path": str(path),
            "exists": False,
            "retained_files": [str(item.relative_to(root)) for item in root.rglob("*") if item.is_file()],
        }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "path": str(path),
        "top_level": "csv",
        "items": len(rows),
        "columns": list(rows[0]) if rows else [],
        "sample": rows[0] if rows else {},
    }


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            target = (destination / item.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe EgoEnv archive member {item.filename!r}.")
        handle.extractall(destination)


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=900)
def publish_egoenv_annotations() -> dict[str, object]:
    """Fetch the official annotation archive into the private data volume, without video media."""

    import hashlib
    from urllib.request import urlopen

    root = DATA_MOUNT / "egoenv"
    archive = root / "annotations.zip"
    extracted = root / "annotations"
    url = "https://dl.fbaipublicfiles.com/ego-env/data/annotations.zip"
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urlopen(url, timeout=120) as response, archive.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    if not extracted.exists():
        _safe_extract_zip(archive, extracted)
    record = {
        "source_url": url,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "extracted_path": str(extracted),
    }
    (root / "annotations-manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return record


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=300)
def resolve_egoenv_identifiers() -> dict[str, object]:
    """Compare every official EgoEnv identifier with the selected current Ego4D manifests."""

    from blindsight.transition.corpus import load_egoenv_roompred_directory

    annotations = DATA_MOUNT / "egoenv" / "annotations"
    clip_manifest = DATA_MOUNT / "ego4d-verification" / "v2" / "clips" / "manifest.csv"
    video_manifest = DATA_MOUNT / "ego4d-verification" / "v2" / "video_540ss" / "manifest.csv"
    with clip_manifest.open(newline="", encoding="utf-8") as handle:
        direct_clip_ids = {row["exported_clip_uid"] for row in csv.DictReader(handle)}
    with video_manifest.open(newline="", encoding="utf-8") as handle:
        parent_video_ids = {row["video_uid"] for row in csv.DictReader(handle)}
    intervals = load_egoenv_roompred_directory(annotations)
    ego4d_pairs = sorted({(item.source_video_id, item.clip_id) for item in intervals if item.corpus == "ego4d"})
    rows: list[dict[str, str]] = []
    for video_uid, clip_uid in ego4d_pairs:
        if clip_uid in direct_clip_ids:
            status, reason = "clip-file", "Found in the current clips manifest."
        elif video_uid in parent_video_ids:
            status, reason = "parent-cut", "No clip file; parent video is in video_540ss."
        else:
            status, reason = "unresolved", "Neither clip nor parent video is in the selected v2_1 manifests."
        rows.append(
            {"video_uid": video_uid, "clip_uid": clip_uid, "status": status, "reason": reason}
        )
    plans = DATA_MOUNT / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / "ego4d-resolution.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    counts = {status: sum(row["status"] == status for row in rows) for status in ("clip-file", "parent-cut", "unresolved")}
    first_direct = next((row["clip_uid"] for row in rows if row["status"] == "clip-file"), None)
    return {"counts": counts, "first_direct_clip_uid": first_direct, "resolution_path": str(plans / "ego4d-resolution.json")}


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=300)
def summarize_egoenv_annotations() -> dict[str, object]:
    """Return source counts from the official private annotation copy, without returning its rows."""

    from blindsight.transition.corpus import build_boundary_table, load_egoenv_roompred_directory

    intervals = load_egoenv_roompred_directory(DATA_MOUNT / "egoenv" / "annotations")
    boundaries, zero_length_rows = build_boundary_table(intervals)
    return {
        "room_intervals": {corpus: sum(item.corpus == corpus for item in intervals) for corpus in ("ego4d", "housetours")},
        "clips": {corpus: len({item.clip_id for item in intervals if item.corpus == corpus}) for corpus in ("ego4d", "housetours")},
        "source_videos": {corpus: len({item.source_video_id for item in intervals if item.corpus == corpus}) for corpus in ("ego4d", "housetours")},
        "proxy_transitions": {corpus: sum(item.corpus == corpus for item in boundaries) for corpus in ("ego4d", "housetours")},
        "zero_length_rows": zero_length_rows,
    }


@app.function(
    image=image,
    secrets=[ego4d_secret],
    volumes={str(DATA_MOUNT): data_volume},
    timeout=900,
)
def preview_full_ego4d_download(*, data_version: str = "v2_1") -> dict[str, object]:
    """Preview the complete resolved Ego4D download from the saved resolution plan.

    Reads every parent-cut ``video_540ss`` identifier already recorded by
    ``resolve_egoenv_identifiers`` and asks the official CLI for the exact total size, declining
    before any media transfers. This is the size the acceptance criteria require before a large
    download is approved.
    """

    plans = DATA_MOUNT / "plans"
    resolution_path = plans / "ego4d-resolution.json"
    rows = json.loads(resolution_path.read_text(encoding="utf-8"))
    video_uids = sorted({row["video_uid"] for row in rows if row["status"] == "parent-cut"})
    if not video_uids:
        raise ValueError("No parent-cut Ego4D video identifiers are recorded on the volume.")
    uid_file = plans / "ego4d-full-uids.txt"
    uid_file.write_text("\n".join(video_uids) + "\n", encoding="utf-8")
    command = ego4d_command(
        output_directory=DATA_MOUNT / "ego4d-full-preview",
        uid_file=uid_file,
        data_version=data_version,
        aws_profile_name=AWS_PROFILE_NAME,
        approve_download=False,
        dataset="video_540ss",
    )
    returncode, output = _run_cli(command, approve_download=False)
    record = {
        "ego4d_cli_version": EGO4D_CLI_VERSION,
        "data_version": data_version,
        "dataset": "video_540ss",
        "video_uid_count": len(video_uids),
        "reported_size": _reported_size(output),
        "returncode": returncode,
        "download_started": False,
        "output_tail": _safe_output_tail(output),
    }
    (plans / "ego4d-full-preview.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return record


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=300)
def rank_ego4d_video_boundary_density() -> dict[str, object]:
    """Rank every resolved Ego4D parent video by its proxy-boundary count, for subset selection.

    Reads the saved resolution plan and the official annotations to report, for each resolved
    ``video_540ss`` parent source, how many proxy transition boundaries and clips it contributes.
    This is read-only: it selects nothing and starts no transfer. A subset-selection step reads this
    ranking to prefer boundary-dense sources for a size-bounded download, while still keeping enough
    lower-density sources in the pool for an unbiased held-out and test draw.
    """

    from blindsight.transition.corpus import build_boundary_table, load_egoenv_roompred_directory

    plans = DATA_MOUNT / "plans"
    rows = json.loads((plans / "ego4d-resolution.json").read_text(encoding="utf-8"))
    resolved_clip_ids = {row["clip_uid"] for row in rows if row["status"] in {"clip-file", "parent-cut"}}
    intervals = load_egoenv_roompred_directory(DATA_MOUNT / "egoenv" / "annotations")
    ego_intervals = [
        item for item in intervals if item.corpus == "ego4d" and item.clip_id in resolved_clip_ids
    ]
    boundaries, _ = build_boundary_table(ego_intervals)
    boundary_counts: dict[str, int] = defaultdict(int)
    for boundary in boundaries:
        boundary_counts[boundary.source_video_id] += 1
    clip_ids_by_video: dict[str, set[str]] = defaultdict(set)
    for interval in ego_intervals:
        clip_ids_by_video[interval.source_video_id].add(interval.clip_id)
    ranking_rows = sorted(
        (
            (video_uid, boundary_counts.get(video_uid, 0), len(clip_ids))
            for video_uid, clip_ids in clip_ids_by_video.items()
        ),
        key=lambda row: (-row[1], row[0]),
    )
    ranking = [
        {"video_uid": video_uid, "boundary_count": boundary_count, "clip_count": clip_count}
        for video_uid, boundary_count, clip_count in ranking_rows
    ]
    (plans / "ego4d-video-boundary-density.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return {
        "video_count": len(ranking_rows),
        "clip_count": sum(clip_count for _, _, clip_count in ranking_rows),
        "boundary_count": sum(boundary_count for _, boundary_count, _ in ranking_rows),
        "ranking": ranking,
    }


@app.function(
    image=image,
    secrets=[ego4d_secret],
    volumes={str(DATA_MOUNT): data_volume},
    timeout=900,
)
def preview_ego4d_subset_download(video_uids: list[str], *, data_version: str = "v2_1") -> dict[str, object]:
    """Preview an explicit Ego4D video-subset download, declining the CLI's own prompt.

    Same pattern as ``preview_full_ego4d_download``, but for a caller-selected ``video_uids`` list
    rather than every resolved parent-cut identifier. Use this to confirm a candidate subset's real
    size, straight from the CLI's own confirmation-prompt estimate, before approving any transfer.
    """

    if not video_uids:
        raise ValueError("The Ego4D subset preview needs at least one video identifier.")
    plans = DATA_MOUNT / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    unique_uids = sorted(set(video_uids))
    uid_file = plans / "ego4d-subset-uids.txt"
    uid_file.write_text("\n".join(unique_uids) + "\n", encoding="utf-8")
    command = ego4d_command(
        output_directory=DATA_MOUNT / "ego4d-subset-preview",
        uid_file=uid_file,
        data_version=data_version,
        aws_profile_name=AWS_PROFILE_NAME,
        approve_download=False,
        dataset="video_540ss",
    )
    returncode, output = _run_cli(command, approve_download=False)
    record = {
        "ego4d_cli_version": EGO4D_CLI_VERSION,
        "data_version": data_version,
        "dataset": "video_540ss",
        "video_uid_count": len(unique_uids),
        "reported_size": _reported_size(output),
        "returncode": returncode,
        "download_started": False,
        "output_tail": _safe_output_tail(output),
    }
    (plans / "ego4d-subset-preview.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return record


@app.function(
    image=image,
    secrets=[ego4d_secret],
    volumes={str(DATA_MOUNT): data_volume},
    timeout=14_400,
)
def download_ego4d_subset(video_uids: list[str], *, data_version: str = "v2_1") -> dict[str, object]:
    """Download an explicit, already-previewed Ego4D video subset onto the shared data volume.

    Same pattern as ``preview_ego4d_subset_download``, but approves the CLI's own confirmation
    prompt and retains the transferred media under ``/data/ego4d-30gb/``. The caller must have
    already confirmed ``preview_ego4d_subset_download``'s reported size is in the approved band
    before calling this; this function does not re-check that band itself.
    """

    if not video_uids:
        raise ValueError("The Ego4D subset download needs at least one video identifier.")
    output_directory = DATA_MOUNT / "ego4d-30gb"
    plans = DATA_MOUNT / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    unique_uids = sorted(set(video_uids))
    uid_file = plans / "ego4d-subset-download-uids.txt"
    uid_file.write_text("\n".join(unique_uids) + "\n", encoding="utf-8")
    command = ego4d_command(
        output_directory=output_directory,
        uid_file=uid_file,
        data_version=data_version,
        aws_profile_name=AWS_PROFILE_NAME,
        approve_download=True,
        dataset="video_540ss",
    )
    returncode, output = _run_cli(command, approve_download=True, subprocess_timeout=14_100)
    retained_bytes = sum(path.stat().st_size for path in output_directory.rglob("*") if path.is_file())
    record = {
        "ego4d_cli_version": EGO4D_CLI_VERSION,
        "data_version": data_version,
        "dataset": "video_540ss",
        "video_uid_count": len(unique_uids),
        "returncode": returncode,
        "download_started": True,
        "retained_bytes": retained_bytes,
        "output_directory": str(output_directory),
        "output_tail": _safe_output_tail(output),
    }
    (plans / "ego4d-subset-download.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return record


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=1_800)
def probe_housetours_sources(target_source_count: int, request_surplus: int) -> dict[str, object]:
    """Probe more public HouseTours sources than the selected target and retain all outcomes."""

    import subprocess

    from blindsight.transition.corpus import load_egoenv_roompred_directory
    from blindsight.transition.housetours import (
        build_source_requests,
        probe_source_requests,
        select_source_requests,
    )

    intervals = load_egoenv_roompred_directory(DATA_MOUNT / "egoenv" / "annotations")
    requests = build_source_requests(intervals)
    selected = select_source_requests(
        requests, target_source_count=target_source_count, request_surplus=request_surplus
    )

    def probe(source_video_id: str) -> tuple[int, str]:
        result = subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--no-playlist",
                "--print",
                "%(id)s",
                f"https://www.youtube.com/watch?v={source_video_id}",
            ],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        return result.returncode, result.stdout + result.stderr

    records = probe_source_requests(selected, probe)
    version = subprocess.run(
        ["yt-dlp", "--version"], text=True, capture_output=True, timeout=30, check=True
    ).stdout.strip()
    plans = DATA_MOUNT / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    output = {
        "yt_dlp_version": version,
        "target_source_count": target_source_count,
        "requested_source_count": len(selected),
        "records": [asdict(record) for record in records],
    }
    (plans / "housetours-probes.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    return {
        "yt_dlp_version": version,
        "target_source_count": target_source_count,
        "requested_source_count": len(selected),
        "outcomes": {status: sum(record.status == status for record in records) for status in {record.status for record in records}},
        "record_path": str(plans / "housetours-probes.json"),
    }


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=60)
def housetours_probe_summary() -> dict[str, object]:
    """Read the saved public-source outcomes without repeating network probes."""

    value = json.loads((DATA_MOUNT / "plans" / "housetours-probes.json").read_text(encoding="utf-8"))
    return {
        "yt_dlp_version": value["yt_dlp_version"],
        "target_source_count": value["target_source_count"],
        "requested_source_count": value["requested_source_count"],
        "outcomes": {
            status: sum(row["status"] == status for row in value["records"])
            for status in sorted({row["status"] for row in value["records"]})
        },
        "failure_samples": [
            {"source_video_id": row["source_video_id"], "status": row["status"], "reason": row["reason"][-500:]}
            for row in value["records"]
            if row["status"] != "successful"
        ][:3],
    }


def _freeze_corpus_from_plans(
    config: dict[str, Any], *, ego4d_video_subset: set[str] | None
) -> dict[str, object]:
    """Shared implementation for the full-corpus freeze and a video-subset freeze.

    When ``ego4d_video_subset`` is given, Ego4D intervals whose parent source is outside it are
    filtered out before boundaries are computed, so an excluded clip is simply absent from the
    resulting manifest rather than carrying an invented resolution status. HouseTours rows are
    unaffected and keep the existing unavailable/unresolved treatment from ADR-0004.
    """

    from blindsight.transition.acquisition import EGO4D_CLI_VERSION
    from blindsight.transition.corpus import (
        CorpusName,
        ResolutionStatus,
        build_boundary_table,
        load_egoenv_roompred_directory,
        write_frozen_corpus_artifacts,
    )

    name = str(config["name"])
    if not name or Path(name).name != name:
        raise ValueError("The frozen corpus name must be one path component.")
    plans = DATA_MOUNT / "plans"
    ego_rows = json.loads((plans / "ego4d-resolution.json").read_text(encoding="utf-8"))
    house_probe = json.loads((plans / "housetours-probes.json").read_text(encoding="utf-8"))
    intervals = load_egoenv_roompred_directory(DATA_MOUNT / "egoenv" / "annotations")
    if ego4d_video_subset is not None:
        intervals = [
            interval
            for interval in intervals
            if interval.corpus != "ego4d" or interval.source_video_id in ego4d_video_subset
        ]
    boundaries, _ = build_boundary_table(intervals)
    resolution_by_clip: dict[tuple[CorpusName, str], ResolutionStatus] = {}
    reasons: dict[tuple[CorpusName, str], str] = {}
    for row in ego_rows:
        if ego4d_video_subset is not None and row["video_uid"] not in ego4d_video_subset:
            continue
        ego_key: tuple[CorpusName, str] = ("ego4d", row["clip_uid"])
        resolution_by_clip[ego_key] = row["status"]
        reasons[ego_key] = row["reason"]
    house_by_source = {row["source_video_id"]: row for row in house_probe["records"]}
    for interval in intervals:
        if interval.corpus != "housetours":
            continue
        house_key: tuple[CorpusName, str] = ("housetours", interval.clip_id)
        outcome = house_by_source.get(interval.source_video_id)
        if outcome is None:
            resolution_by_clip[house_key] = "unresolved"
            reasons[house_key] = "Source was not in the requested HouseTours probe pool."
        elif outcome["status"] == "successful":
            resolution_by_clip[house_key] = "source-video"
            reasons[house_key] = "Public source probe succeeded."
        else:
            resolution_by_clip[house_key] = "failed"
            reasons[house_key] = outcome["reason"] or f"HouseTours source probe was {outcome['status']}."
    explanations = {
        tuple(explanation_key.split(".", 1)): value
        for explanation_key, value in config["difference_explanations"].items()
    }
    destination = DATA_MOUNT / "frozen-corpora" / name
    manifest, report = write_frozen_corpus_artifacts(
        destination,
        intervals,
        boundaries,
        random_seed=int(config["random_seed"]),
        heldout_counts=config["heldout_counts"],
        train_counts=config["train_counts"],
        test_counts=config.get("test_counts"),
        ego4d_data_version=str(config["ego4d_data_version"]),
        ego4d_cli_version=EGO4D_CLI_VERSION,
        guard_band_seconds=float(config["guard_band_seconds"]),
        resolution_by_clip=resolution_by_clip,
        resolution_reasons=reasons,
        specification_counts=config["specification_counts"],
        difference_explanations=explanations,
    )
    data_volume.commit()
    return {
        "manifest_path": str(destination / "manifest.json"),
        "report_path": str(destination / "corpus-report.json"),
        "clip_count": len(manifest.clips),
        "unresolved_ego4d_count": len(manifest.unresolved_ego4d),
        "corpora": report["corpora"],
    }


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=900)
def freeze_transition_corpus(config_json: str) -> dict[str, object]:
    """Create the immutable cross-corpus manifest and report from saved acquisition records."""

    return _freeze_corpus_from_plans(json.loads(config_json), ego4d_video_subset=None)


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume}, timeout=900)
def freeze_ego4d_subset_corpus(config_json: str) -> dict[str, object]:
    """Freeze a manifest restricted to an explicit Ego4D parent-video subset.

    ``config_json`` takes the same shape as ``freeze_transition_corpus``, plus a required
    ``video_uids`` list naming the Ego4D ``video_540ss`` parent sources this manifest covers (for
    example, the sources actually pulled by ``download_ego4d_subset``). Ego4D clips whose parent is
    outside that list do not appear in the resulting manifest at all.
    """

    config = json.loads(config_json)
    video_uids = config["video_uids"]
    if not video_uids:
        raise ValueError("freeze_ego4d_subset_corpus requires a non-empty video_uids list.")
    return _freeze_corpus_from_plans(config, ego4d_video_subset=set(video_uids))


@app.local_entrypoint()
def main(
    action: str = "secret-status",
    value: str = "",
    data_version: str = "v2_1",
    target_source_count: int = 0,
    request_surplus: int = 0,
) -> None:
    if action == "secret-status":
        result = secret_status.remote()
    elif action == "list-datasets":
        result = list_ego4d_datasets.remote()
    elif action == "verify-one-clip":
        result = verify_one_ego4d_clip.remote(value, data_version=data_version)
    elif action == "verify-one-parent":
        result = verify_one_ego4d_clip.remote(value, data_version=data_version, dataset="video_540ss")
    elif action == "metadata-summary":
        result = ego4d_metadata_summary.remote()
    elif action == "dataset-manifest-summary":
        result = ego4d_dataset_manifest_summary.remote()
    elif action == "video-540ss-manifest-summary":
        result = ego4d_dataset_manifest_summary.remote("video_540ss")
    elif action == "publish-annotations":
        result = publish_egoenv_annotations.remote()
    elif action == "resolve-annotations":
        result = resolve_egoenv_identifiers.remote()
    elif action == "preview-full-download":
        result = preview_full_ego4d_download.remote(data_version=data_version)
    elif action == "summarize-annotations":
        result = summarize_egoenv_annotations.remote()
    elif action == "probe-housetours":
        result = probe_housetours_sources.remote(target_source_count, request_surplus)
    elif action == "housetours-probe-summary":
        result = housetours_probe_summary.remote()
    elif action == "freeze":
        result = freeze_transition_corpus.remote(Path(value).read_text(encoding="utf-8"))
    elif action == "video-boundary-density":
        result = rank_ego4d_video_boundary_density.remote()
    elif action == "preview-subset-download":
        uids = [line.strip() for line in Path(value).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = preview_ego4d_subset_download.remote(uids, data_version=data_version)
    elif action == "download-subset":
        uids = [line.strip() for line in Path(value).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = download_ego4d_subset.remote(uids, data_version=data_version)
    elif action == "freeze-subset":
        result = freeze_ego4d_subset_corpus.remote(Path(value).read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unknown action {action!r}.")
    print(json.dumps(result, indent=2, sort_keys=True))
