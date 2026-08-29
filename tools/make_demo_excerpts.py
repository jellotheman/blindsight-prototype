"""Generate the placeholder demonstration excerpt manifest, posters, and eight-second clips.

Real excerpts will be cut from licensed first-person footage (see REFERENCE.local.md for the
private prototype's fetch/cut pipeline); that dataset prep is out of scope for the walking
skeleton and needs network access to Hugging Face plus ffmpeg. Until that prep lands, these
synthetic media exercises the real listing, poster, decoding, and capture contracts with files on
disk rather than mocked bytes. Re-run this script to regenerate it; nothing here should be
hand-edited.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "excerpts"

DEMO_ITEMS = [
    {"excerpt_id": "via-001-entry-02", "label": "Elevator lobby", "duration_seconds": 8.0},
    {"excerpt_id": "via-014-exit-01", "label": "Shared kitchen", "duration_seconds": 8.0},
    {"excerpt_id": "via-027-entry-05", "label": "Dorm room doorway", "duration_seconds": 8.0},
]

COLORS = [(58, 90, 128), (110, 74, 90), (74, 110, 90)]


def _make_poster(path: Path, label: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (320, 180), color=color)
    draw = ImageDraw.Draw(image)
    draw.text((12, 80), label, fill=(240, 240, 240))
    image.save(path, format="JPEG", quality=80)


def _make_clip(poster_path: Path, clip_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(poster_path),
            "-t",
            "8",
            "-r",
            "2",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-movflags",
            "+faststart",
            str(clip_path),
        ],
        check=True,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for item, color in zip(DEMO_ITEMS, COLORS):
        poster_file = f"{item['excerpt_id']}.jpg"
        clip_file = f"{item['excerpt_id']}.mp4"
        poster_path = OUT_DIR / poster_file
        _make_poster(poster_path, item["label"], color)
        _make_clip(poster_path, OUT_DIR / clip_file)
        manifest_items.append({**item, "poster_file": poster_file, "clip_file": clip_file})

    manifest = {"items": manifest_items}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest_items)} demo excerpts to {OUT_DIR}")


if __name__ == "__main__":
    main()
