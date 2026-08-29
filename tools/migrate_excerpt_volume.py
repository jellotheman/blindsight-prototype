"""One-off migration: copy the real 74-excerpt demonstration library from the prototype
preparation-artifacts volume onto an explicitly owned public volume ``blindsight-excerpts``,
and write a public-contract manifest onto it.

Run once from the repo root:

    & .venv\\Scripts\\modal.exe run tools/migrate_excerpt_volume.py

The source is the prototype's ``blindsight-prep-artifacts`` volume (read-only provenance); the
destination is ``blindsight-excerpts`` (create_if_missing=True). The private index metadata is
read from the source volume itself, so the container needs no git access.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

prep_volume = modal.Volume.from_name("blindsight-prep-artifacts")
dest_volume = modal.Volume.from_name("blindsight-excerpts", create_if_missing=True)

app = modal.App("blindsight-excerpt-migration")


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/prep": prep_volume, "/excerpts": dest_volume},
    timeout=600,
)
def migrate() -> dict:
    src_dir = Path("/prep/stage0-excerpts")
    dst_dir = Path("/excerpts")
    dst_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing source index: {index_path}")
    private_manifest = json.loads(index_path.read_text(encoding="utf-8-sig"))

    items: list[dict] = []
    copied = 0
    for entry in private_manifest:
        clip_name = entry["excerpt"]
        poster_name = entry["poster"]
        src_clip = src_dir / clip_name
        src_poster = src_dir / poster_name
        dst_clip = dst_dir / clip_name
        dst_poster = dst_dir / poster_name

        if not src_clip.is_file():
            raise FileNotFoundError(f"missing source clip: {src_clip}")
        if not src_poster.is_file():
            raise FileNotFoundError(f"missing source poster: {src_poster}")

        shutil.copyfile(src_clip, dst_clip)
        shutil.copyfile(src_poster, dst_poster)

        dst_size = dst_clip.stat().st_size
        if dst_size != entry["bytes"]:
            raise ValueError(
                f"size mismatch for {clip_name}: {dst_size} != {entry['bytes']}"
            )

        excerpt_id = clip_name.removesuffix(".mp4").replace(".", "-")
        kind = entry.get("kind", "excerpt")
        label = f"{kind.replace('-', ' ').title()}"

        items.append(
            {
                "excerpt_id": excerpt_id,
                "label": label,
                "duration_seconds": entry["duration_s"],
                "poster_file": poster_name,
                "clip_file": clip_name,
            }
        )
        copied += 1

    manifest = {"items": items}
    (dst_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    dest_volume.commit()
    return {"copied": copied, "manifest_items": len(items)}


if __name__ == "__main__":
    with app.run():
        result = migrate.remote()
        print(json.dumps(result, indent=2))
