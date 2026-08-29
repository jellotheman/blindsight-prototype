"""The preloaded demonstration excerpt catalog.

Backed by a manifest checked into the repository rather than a Modal volume: the library is
small and static, so it ships inside the container image like any other bundled asset. Excerpt
identifiers are contract identifiers (see `identifiers.py`), never the underlying poster
filename or a volume path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .identifiers import is_valid_identifier


@dataclass(frozen=True)
class ExcerptEntry:
    excerpt_id: str
    label: str
    duration_seconds: float
    poster_path: Path
    clip_path: Path


class ExcerptCatalog:
    """Read-only view over a manifest of preloaded excerpts and their poster images."""

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = manifest_path
        self._entries: dict[str, ExcerptEntry] = {}
        self._load()

    def _load(self) -> None:
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        base_dir = self._manifest_path.parent
        entries: dict[str, ExcerptEntry] = {}
        for item in manifest["items"]:
            excerpt_id = item["excerpt_id"]
            if not is_valid_identifier(excerpt_id):
                raise ValueError(f"excerpt id {excerpt_id!r} is not a valid contract identifier")
            entries[excerpt_id] = ExcerptEntry(
                excerpt_id=excerpt_id,
                label=item["label"],
                duration_seconds=item["duration_seconds"],
                poster_path=base_dir / item["poster_file"],
                clip_path=base_dir / item["clip_file"],
            )
        self._entries = entries

    def list_excerpts(self) -> list[dict[str, object]]:
        return [
            {
                "excerpt_id": entry.excerpt_id,
                "label": entry.label,
                "duration_seconds": entry.duration_seconds,
                "poster_url": f"/v1/excerpts/{entry.excerpt_id}/poster",
            }
            for entry in self._entries.values()
        ]

    def poster_bytes(self, excerpt_id: str) -> bytes | None:
        entry = self._entries.get(excerpt_id)
        if entry is None:
            return None
        return entry.poster_path.read_bytes()

    def evidence_bytes(self, excerpt_id: str) -> bytes | None:
        entry = self._entries.get(excerpt_id)
        if entry is None:
            return None
        return entry.clip_path.read_bytes()
