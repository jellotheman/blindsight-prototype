"""Immutable retained evidence for production runs and later re-judgment."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .providers import CaptureEvidence, ProviderResult
from .scene_card import SceneCardBody


class RunClock:
    """All recorded durations for one run share this monotonic origin."""

    def __init__(self) -> None:
        self._origin = time.perf_counter()
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> float:
        elapsed = round((time.perf_counter() - self._origin) * 1000, 1)
        self._marks[name] = elapsed
        return elapsed

    def as_dict(self) -> dict[str, float]:
        return dict(self._marks)

    def remaining_seconds(self, total_seconds: float) -> float:
        elapsed = time.perf_counter() - self._origin
        return max(0.0, total_seconds - elapsed)


class EvidenceStore(Protocol):
    def retain_capture(
        self,
        capture_id: str,
        scene_session_id: str,
        source: dict[str, Any],
        evidence: CaptureEvidence,
    ) -> None: ...

    def finish(
        self,
        capture_id: str,
        result: ProviderResult | None,
        *,
        accepted_card: dict[str, Any] | None,
        failure: dict[str, Any] | None,
        timings: dict[str, float],
    ) -> None: ...


class NullEvidenceStore:
    def retain_capture(
        self,
        capture_id: str,
        scene_session_id: str,
        source: dict[str, Any],
        evidence: CaptureEvidence,
    ) -> None:
        return None

    def finish(
        self,
        capture_id: str,
        result: ProviderResult | None,
        *,
        accepted_card: dict[str, Any] | None,
        failure: dict[str, Any] | None,
        timings: dict[str, float],
    ) -> None:
        return None


class FileEvidenceStore:
    """One immutable directory per capture, suitable for a mounted Modal Volume."""

    _lock = threading.Lock()

    def __init__(self, root: Path, *, flush: Callable[[], None] | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._flush = flush or (lambda: None)

    def retain_capture(
        self,
        capture_id: str,
        scene_session_id: str,
        source: dict[str, Any],
        evidence: CaptureEvidence,
    ) -> None:
        run_dir = self.root / capture_id
        run_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".webm" if evidence.media_type == "video/webm" else ".mp4"
        capture_path = run_dir / f"capture{suffix}"
        if not capture_path.exists():
            capture_path.write_bytes(evidence.content)
        metadata = {
            "capture_id": capture_id,
            "scene_session_id": scene_session_id,
            "source": source,
            "media_type": evidence.media_type,
            "capture_file": capture_path.name,
            "capture_sha256": hashlib.sha256(evidence.content).hexdigest(),
            "capture_bytes": len(evidence.content),
        }
        self._write_once(run_dir / "capture.json", metadata)
        self._flush()

    def finish(
        self,
        capture_id: str,
        result: ProviderResult | None,
        *,
        accepted_card: dict[str, Any] | None,
        failure: dict[str, Any] | None,
        timings: dict[str, float],
    ) -> None:
        run_dir = self.root / capture_id
        metadata = json.loads((run_dir / "capture.json").read_text(encoding="utf-8"))
        attempts = [asdict(attempt) for attempt in (result.attempts if result else [])]
        if result is not None and not attempts and (
            result.raw_text or result.error or result.card_body is not None
        ):
            attempts = [
                {
                    "provider": result.provider,
                    "model": result.model,
                    "attempt": result.attempt,
                    "raw_text": result.raw_text,
                    "card_body": result.card_body,
                    "failure_kind": result.failure_kind,
                    "error": result.error,
                    "usage": result.usage,
                    "timings": result.timings,
                }
            ]
        selection = None
        if result is not None and result.card_body is not None:
            selection = {
                "provider": result.provider,
                "model": result.model,
                "attempt": result.attempt,
            }
        record = {
            **metadata,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
            "selection": selection,
            "timings": timings,
            "failure": failure,
            "prompt": result.prompt if result is not None else None,
            "scene_card_schema": SceneCardBody.model_json_schema(),
        }
        self._write_once(run_dir / "run.json", record)
        if accepted_card is not None:
            self._write_once(run_dir / "card.json", accepted_card)
        with self._lock:
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "capture_id": capture_id,
                    "scene_session_id": metadata["scene_session_id"],
                    "recorded_at": record["recorded_at"],
                    "succeeded": accepted_card is not None,
                    "provider": selection["provider"] if selection else None,
                }) + "\n")
        self._flush()

    @staticmethod
    def _write_once(path: Path, value: dict[str, Any]) -> None:
        if path.exists():
            raise FileExistsError(f"retained evidence is immutable: {path}")
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
