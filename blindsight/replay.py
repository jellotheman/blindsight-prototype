"""Re-run retained captured evidence without touching its accepted run or scene session."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .providers import CaptureEvidence, CaptureProvider
from .scene_card import SceneCardBody


class ReplayService:
    def __init__(self, provider: CaptureProvider) -> None:
        self.provider = provider

    def replay(self, run_dir: Path) -> Path:
        metadata = json.loads((run_dir / "capture.json").read_text(encoding="utf-8"))
        capture = (run_dir / metadata["capture_file"]).read_bytes()
        result = self.provider.describe(
            CaptureEvidence(content=capture, media_type=metadata["media_type"])
        )
        error: str | None
        try:
            body = SceneCardBody.model_validate(result.card_body).model_dump(mode="json")
        except ValidationError as exc:
            body = None
            error = result.error or str(exc)
        else:
            error = result.error

        provider = result.provider or getattr(self.provider, "name", type(self.provider).__name__)
        model = result.model or getattr(self.provider, "model", None)
        replay = {
            "replayed_at": datetime.now(timezone.utc).isoformat(),
            "capture_id": metadata["capture_id"],
            "provider": provider,
            "model": model,
            "valid": body is not None,
            "raw_text": result.raw_text,
            "card": body,
            "failure_kind": result.failure_kind,
            "error": error,
            "usage": result.usage,
            "attempts": [attempt.__dict__ for attempt in result.attempts],
        }
        replay_dir = run_dir / "replays"
        replay_dir.mkdir(exist_ok=True)
        path = replay_dir / f"replay-{provider}-{uuid.uuid4().hex[:12]}.json"
        path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
        return path

    def replay_many(self, root: Path, capture_ids: list[str]) -> list[Path]:
        return [self.replay(root / capture_id) for capture_id in capture_ids]
