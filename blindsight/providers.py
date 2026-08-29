"""Provider seam for captured-view understanding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CaptureEvidence:
    content: bytes
    media_type: str


@dataclass(frozen=True)
class ProviderResult:
    raw_text: str
    card_body: dict[str, Any] | None
    error: str | None = None


class CaptureProvider(Protocol):
    def describe(self, evidence: CaptureEvidence) -> ProviderResult: ...


class DeterministicProvider:
    """A predictable provider double for the HTTP acceptance suite and walking deployment."""

    def __init__(self, *, card_body: dict[str, Any]) -> None:
        self._card_body = card_body

    def describe(self, evidence: CaptureEvidence) -> ProviderResult:
        return ProviderResult(raw_text=json.dumps(self._card_body), card_body=self._card_body)
