"""Short-lived provider transport URLs; never part of the public client API."""

from __future__ import annotations

import copy
import secrets
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from .providers import CaptureEvidence


@dataclass(frozen=True)
class PublishedMedia:
    content: bytes
    media_type: str
    expires_at: float


class ProviderMediaUrls(Protocol):
    def publish(self, evidence: CaptureEvidence) -> str: ...

    def resolve(self, token: str) -> PublishedMedia | None: ...

    def revoke(self, url: str) -> None: ...


class MemoryMediaUrlStore:
    def __init__(self, base_url: str, *, ttl_seconds: int = 300) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/")
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, PublishedMedia] = {}

    def publish(self, evidence: CaptureEvidence) -> str:
        token = secrets.token_urlsafe(32)
        self._items[token] = PublishedMedia(
            content=bytes(evidence.content),
            media_type=evidence.media_type,
            expires_at=time.time() + self.ttl_seconds,
        )
        return f"{self.base_url}/_provider-media/{token}"

    def resolve(self, token: str) -> PublishedMedia | None:
        item = self._items.get(token)
        if item is None:
            return None
        if item.expires_at <= time.time():
            self._items.pop(token, None)
            return None
        return item

    def revoke(self, url: str) -> None:
        self._items.pop(url.rsplit("/", 1)[-1], None)


class ModalMediaUrlStore:
    """A Modal Dict-backed URL store shared by independently scaled ASGI containers."""

    def __init__(self, dictionary: Any, base_url: str, *, ttl_seconds: int = 300) -> None:
        self._dictionary = dictionary
        self._memory = MemoryMediaUrlStore(base_url, ttl_seconds=ttl_seconds)
        self.base_url = self._memory.base_url
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(token: str) -> str:
        return f"provider-media:{token}"

    def publish(self, evidence: CaptureEvidence) -> str:
        token = secrets.token_urlsafe(32)
        item = PublishedMedia(
            content=bytes(evidence.content),
            media_type=evidence.media_type,
            expires_at=time.time() + self.ttl_seconds,
        )
        self._dictionary.put(self._key(token), item)
        return f"{self.base_url}/_provider-media/{token}"

    def resolve(self, token: str) -> PublishedMedia | None:
        item = self._dictionary.get(self._key(token))
        if item is None:
            return None
        item = copy.deepcopy(item)
        if item.expires_at <= time.time():
            self._dictionary.pop(self._key(token), None)
            return None
        return item

    def revoke(self, url: str) -> None:
        self._dictionary.pop(self._key(url.rsplit("/", 1)[-1]), None)
