"""Distributed-store seam for capture jobs, sessions, and (later) conversation state."""

from __future__ import annotations

import copy
import threading
from typing import Any, Protocol


class CaptureStore(Protocol):
    def create_capture(self, resource: dict[str, Any], session: dict[str, Any]) -> None: ...

    def get_capture(self, capture_id: str) -> dict[str, Any] | None: ...

    def put_capture(self, capture_id: str, resource: dict[str, Any]) -> None: ...

    def put_chunk(
        self, capture_id: str, index: int, content: bytes, max_total_bytes: int
    ) -> str: ...

    def get_chunks(self, capture_id: str, count: int) -> dict[int, bytes]: ...


class MemoryCaptureStore:
    """Thread-safe deterministic adapter; share one instance across app instances in tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._captures: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._chunks: dict[tuple[str, int], bytes] = {}

    def create_capture(self, resource: dict[str, Any], session: dict[str, Any]) -> None:
        with self._lock:
            self._captures[resource["capture_id"]] = copy.deepcopy(resource)
            self._sessions[session["scene_session_id"]] = copy.deepcopy(session)

    def get_capture(self, capture_id: str) -> dict[str, Any] | None:
        with self._lock:
            resource = self._captures.get(capture_id)
            return copy.deepcopy(resource) if resource is not None else None

    def put_capture(self, capture_id: str, resource: dict[str, Any]) -> None:
        with self._lock:
            self._captures[capture_id] = copy.deepcopy(resource)

    def put_chunk(
        self, capture_id: str, index: int, content: bytes, max_total_bytes: int
    ) -> str:
        key = (capture_id, index)
        with self._lock:
            existing = self._chunks.get(key)
            if existing is not None:
                return "idempotent" if existing == content else "conflict"
            total = sum(
                len(chunk)
                for (stored_capture_id, _), chunk in self._chunks.items()
                if stored_capture_id == capture_id
            )
            if total + len(content) > max_total_bytes:
                return "too_large"
            self._chunks[key] = bytes(content)
            return "stored"

    def get_chunks(self, capture_id: str, count: int) -> dict[int, bytes]:
        with self._lock:
            return {
                index: bytes(content)
                for (stored_capture_id, index), content in self._chunks.items()
                if stored_capture_id == capture_id and index < count
            }


class ModalCaptureStore:
    """Capture store backed by a named Modal Dict shared by every web container."""

    def __init__(self, dictionary: Any) -> None:
        self._dictionary = dictionary

    @staticmethod
    def _capture_key(capture_id: str) -> str:
        return f"capture:{capture_id}"

    @staticmethod
    def _session_key(scene_session_id: str) -> str:
        return f"session:{scene_session_id}"

    @staticmethod
    def _chunk_key(capture_id: str, index: int) -> str:
        return f"chunk:{capture_id}:{index}"

    def create_capture(self, resource: dict[str, Any], session: dict[str, Any]) -> None:
        self._dictionary.put(self._capture_key(resource["capture_id"]), resource)
        self._dictionary.put(self._session_key(session["scene_session_id"]), session)

    def get_capture(self, capture_id: str) -> dict[str, Any] | None:
        resource = self._dictionary.get(self._capture_key(capture_id))
        return copy.deepcopy(resource) if resource is not None else None

    def put_capture(self, capture_id: str, resource: dict[str, Any]) -> None:
        self._dictionary.put(self._capture_key(capture_id), resource)

    def put_chunk(
        self, capture_id: str, index: int, content: bytes, max_total_bytes: int
    ) -> str:
        key = self._chunk_key(capture_id, index)
        stored = self._dictionary.put(key, content, skip_if_exists=True)
        if not stored:
            existing = self._dictionary.get(key)
            return "idempotent" if existing == content else "conflict"
        prefix = f"chunk:{capture_id}:"
        total = sum(
            len(value)
            for stored_key, value in self._dictionary.items()
            if isinstance(stored_key, str) and stored_key.startswith(prefix)
        )
        if total > max_total_bytes:
            self._dictionary.pop(key, None)
            return "too_large"
        return "stored"

    def get_chunks(self, capture_id: str, count: int) -> dict[int, bytes]:
        chunks: dict[int, bytes] = {}
        for index in range(count):
            content = self._dictionary.get(self._chunk_key(capture_id, index))
            if content is not None:
                chunks[index] = content
        return chunks
