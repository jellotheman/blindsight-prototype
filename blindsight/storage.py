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

    def delete_chunks(self, capture_id: str, count: int) -> None: ...

    def put_media(self, capture_id: str, content: bytes, media_type: str) -> None: ...

    def get_media(self, capture_id: str) -> tuple[bytes, str] | None: ...

    def get_session(self, scene_session_id: str) -> dict[str, Any] | None: ...

    def put_session(self, scene_session_id: str, session: dict[str, Any]) -> None: ...

    def delete_session(self, scene_session_id: str) -> None: ...

    def create_question(self, resource: dict[str, Any]) -> None: ...

    def get_question(self, question_id: str) -> dict[str, Any] | None: ...

    def put_question(self, question_id: str, resource: dict[str, Any]) -> None: ...


class MemoryCaptureStore:
    """Thread-safe deterministic adapter; share one instance across app instances in tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._captures: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._chunks: dict[tuple[str, int], bytes] = {}
        self._media: dict[str, tuple[bytes, str]] = {}
        self._questions: dict[str, dict[str, Any]] = {}

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

    def delete_chunks(self, capture_id: str, count: int) -> None:
        with self._lock:
            for index in range(count):
                self._chunks.pop((capture_id, index), None)

    def put_media(self, capture_id: str, content: bytes, media_type: str) -> None:
        with self._lock:
            self._media[capture_id] = (bytes(content), media_type)

    def get_media(self, capture_id: str) -> tuple[bytes, str] | None:
        with self._lock:
            item = self._media.get(capture_id)
            return (bytes(item[0]), item[1]) if item is not None else None

    def get_session(self, scene_session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(scene_session_id)
            return copy.deepcopy(session) if session is not None else None

    def put_session(self, scene_session_id: str, session: dict[str, Any]) -> None:
        with self._lock:
            self._sessions[scene_session_id] = copy.deepcopy(session)

    def delete_session(self, scene_session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(scene_session_id, None)
            if session is None:
                return
            capture_id = session.get("capture_id")
            if isinstance(capture_id, str):
                self._media.pop(capture_id, None)
            for question_id in session.get("question_ids", []):
                self._questions.pop(question_id, None)

    def create_question(self, resource: dict[str, Any]) -> None:
        with self._lock:
            self._questions[resource["question_id"]] = copy.deepcopy(resource)

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        with self._lock:
            resource = self._questions.get(question_id)
            return copy.deepcopy(resource) if resource is not None else None

    def put_question(self, question_id: str, resource: dict[str, Any]) -> None:
        with self._lock:
            self._questions[question_id] = copy.deepcopy(resource)


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

    @staticmethod
    def _chunk_sizes_key(capture_id: str) -> str:
        return f"chunksizes:{capture_id}"

    @staticmethod
    def _media_key(capture_id: str) -> str:
        return f"media:{capture_id}"

    @staticmethod
    def _question_key(question_id: str) -> str:
        return f"question:{question_id}"

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
        # Accumulated size is tracked in a small per-capture sidecar rather than derived by
        # scanning the Dict: a whole-Dict scan transfers every stored value -- including other
        # captures' chunk payloads -- on every chunk upload, which grows without bound and
        # eventually exceeds the request deadline.
        sizes_key = self._chunk_sizes_key(capture_id)
        stored_sizes = self._dictionary.get(sizes_key)
        sizes = dict(stored_sizes) if isinstance(stored_sizes, dict) else {}
        sizes[index] = len(content)
        if sum(sizes.values()) > max_total_bytes:
            self._dictionary.pop(key, None)
            return "too_large"
        self._dictionary.put(sizes_key, sizes)
        return "stored"

    def get_chunks(self, capture_id: str, count: int) -> dict[int, bytes]:
        chunks: dict[int, bytes] = {}
        for index in range(count):
            content = self._dictionary.get(self._chunk_key(capture_id, index))
            if content is not None:
                chunks[index] = content
        return chunks

    def delete_chunks(self, capture_id: str, count: int) -> None:
        for index in range(count):
            self._dictionary.pop(self._chunk_key(capture_id, index), None)
        self._dictionary.pop(self._chunk_sizes_key(capture_id), None)

    def put_media(self, capture_id: str, content: bytes, media_type: str) -> None:
        self._dictionary.put(self._media_key(capture_id), (bytes(content), media_type))

    def get_media(self, capture_id: str) -> tuple[bytes, str] | None:
        item = self._dictionary.get(self._media_key(capture_id))
        if item is None:
            return None
        content, media_type = item
        return (bytes(content), media_type)

    def get_session(self, scene_session_id: str) -> dict[str, Any] | None:
        session = self._dictionary.get(self._session_key(scene_session_id))
        return copy.deepcopy(session) if session is not None else None

    def put_session(self, scene_session_id: str, session: dict[str, Any]) -> None:
        self._dictionary.put(self._session_key(scene_session_id), session)

    def delete_session(self, scene_session_id: str) -> None:
        session = self._dictionary.get(self._session_key(scene_session_id))
        self._dictionary.pop(self._session_key(scene_session_id), None)
        if not isinstance(session, dict):
            return
        capture_id = session.get("capture_id")
        if isinstance(capture_id, str):
            self._dictionary.pop(self._media_key(capture_id), None)
        for question_id in session.get("question_ids", []):
            self._dictionary.pop(self._question_key(question_id), None)

    def create_question(self, resource: dict[str, Any]) -> None:
        self._dictionary.put(self._question_key(resource["question_id"]), resource)

    def get_question(self, question_id: str) -> dict[str, Any] | None:
        resource = self._dictionary.get(self._question_key(question_id))
        return copy.deepcopy(resource) if resource is not None else None

    def put_question(self, question_id: str, resource: dict[str, Any]) -> None:
        self._dictionary.put(self._question_key(question_id), resource)
