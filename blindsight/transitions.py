"""Stage 3 transition-session lifecycle, isolated from capture processing.

The public surface reports only session state and transition events.  Continuous-media ordering,
queued bytes, adapter work, and terminal failures stay behind this module's small adapter seam.
"""

from __future__ import annotations

import copy
import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from .concurrency import run_with_deadline
from .errors import ApiError, NotFound


MEDIA_TYPES = {"video/webm", "video/mp4", "video/quicktime"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class TransitionObservation:
    """One observation from causal inference; it intentionally carries no scene description."""

    observed_at: str


class TransitionAdapter(Protocol):
    """Continuous-inference boundary used by local tests and the Modal deployment."""

    def start(self, transition_session_id: str) -> None: ...

    def process(
        self, transition_session_id: str, index: int, content: bytes, media_type: str
    ) -> list[TransitionObservation]: ...

    def stop(self, transition_session_id: str) -> None: ...


class InMemoryTransitionAdapter:
    """Deterministic local adapter; observations can be selected by processed chunk index."""

    def __init__(
        self,
        *,
        observations_by_index: dict[int, list[TransitionObservation]] | None = None,
        fail_on_start: Exception | None = None,
        fail_on_process: Exception | None = None,
    ) -> None:
        self._observations_by_index = observations_by_index or {}
        self._fail_on_start = fail_on_start
        self._fail_on_process = fail_on_process
        self.processed_chunks: list[dict[str, Any]] = []
        self.stopped_session_ids: list[str] = []

    def start(self, transition_session_id: str) -> None:
        if self._fail_on_start is not None:
            raise self._fail_on_start

    def process(
        self, transition_session_id: str, index: int, content: bytes, media_type: str
    ) -> list[TransitionObservation]:
        if self._fail_on_process is not None:
            raise self._fail_on_process
        self.processed_chunks.append(
            {
                "transition_session_id": transition_session_id,
                "index": index,
                "content": bytes(content),
                "media_type": media_type,
            }
        )
        return list(self._observations_by_index.get(index, []))

    def stop(self, transition_session_id: str) -> None:
        self.stopped_session_ids.append(transition_session_id)


class ModalTransitionWorker(Protocol):
    """The narrowly scoped worker shape used by the production Modal adapter."""

    def start(self, transition_session_id: str) -> None: ...

    def process(
        self, transition_session_id: str, index: int, content: bytes, media_type: str
    ) -> list[TransitionObservation]: ...

    def stop(self, transition_session_id: str) -> None: ...


class ModalTransitionAdapter:
    """Production adapter delegating continuous inference to a pre-created Modal worker."""

    def __init__(self, worker: ModalTransitionWorker) -> None:
        self._worker = worker

    def start(self, transition_session_id: str) -> None:
        self._worker.start(transition_session_id)

    def process(
        self, transition_session_id: str, index: int, content: bytes, media_type: str
    ) -> list[TransitionObservation]:
        return self._worker.process(transition_session_id, index, content, media_type)

    def stop(self, transition_session_id: str) -> None:
        self._worker.stop(transition_session_id)


class ThreadRunner:
    def submit(self, operation: Callable[[], None]) -> None:
        threading.Thread(target=operation, daemon=True).start()


class TransitionSessionStore(Protocol):
    def create(self, resource: dict[str, Any]) -> None: ...

    def get(self, transition_session_id: str) -> dict[str, Any] | None: ...

    def put(self, transition_session_id: str, resource: dict[str, Any]) -> None: ...

    def delete(self, transition_session_id: str) -> None: ...

    def put_chunk(
        self,
        transition_session_id: str,
        index: int,
        content: bytes,
        media_type: str,
        max_queued_bytes: int,
    ) -> str: ...

    def get_chunk(self, transition_session_id: str, index: int) -> tuple[bytes, str] | None: ...

    def mark_processed(self, transition_session_id: str, index: int, content: bytes) -> None: ...

    def clear_chunks(self, transition_session_id: str, indices: list[int]) -> None: ...


class MemoryTransitionSessionStore:
    """Thread-safe local session store.  It retains hashes after processing for idempotency."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources: dict[str, dict[str, Any]] = {}
        self._chunks: dict[tuple[str, int], tuple[bytes, str]] = {}
        self._processed_hashes: dict[tuple[str, int], str] = {}

    def create(self, resource: dict[str, Any]) -> None:
        with self._lock:
            self._resources[resource["transition_session_id"]] = copy.deepcopy(resource)

    def get(self, transition_session_id: str) -> dict[str, Any] | None:
        with self._lock:
            resource = self._resources.get(transition_session_id)
            return copy.deepcopy(resource) if resource is not None else None

    def put(self, transition_session_id: str, resource: dict[str, Any]) -> None:
        with self._lock:
            self._resources[transition_session_id] = copy.deepcopy(resource)

    def delete(self, transition_session_id: str) -> None:
        with self._lock:
            self._resources.pop(transition_session_id, None)

    def put_chunk(
        self,
        transition_session_id: str,
        index: int,
        content: bytes,
        media_type: str,
        max_queued_bytes: int,
    ) -> str:
        key = (transition_session_id, index)
        digest = hashlib.sha256(content).hexdigest()
        with self._lock:
            existing = self._chunks.get(key)
            if existing is not None:
                return "idempotent" if existing[0] == content else "conflict"
            processed_digest = self._processed_hashes.get(key)
            if processed_digest is not None:
                return "idempotent" if processed_digest == digest else "conflict"
            queued_bytes = sum(
                len(stored_content)
                for (stored_session_id, _), (stored_content, _) in self._chunks.items()
                if stored_session_id == transition_session_id
            )
            if queued_bytes + len(content) > max_queued_bytes:
                return "too_large"
            self._chunks[key] = (bytes(content), media_type)
            return "stored"

    def get_chunk(self, transition_session_id: str, index: int) -> tuple[bytes, str] | None:
        key = (transition_session_id, index)
        with self._lock:
            chunk = self._chunks.get(key)
            return (bytes(chunk[0]), chunk[1]) if chunk is not None else None

    def mark_processed(self, transition_session_id: str, index: int, content: bytes) -> None:
        key = (transition_session_id, index)
        with self._lock:
            self._chunks.pop(key, None)
            self._processed_hashes[key] = hashlib.sha256(content).hexdigest()

    def clear_chunks(self, transition_session_id: str, indices: list[int]) -> None:
        with self._lock:
            for index in indices:
                key = (transition_session_id, index)
                self._chunks.pop(key, None)
                self._processed_hashes.pop(key, None)


class ModalTransitionSessionStore:
    """Transition-session state kept in the existing Modal Dict, never in request memory."""

    def __init__(self, dictionary: Any) -> None:
        self._dictionary = dictionary

    @staticmethod
    def _resource_key(transition_session_id: str) -> str:
        return f"transition-session:{transition_session_id}"

    @staticmethod
    def _chunk_key(transition_session_id: str, index: int) -> str:
        return f"transition-chunk:{transition_session_id}:{index}"

    @staticmethod
    def _processed_key(transition_session_id: str, index: int) -> str:
        return f"transition-processed:{transition_session_id}:{index}"

    @staticmethod
    def _sizes_key(transition_session_id: str) -> str:
        return f"transition-chunk-sizes:{transition_session_id}"

    def create(self, resource: dict[str, Any]) -> None:
        self._dictionary.put(self._resource_key(resource["transition_session_id"]), resource)

    def get(self, transition_session_id: str) -> dict[str, Any] | None:
        resource = self._dictionary.get(self._resource_key(transition_session_id))
        return copy.deepcopy(resource) if resource is not None else None

    def put(self, transition_session_id: str, resource: dict[str, Any]) -> None:
        self._dictionary.put(self._resource_key(transition_session_id), resource)

    def delete(self, transition_session_id: str) -> None:
        self._dictionary.pop(self._resource_key(transition_session_id), None)

    def put_chunk(
        self,
        transition_session_id: str,
        index: int,
        content: bytes,
        media_type: str,
        max_queued_bytes: int,
    ) -> str:
        key = self._chunk_key(transition_session_id, index)
        stored = self._dictionary.put(key, (bytes(content), media_type), skip_if_exists=True)
        if not stored:
            existing = self._dictionary.get(key)
            if existing is not None:
                return "idempotent" if existing[0] == content else "conflict"
            digest = self._dictionary.get(self._processed_key(transition_session_id, index))
            return "idempotent" if digest == hashlib.sha256(content).hexdigest() else "conflict"
        sizes_key = self._sizes_key(transition_session_id)
        stored_sizes = self._dictionary.get(sizes_key)
        sizes = dict(stored_sizes) if isinstance(stored_sizes, dict) else {}
        sizes[index] = len(content)
        if sum(sizes.values()) > max_queued_bytes:
            self._dictionary.pop(key, None)
            return "too_large"
        self._dictionary.put(sizes_key, sizes)
        return "stored"

    def get_chunk(self, transition_session_id: str, index: int) -> tuple[bytes, str] | None:
        item = self._dictionary.get(self._chunk_key(transition_session_id, index))
        return (bytes(item[0]), item[1]) if item is not None else None

    def mark_processed(self, transition_session_id: str, index: int, content: bytes) -> None:
        self._dictionary.pop(self._chunk_key(transition_session_id, index), None)
        self._dictionary.put(
            self._processed_key(transition_session_id, index), hashlib.sha256(content).hexdigest()
        )
        sizes_key = self._sizes_key(transition_session_id)
        stored_sizes = self._dictionary.get(sizes_key)
        sizes = dict(stored_sizes) if isinstance(stored_sizes, dict) else {}
        sizes.pop(index, None)
        self._dictionary.put(sizes_key, sizes)

    def clear_chunks(self, transition_session_id: str, indices: list[int]) -> None:
        for index in indices:
            self._dictionary.pop(self._chunk_key(transition_session_id, index), None)
            self._dictionary.pop(self._processed_key(transition_session_id, index), None)
        self._dictionary.pop(self._sizes_key(transition_session_id), None)


class TransitionService:
    """One product interface over session state, queueing, and continuous-inference adapters."""

    def __init__(
        self,
        *,
        store: TransitionSessionStore,
        adapter: TransitionAdapter,
        runner: ThreadRunner | None = None,
        max_chunk_bytes: int = 10 * 1024 * 1024,
        max_queued_bytes: int = 100 * 1024 * 1024,
        processing_deadline_seconds: float = 30.0,
    ) -> None:
        self._store = store
        self._adapter = adapter
        self._runner = runner or ThreadRunner()
        self._max_chunk_bytes = max_chunk_bytes
        self._max_queued_bytes = max_queued_bytes
        self._processing_deadline_seconds = processing_deadline_seconds
        self._drain_lock = threading.Lock()

    def create(self) -> dict[str, Any]:
        session_id = _identifier("trs")
        created_at = _now()
        resource: dict[str, Any] = {
            "transition_session_id": session_id,
            "status": "starting",
            "events": [],
            "failure": None,
            "created_at": created_at,
            "updated_at": created_at,
            "_next_index": 0,
            "_known_indices": [],
        }
        self._store.create(resource)
        self._runner.submit(lambda: self._start(session_id))
        return self._public(resource, cursor=None)

    def get(self, transition_session_id: str, cursor: str | None) -> dict[str, Any]:
        resource = self._store.get(transition_session_id)
        if resource is None:
            raise NotFound(f"No transition session with id {transition_session_id!r}.")
        if cursor is not None and cursor not in {
            event["transition_event_id"] for event in resource["events"]
        }:
            raise ApiError(400, "INVALID_REQUEST", "cursor must identify a retained transition event.")
        return self._public(resource, cursor=cursor)

    def put_chunk(
        self, transition_session_id: str, index: int, content: bytes, media_type: str
    ) -> dict[str, Any]:
        resource = self._store.get(transition_session_id)
        if resource is None:
            raise NotFound(f"No transition session with id {transition_session_id!r}.")
        if resource["status"] not in {"starting", "active"}:
            raise ApiError(
                409,
                "INVALID_STATE",
                "The transition session is no longer accepting chunks.",
                details={"status": resource["status"]},
            )
        if not 0 <= index <= 999_999 or not content:
            raise ApiError(400, "INVALID_REQUEST", "A non-empty chunk and valid index are required.")
        if media_type not in MEDIA_TYPES:
            raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported transition chunk media type.")
        if len(content) > self._max_chunk_bytes:
            raise ApiError(413, "TRANSITION_QUEUE_TOO_LARGE", "The chunk exceeds the byte limit.")
        outcome = self._store.put_chunk(
            transition_session_id, index, content, media_type, self._max_queued_bytes
        )
        if outcome == "conflict":
            raise ApiError(
                409,
                "CHUNK_CONFLICT",
                "This chunk index already contains different bytes.",
                details={"index": index},
            )
        if outcome == "too_large":
            raise ApiError(
                413,
                "TRANSITION_QUEUE_TOO_LARGE",
                "The queued transition media exceeds the byte limit.",
            )
        if outcome == "stored":
            resource["_known_indices"].append(index)
            resource["updated_at"] = _now()
            self._store.put(transition_session_id, resource)
            self._runner.submit(lambda: self._drain(transition_session_id))
        return {
            "transition_session_id": transition_session_id,
            "index": index,
            "bytes": len(content),
            "idempotent": outcome == "idempotent",
        }

    def delete(self, transition_session_id: str) -> None:
        resource = self._store.get(transition_session_id)
        if resource is None:
            raise NotFound(f"No transition session with id {transition_session_id!r}.")
        self._store.delete(transition_session_id)
        self._store.clear_chunks(transition_session_id, resource["_known_indices"])
        try:
            run_with_deadline(
                lambda: self._adapter.stop(transition_session_id), self._processing_deadline_seconds
            )
        except Exception:
            # Stop was accepted locally; a worker-specific cleanup failure cannot resurrect a
            # deleted public resource or make it accept more media.
            pass

    def _start(self, transition_session_id: str) -> None:
        resource = self._store.get(transition_session_id)
        if resource is None or resource["status"] != "starting":
            return
        outcome, value = run_with_deadline(
            lambda: self._adapter.start(transition_session_id), self._processing_deadline_seconds
        )
        resource = self._store.get(transition_session_id)
        if resource is None or resource["status"] != "starting":
            return
        if outcome != "result":
            self._fail(resource, "TRANSITION_ADAPTER_FAILED")
            return
        resource["status"] = "active"
        resource["updated_at"] = _now()
        self._store.put(transition_session_id, resource)
        self._drain(transition_session_id)

    def _drain(self, transition_session_id: str) -> None:
        with self._drain_lock:
            while True:
                resource = self._store.get(transition_session_id)
                if resource is None or resource["status"] != "active":
                    return
                index = resource["_next_index"]
                chunk = self._store.get_chunk(transition_session_id, index)
                if chunk is None:
                    return
                content, media_type = chunk
                outcome, value = run_with_deadline(
                    lambda: self._adapter.process(transition_session_id, index, content, media_type),
                    self._processing_deadline_seconds,
                )
                resource = self._store.get(transition_session_id)
                if resource is None or resource["status"] != "active":
                    return
                if outcome != "result" or not isinstance(value, list):
                    self._fail(resource, "TRANSITION_ADAPTER_FAILED")
                    return
                if not all(isinstance(observation, TransitionObservation) for observation in value):
                    self._fail(resource, "TRANSITION_ADAPTER_FAILED")
                    return
                self._store.mark_processed(transition_session_id, index, content)
                resource["_next_index"] = index + 1
                resource["events"].extend(
                    {
                        "transition_event_id": _identifier("tev"),
                        "observed_at": observation.observed_at,
                    }
                    for observation in value
                )
                resource["updated_at"] = _now()
                self._store.put(transition_session_id, resource)

    def _fail(self, resource: dict[str, Any], code: str) -> None:
        resource["status"] = "failed"
        resource["failure"] = {
            "code": code,
            "message": "Transition processing could not be completed.",
            "retryable": code == "TRANSITION_ADAPTER_FAILED",
        }
        resource["updated_at"] = _now()
        self._store.clear_chunks(resource["transition_session_id"], resource["_known_indices"])
        self._store.put(resource["transition_session_id"], resource)

    @staticmethod
    def _public(resource: dict[str, Any], cursor: str | None) -> dict[str, Any]:
        events = resource["events"]
        if cursor is not None:
            cursor_index = next(
                index
                for index, event in enumerate(events)
                if event["transition_event_id"] == cursor
            )
            events = events[cursor_index + 1 :]
        next_cursor = events[-1]["transition_event_id"] if events else cursor
        return {
            "transition_session_id": resource["transition_session_id"],
            "status": resource["status"],
            "events": copy.deepcopy(events),
            "next_event_cursor": next_cursor,
            "failure": copy.deepcopy(resource["failure"]),
            "created_at": resource["created_at"],
            "updated_at": resource["updated_at"],
        }
