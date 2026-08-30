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
    return _iso(datetime.now(timezone.utc))


def _iso(at: datetime) -> str:
    return at.isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class TransitionObservation:
    """One observation from causal inference; it intentionally carries no scene description."""

    observed_at: str


class TransitionAdapter(Protocol):
    """Continuous-inference boundary used by local tests and the Modal deployment."""

    def start(self, transition_session_id: str) -> None: ...

    def process_prefix(
        self, transition_session_id: str, items: list[tuple[int, bytes, str]]
    ) -> list[TransitionObservation]: ...

    def stop(self, transition_session_id: str) -> None: ...


class InMemoryTransitionAdapter:
    """Deterministic local adapter; observations can be selected by processed chunk index.

    ``processed_prefixes`` records one entry per prefix call: the session id, the claimed
    contiguous indices, and the ``(index, content, media_type)`` items handed to inference.
    """

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
        self.processed_prefixes: list[dict[str, Any]] = []
        self.stopped_session_ids: list[str] = []

    def start(self, transition_session_id: str) -> None:
        if self._fail_on_start is not None:
            raise self._fail_on_start

    def process_prefix(
        self, transition_session_id: str, items: list[tuple[int, bytes, str]]
    ) -> list[TransitionObservation]:
        if self._fail_on_process is not None:
            raise self._fail_on_process
        self.processed_prefixes.append(
            {
                "transition_session_id": transition_session_id,
                "indices": [index for index, _, _ in items],
                "items": [
                    (index, bytes(content), media_type) for index, content, media_type in items
                ],
            }
        )
        return [
            observation
            for index, _, _ in items
            for observation in self._observations_by_index.get(index, [])
        ]

    def stop(self, transition_session_id: str) -> None:
        self.stopped_session_ids.append(transition_session_id)


class ModalTransitionWorker(Protocol):
    """The narrowly scoped worker shape used by the production Modal adapter."""

    def start(self, transition_session_id: str) -> None: ...

    def process_prefix(
        self, transition_session_id: str, items: list[tuple[int, bytes, str]]
    ) -> list[TransitionObservation]: ...

    def stop(self, transition_session_id: str) -> None: ...


class ModalTransitionAdapter:
    """Production adapter delegating continuous inference to a pre-created Modal worker."""

    def __init__(self, worker: ModalTransitionWorker) -> None:
        self._worker = worker

    def start(self, transition_session_id: str) -> None:
        self._worker.start(transition_session_id)

    def process_prefix(
        self, transition_session_id: str, items: list[tuple[int, bytes, str]]
    ) -> list[TransitionObservation]:
        return self._worker.process_prefix(transition_session_id, items)

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

    def try_claim(
        self, transition_session_id: str, index: int, *, lease_seconds: float
    ) -> str | None: ...

    def release_claim(self, transition_session_id: str, index: int, token: str) -> None: ...


class MemoryTransitionSessionStore:
    """Thread-safe local session store.  It retains hashes after processing for idempotency."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._lock = threading.RLock()
        self._clock = clock or _utc_now
        self._resources: dict[str, dict[str, Any]] = {}
        self._chunks: dict[tuple[str, int], tuple[bytes, str]] = {}
        self._processed_hashes: dict[tuple[str, int], str] = {}
        self._claims: dict[tuple[str, int], dict[str, Any]] = {}

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

    def try_claim(self, transition_session_id: str, index: int, *, lease_seconds: float) -> str | None:
        key = (transition_session_id, index)
        token = uuid.uuid4().hex
        with self._lock:
            existing = self._claims.get(key)
            if existing is not None:
                claimed_at = _parse_timestamp(existing.get("claimed_at"))
                is_fresh = claimed_at is not None and (
                    self._clock() - claimed_at
                ).total_seconds() < lease_seconds
                if is_fresh:
                    return None
            self._claims[key] = {"token": token, "claimed_at": _iso(self._clock())}
            return token

    def release_claim(self, transition_session_id: str, index: int, token: str) -> None:
        key = (transition_session_id, index)
        with self._lock:
            claim = self._claims.get(key)
            if claim is not None and claim.get("token") == token:
                self._claims.pop(key, None)


class ModalTransitionSessionStore:
    """Transition-session state kept in the existing Modal Dict, never in request memory."""

    def __init__(self, dictionary: Any, *, clock: Callable[[], datetime] | None = None) -> None:
        self._dictionary = dictionary
        self._clock = clock or _utc_now

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

    @staticmethod
    def _claim_key(transition_session_id: str, index: int) -> str:
        return f"transition-claim:{transition_session_id}:{index}"

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

    def try_claim(self, transition_session_id: str, index: int, *, lease_seconds: float) -> str | None:
        key = self._claim_key(transition_session_id, index)
        token = uuid.uuid4().hex
        claim = {"token": token, "claimed_at": _iso(self._clock())}
        if self._dictionary.put(key, claim, skip_if_exists=True):
            return token
        existing = self._dictionary.get(key)
        if not isinstance(existing, dict):
            return None
        claimed_at = _parse_timestamp(existing.get("claimed_at"))
        if claimed_at is not None and (
            self._clock() - claimed_at
        ).total_seconds() < lease_seconds:
            return None
        # The lease is stale (or unreadable).  Steal it only while the stored token still
        # matches what we read; otherwise another container already took the index.
        current = self._dictionary.get(key)
        if not isinstance(current, dict) or current.get("token") != existing.get("token"):
            return None
        self._dictionary.pop(key, None)
        if self._dictionary.put(key, claim, skip_if_exists=True):
            return token
        return None

    def release_claim(self, transition_session_id: str, index: int, token: str) -> None:
        key = self._claim_key(transition_session_id, index)
        claim = self._dictionary.get(key)
        if isinstance(claim, dict) and claim.get("token") == token:
            self._dictionary.pop(key, None)


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
        idle_timeout_seconds: float = 300.0,
        max_batch_items: int = 30,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._adapter = adapter
        self._runner = runner or ThreadRunner()
        self._max_chunk_bytes = max_chunk_bytes
        self._max_queued_bytes = max_queued_bytes
        self._processing_deadline_seconds = processing_deadline_seconds
        self._idle_timeout_seconds = idle_timeout_seconds
        self._max_batch_items = max_batch_items
        self._claim_lease_seconds = 2 * processing_deadline_seconds + 30.0
        self._clock = clock or _utc_now
        self._drain_lock = threading.Lock()

    def _timestamp(self) -> str:
        return _iso(self._clock())

    def create(self) -> dict[str, Any]:
        session_id = _identifier("trs")
        created_at = self._timestamp()
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
        resource = self._expire_if_idle(transition_session_id, resource)
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
            resource["updated_at"] = self._timestamp()
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
        resource["updated_at"] = self._timestamp()
        self._store.put(transition_session_id, resource)
        self._drain(transition_session_id)

    def _drain(self, transition_session_id: str) -> None:
        with self._drain_lock:
            while True:
                resource = self._store.get(transition_session_id)
                if resource is None or resource["status"] != "active":
                    return
                head_index = resource["_next_index"]
                items: list[tuple[int, bytes, str]] = []
                claims: list[tuple[int, str]] = []
                retry = False
                index = head_index
                while len(items) < self._max_batch_items:
                    chunk = self._store.get_chunk(transition_session_id, index)
                    if chunk is None:
                        break
                    token = self._store.try_claim(
                        transition_session_id, index, lease_seconds=self._claim_lease_seconds
                    )
                    if token is None:
                        if items:
                            # A later index is leased elsewhere; the prefix claimed so far is
                            # still contiguous, so process it rather than waiting on the queue.
                            break
                        # Another container owns a fresh lease on the head index.  Once the
                        # owner advances `_next_index`, the next chunk PUT re-triggers the
                        # drain here; if it already advanced, retry from the new head.
                        resource = self._store.get(transition_session_id)
                        if resource is None or resource["status"] != "active":
                            return
                        if resource["_next_index"] == head_index:
                            return
                        retry = True
                        break
                    items.append((index, chunk[0], chunk[1]))
                    claims.append((index, token))
                    index += 1
                if retry:
                    continue
                if not items:
                    return
                outcome, value = run_with_deadline(
                    lambda: self._adapter.process_prefix(transition_session_id, items),
                    self._processing_deadline_seconds,
                )
                resource = self._store.get(transition_session_id)
                if resource is None or resource["status"] != "active":
                    self._release_claims(transition_session_id, claims)
                    return
                if (
                    outcome != "result"
                    or not isinstance(value, list)
                    or not all(
                        isinstance(observation, TransitionObservation) for observation in value
                    )
                ):
                    self._release_claims(transition_session_id, claims)
                    self._fail(resource, "TRANSITION_ADAPTER_FAILED")
                    return
                for item_index, content, _ in items:
                    self._store.mark_processed(transition_session_id, item_index, content)
                resource["_next_index"] = items[-1][0] + 1
                resource["events"].extend(
                    {
                        "transition_event_id": _identifier("tev"),
                        "observed_at": observation.observed_at,
                    }
                    for observation in value
                )
                resource["updated_at"] = self._timestamp()
                self._store.put(transition_session_id, resource)
                self._release_claims(transition_session_id, claims)

    def _release_claims(
        self, transition_session_id: str, claims: list[tuple[int, str]]
    ) -> None:
        for index, token in claims:
            self._store.release_claim(transition_session_id, index, token)

    def _fail(self, resource: dict[str, Any], code: str) -> None:
        resource["status"] = "failed"
        resource["failure"] = {
            "code": code,
            "message": "Transition processing could not be completed.",
            "retryable": code == "TRANSITION_ADAPTER_FAILED",
        }
        resource["updated_at"] = self._timestamp()
        self._store.clear_chunks(resource["transition_session_id"], resource["_known_indices"])
        self._store.put(resource["transition_session_id"], resource)

    def _expire_if_idle(self, transition_session_id: str, resource: dict[str, Any]) -> dict[str, Any]:
        if self._idle_timeout_seconds <= 0 or resource["status"] != "active":
            return resource
        updated_at = _parse_timestamp(resource.get("updated_at"))
        if updated_at is None:
            return resource
        if (self._clock() - updated_at).total_seconds() < self._idle_timeout_seconds:
            return resource
        resource["status"] = "failed"
        resource["failure"] = {
            "code": "TRANSITION_ADAPTER_FAILED",
            "message": "The transition session expired after a period of inactivity.",
            "retryable": False,
        }
        resource["updated_at"] = self._timestamp()
        self._store.clear_chunks(transition_session_id, resource["_known_indices"])
        self._store.put(transition_session_id, resource)
        try:
            run_with_deadline(
                lambda: self._adapter.stop(transition_session_id), self._processing_deadline_seconds
            )
        except Exception:
            # Stop was accepted locally; a worker-specific cleanup failure cannot resurrect an
            # expired session or make it accept more media.
            pass
        return resource

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
