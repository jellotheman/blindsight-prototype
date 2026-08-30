"""Stage 0 capture lifecycle, independent of HTTP, storage, and provider implementations."""

from __future__ import annotations

import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError

from .concurrency import run_with_deadline
from .errors import ApiError, NotFound
from .evidence import EvidenceStore, NullEvidenceStore, RunClock
from .excerpts import ExcerptCatalog
from .media import MediaValidator
from .providers import CaptureEvidence, CaptureProvider, ProviderResult
from .scene_card import SceneCard, SceneCardBody
from .storage import CaptureStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class ThreadRunner:
    def submit(self, operation: Callable[[], None]) -> None:
        threading.Thread(target=operation, daemon=True).start()


class CaptureService:
    def __init__(
        self,
        *,
        store: CaptureStore,
        provider: CaptureProvider,
        catalog: ExcerptCatalog,
        media_validator: MediaValidator,
        max_chunk_bytes: int = 10 * 1024 * 1024,
        max_capture_bytes: int = 100 * 1024 * 1024,
        runner: ThreadRunner | None = None,
        evidence_store: EvidenceStore | None = None,
        processing_deadline_seconds: float = 90.0,
    ) -> None:
        self.store = store
        self.provider = provider
        self.catalog = catalog
        self.media_validator = media_validator
        self.max_chunk_bytes = max_chunk_bytes
        self.max_capture_bytes = max_capture_bytes
        self.runner = runner or ThreadRunner()
        self.evidence_store = evidence_store or NullEvidenceStore()
        self.processing_deadline_seconds = processing_deadline_seconds

    def create_excerpt(self, excerpt_id: str) -> dict[str, Any] | None:
        evidence = self.catalog.evidence_bytes(excerpt_id)
        if evidence is None:
            return None

        resource = self._create_resource({"type": "excerpt", "excerpt_id": excerpt_id}, "processing")
        self.runner.submit(
            lambda: self._process(
                resource["capture_id"], CaptureEvidence(content=evidence, media_type="video/mp4")
            )
        )
        return resource

    def create_live(self, mime_type: str) -> dict[str, Any]:
        return self._create_resource({"type": "live", "mime_type": mime_type}, "recording")

    def get(self, capture_id: str) -> dict[str, Any] | None:
        return self.store.get_capture(capture_id)

    def put_chunk(self, capture_id: str, index: int, content: bytes) -> dict[str, Any]:
        resource = self.store.get_capture(capture_id)
        if resource is None:
            raise NotFound(f"No capture with id {capture_id!r}.")
        if resource["source"]["type"] != "live" or resource["status"] != "recording":
            raise ApiError(
                409,
                "INVALID_STATE",
                "The capture is no longer accepting chunks.",
                details={"status": resource["status"]},
            )
        if not 0 <= index <= 9999 or not content:
            raise ApiError(400, "INVALID_REQUEST", "A non-empty chunk and valid index are required.")
        if len(content) > self.max_chunk_bytes:
            raise ApiError(413, "CAPTURE_TOO_LARGE", "The chunk exceeds the configured byte limit.")

        outcome = self.store.put_chunk(capture_id, index, content, self.max_capture_bytes)
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
                "CAPTURE_TOO_LARGE",
                "The capture exceeds the configured byte limit.",
            )
        return {
            "capture_id": capture_id,
            "index": index,
            "bytes": len(content),
            "idempotent": outcome == "idempotent",
        }

    def complete(self, capture_id: str, chunk_count: int, mime_type: str) -> dict[str, Any]:
        resource = self.store.get_capture(capture_id)
        if resource is None:
            raise NotFound(f"No capture with id {capture_id!r}.")
        if resource["source"]["type"] != "live" or resource["status"] != "recording":
            raise ApiError(
                409,
                "INVALID_STATE",
                "The capture cannot be completed from its current state.",
                details={"status": resource["status"]},
            )
        if not 1 <= chunk_count <= 10000:
            raise ApiError(400, "INVALID_REQUEST", "chunk_count must be between 1 and 10000.")
        if mime_type not in {"video/webm", "video/mp4"}:
            raise ApiError(
                415,
                "UNSUPPORTED_MEDIA_TYPE",
                "Supported live capture types are video/webm and video/mp4.",
            )
        if mime_type != resource["source"]["mime_type"]:
            raise ApiError(400, "INVALID_REQUEST", "Completion MIME type must match the capture.")

        chunks = self.store.get_chunks(capture_id, chunk_count)
        missing = [index for index in range(chunk_count) if index not in chunks]
        if missing:
            raise ApiError(
                409,
                "CAPTURE_INCOMPLETE",
                "One or more declared chunks have not arrived.",
                retryable=True,
                details={"missing_indices": missing},
            )
        content = b"".join(chunks[index] for index in range(chunk_count))
        if len(content) > self.max_capture_bytes:
            raise ApiError(
                413,
                "CAPTURE_TOO_LARGE",
                "The capture exceeds the configured byte limit.",
            )

        resource["status"] = "processing"
        resource["updated_at"] = _now()
        self.store.put_capture(capture_id, resource)
        # The clip is assembled; the capture no longer accepts chunks, so the per-chunk copies
        # are dead weight in the shared store from here on.
        self.store.delete_chunks(capture_id, chunk_count)
        evidence = CaptureEvidence(content=content, media_type=mime_type)
        self.runner.submit(lambda: self._process(capture_id, evidence))
        return resource

    def _create_resource(self, source: dict[str, Any], status: str) -> dict[str, Any]:
        capture_id = _identifier("cap")
        scene_session_id = _identifier("ses")
        created_at = _now()
        resource: dict[str, Any] = {
            "capture_id": capture_id,
            "scene_session_id": scene_session_id,
            "source": source,
            "status": status,
            "card": None,
            "failure": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        self.store.create_capture(
            resource,
            {"scene_session_id": scene_session_id, "capture_id": capture_id, "conversation": []},
        )
        return resource

    def _process(self, capture_id: str, evidence: CaptureEvidence) -> None:
        resource = self.store.get_capture(capture_id)
        if resource is None:
            return
        clock = RunClock()
        evidence = replace(evidence, clock=clock)
        resource["updated_at"] = _now()
        try:
            self.evidence_store.retain_capture(
                capture_id,
                resource["scene_session_id"],
                resource["source"],
                evidence,
            )
        except Exception:
            self._settle_internal_error(resource)
            return
        clock.mark("evidence_retained_ms")
        try:
            decodable = self.media_validator.is_decodable(evidence)
        except Exception:
            self._settle_internal_error(resource, clock=clock)
            return
        clock.mark("media_validated_ms")
        if not decodable:
            resource["status"] = "failed"
            resource["failure"] = {
                "code": "CAPTURE_UNDECODABLE",
                "message": "The captured view was not decodable media.",
                "retryable": False,
            }
            resource["updated_at"] = _now()
            clock.mark("completed_ms")
            if not self._finish_evidence(
                capture_id,
                None,
                accepted_card=None,
                failure=resource["failure"],
                timings=clock.as_dict(),
            ):
                self._settle_evidence_error(resource)
                return
            self.store.put_capture(capture_id, resource)
            return
        try:
            remaining_seconds = clock.remaining_seconds(self.processing_deadline_seconds)
            outcome: str
            value: Any
            if remaining_seconds <= 0:
                outcome, value = ("timeout", None)
            else:
                outcome, value = self._describe_with_deadline(evidence, remaining_seconds)
            if outcome == "timeout":
                resource["status"] = "failed"
                resource["failure"] = {
                    "code": "PROVIDER_TIMEOUT",
                    "message": "The provider exceeded the processing deadline.",
                    "retryable": True,
                }
                resource["updated_at"] = _now()
                clock.mark("completed_ms")
                if not self._finish_evidence(
                    capture_id,
                    None,
                    accepted_card=None,
                    failure=resource["failure"],
                    timings=clock.as_dict(),
                ):
                    self._settle_evidence_error(resource)
                    return
                self.store.put_capture(capture_id, resource)
                return
            if outcome == "error":
                if isinstance(value, BaseException):
                    raise value
                raise RuntimeError("Provider execution failed without an exception.")
            if not isinstance(value, ProviderResult):
                raise RuntimeError("Provider returned an invalid result type.")
            result = value
        except Exception:
            self._settle_internal_error(resource, clock=clock)
            return
        clock.mark("provider_completed_ms")
        if result.failure_kind is not None:
            failure_code = {
                "timeout": "PROVIDER_TIMEOUT",
                "transport": "PROVIDER_UNAVAILABLE",
                "invalid_output": "MODEL_OUTPUT_INVALID",
            }[result.failure_kind]
            resource["status"] = "failed"
            resource["failure"] = {
                "code": failure_code,
                "message": {
                    "timeout": "The provider timed out while processing the captured view.",
                    "transport": "The provider was unavailable while processing the captured view.",
                    "invalid_output": "The provider returned an invalid scene card.",
                }[result.failure_kind],
                "retryable": result.failure_kind in {"timeout", "transport"},
            }
            resource["updated_at"] = _now()
            clock.mark("completed_ms")
            if not self._finish_evidence(
                capture_id,
                result,
                accepted_card=None,
                failure=resource["failure"],
                timings=clock.as_dict(),
            ):
                self._settle_evidence_error(resource)
                return
            self.store.put_capture(capture_id, resource)
            return
        try:
            body = SceneCardBody.model_validate(result.card_body)
        except ValidationError:
            resource["status"] = "failed"
            resource["failure"] = {
                "code": "MODEL_OUTPUT_INVALID",
                "message": result.error or "The provider returned an invalid scene card.",
                "retryable": False,
            }
        else:
            resource["status"] = "succeeded"
            resource["card"] = SceneCard(
                capture_id=resource["capture_id"],
                scene_session_id=resource["scene_session_id"],
                revision=1,
                evidence=[resource["capture_id"]],
                card=body,
            ).model_dump(mode="json")
            self.store.put_media(capture_id, evidence.content, evidence.media_type)
        resource["updated_at"] = _now()
        clock.mark("completed_ms")
        if not self._finish_evidence(
            capture_id,
            result,
            accepted_card=resource["card"],
            failure=resource["failure"],
            timings=clock.as_dict(),
        ):
            self._settle_evidence_error(resource)
            return
        self.store.put_capture(capture_id, resource)

    def _describe_with_deadline(
        self, evidence: CaptureEvidence, timeout_seconds: float
    ) -> tuple[str, Any]:
        return run_with_deadline(lambda: self.provider.describe(evidence), timeout_seconds)

    def _settle_internal_error(
        self, resource: dict[str, Any], *, clock: RunClock | None = None
    ) -> None:
        resource["status"] = "failed"
        resource["failure"] = {
            "code": "INTERNAL_ERROR",
            "message": "The capture could not be processed.",
            "retryable": True,
        }
        resource["updated_at"] = _now()
        if clock is not None:
            clock.mark("completed_ms")
            try:
                self.evidence_store.finish(
                    resource["capture_id"],
                    None,
                    accepted_card=None,
                    failure=resource["failure"],
                    timings=clock.as_dict(),
                )
            except Exception:
                pass
        self.store.put_capture(resource["capture_id"], resource)

    def _finish_evidence(
        self,
        capture_id: str,
        result: Any,
        *,
        accepted_card: dict[str, Any] | None,
        failure: dict[str, Any] | None,
        timings: dict[str, float],
    ) -> bool:
        try:
            self.evidence_store.finish(
                capture_id,
                result,
                accepted_card=accepted_card,
                failure=failure,
                timings=timings,
            )
        except Exception:
            return False
        return True

    def _settle_evidence_error(self, resource: dict[str, Any]) -> None:
        resource["status"] = "failed"
        resource["card"] = None
        resource["failure"] = {
            "code": "INTERNAL_ERROR",
            "message": "The capture evidence could not be retained.",
            "retryable": True,
        }
        resource["updated_at"] = _now()
        self.store.put_capture(resource["capture_id"], resource)
