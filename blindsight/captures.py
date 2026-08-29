"""Stage 0 capture lifecycle, independent of HTTP, storage, and provider implementations."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError

from .errors import ApiError, NotFound
from .excerpts import ExcerptCatalog
from .media import MediaValidator
from .providers import CaptureEvidence, CaptureProvider
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
    ) -> None:
        self.store = store
        self.provider = provider
        self.catalog = catalog
        self.media_validator = media_validator
        self.max_chunk_bytes = max_chunk_bytes
        self.max_capture_bytes = max_capture_bytes
        self.runner = runner or ThreadRunner()

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
        resource["updated_at"] = _now()
        try:
            decodable = self.media_validator.is_decodable(evidence)
        except Exception:
            self._settle_internal_error(resource)
            return
        if not decodable:
            resource["status"] = "failed"
            resource["failure"] = {
                "code": "CAPTURE_UNDECODABLE",
                "message": "The captured view was not decodable media.",
                "retryable": False,
            }
            self.store.put_capture(capture_id, resource)
            return
        try:
            result = self.provider.describe(evidence)
        except Exception:
            self._settle_internal_error(resource)
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
        self.store.put_capture(capture_id, resource)

    def _settle_internal_error(self, resource: dict[str, Any]) -> None:
        resource["status"] = "failed"
        resource["failure"] = {
            "code": "INTERNAL_ERROR",
            "message": "The capture could not be processed.",
            "retryable": True,
        }
        resource["updated_at"] = _now()
        self.store.put_capture(resource["capture_id"], resource)
