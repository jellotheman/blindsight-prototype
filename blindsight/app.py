"""The BlindSight HTTP interface, documented in full by docs/spec/openapi.yaml.

This module wires authentication and error mapping around the excerpt catalog and Stage 0 capture
resource. Later tickets add production providers and Stage 1 questions to the same application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .auth import ApiKeyMiddleware
from .captures import CaptureService
from .errors import ApiError, InternalError, NotFound
from .excerpts import ExcerptCatalog
from .media import FfprobeMediaValidator, MediaValidator
from .providers import CaptureProvider, DeterministicProvider
from .storage import CaptureStore, MemoryCaptureStore

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "excerpts" / "manifest.json"


async def _read_json(request: Request) -> Any:
    try:
        return await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ApiError(400, "INVALID_REQUEST", "The request body must be valid JSON.") from exc


DEFAULT_CARD = {
    "place_type": "demonstration excerpt",
    "place_type_confidence": "medium",
    "overview": (
        "The captured view showed a demonstration excerpt prepared for the BlindSight Stage 0 "
        "pipeline. Its visual details will be supplied by the configured production provider."
    ),
    "layout": None,
    "open_space": None,
    "people": None,
    "visual_character": None,
    "uncertainties": [
        {
            "claim": "The excerpt's visual contents were identified.",
            "detail": "This deterministic provider does not inspect visual evidence.",
        }
    ],
}


def create_app(
    *,
    api_key: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    store: CaptureStore | None = None,
    provider: CaptureProvider | None = None,
    media_validator: MediaValidator | None = None,
    max_chunk_bytes: int = 10 * 1024 * 1024,
    max_capture_bytes: int = 100 * 1024 * 1024,
) -> FastAPI:
    catalog = ExcerptCatalog(manifest_path)
    capture_service = CaptureService(
        store=store or MemoryCaptureStore(),
        provider=provider or DeterministicProvider(card_body=DEFAULT_CARD),
        catalog=catalog,
        media_validator=media_validator or FfprobeMediaValidator(),
        max_chunk_bytes=max_chunk_bytes,
        max_capture_bytes=max_capture_bytes,
    )

    app = FastAPI(title="BlindSight Stage 0/1 API")
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.envelope())

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = ApiError(400, "INVALID_REQUEST", "The request did not match the API contract.")
        return JSONResponse(status_code=error.status_code, content=error.envelope())

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        error = InternalError()
        return JSONResponse(status_code=error.status_code, content=error.envelope())

    @app.get("/v1/excerpts")
    def list_excerpts() -> dict:
        return {"items": catalog.list_excerpts()}

    @app.get("/v1/excerpts/{excerpt_id}/poster")
    def get_excerpt_poster(excerpt_id: str) -> Response:
        poster = catalog.poster_bytes(excerpt_id)
        if poster is None:
            raise NotFound(f"No excerpt with id {excerpt_id!r}.")
        return Response(
            content=poster,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/v1/captures", status_code=201)
    async def create_capture(request: Request) -> JSONResponse:
        body = await _read_json(request)
        if not isinstance(body, dict) or set(body) != {"source"}:
            raise ApiError(400, "INVALID_REQUEST", "A valid capture source is required.")
        source = body.get("source")
        if not isinstance(source, dict):
            raise ApiError(400, "INVALID_REQUEST", "A valid capture source is required.")
        is_excerpt = source.get("type") == "excerpt" and set(source) == {
            "type",
            "excerpt_id",
        }
        if is_excerpt and isinstance(source.get("excerpt_id"), str):
            resource = capture_service.create_excerpt(source["excerpt_id"])
            if resource is None:
                raise NotFound(f"No excerpt with id {source['excerpt_id']!r}.")
        elif source.get("type") == "live" and set(source) == {"type", "mime_type"}:
            mime_type = source.get("mime_type")
            if not isinstance(mime_type, str):
                raise ApiError(400, "INVALID_REQUEST", "A live MIME type is required.")
            if mime_type not in {"video/webm", "video/mp4"}:
                raise ApiError(
                    415,
                    "UNSUPPORTED_MEDIA_TYPE",
                    "Supported live capture types are video/webm and video/mp4.",
                )
            resource = capture_service.create_live(mime_type)
        else:
            raise ApiError(400, "INVALID_REQUEST", "A valid capture source is required.")
        location = f"/v1/captures/{resource['capture_id']}"
        return JSONResponse(status_code=201, content=resource, headers={"Location": location})

    @app.get("/v1/captures/{capture_id}")
    def get_capture(capture_id: str) -> JSONResponse:
        resource = capture_service.get(capture_id)
        if resource is None:
            raise NotFound(f"No capture with id {capture_id!r}.")
        headers = {"Retry-After": "1"} if resource["status"] == "processing" else {}
        return JSONResponse(content=resource, headers=headers)

    @app.put("/v1/captures/{capture_id}/chunks/{index}")
    async def put_capture_chunk(capture_id: str, index: int, request: Request) -> dict:
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/octet-stream":
            raise ApiError(
                415,
                "UNSUPPORTED_MEDIA_TYPE",
                "Capture chunks require application/octet-stream.",
            )
        return capture_service.put_chunk(capture_id, index, await request.body())

    @app.post("/v1/captures/{capture_id}/complete", status_code=202)
    async def complete_capture(capture_id: str, request: Request) -> JSONResponse:
        body = await _read_json(request)
        if not isinstance(body, dict) or set(body) != {"chunk_count", "mime_type"}:
            raise ApiError(400, "INVALID_REQUEST", "A completion body is required.")
        chunk_count = body.get("chunk_count")
        mime_type = body.get("mime_type")
        if type(chunk_count) is not int or not isinstance(mime_type, str):
            raise ApiError(400, "INVALID_REQUEST", "chunk_count and mime_type are required.")
        resource = capture_service.complete(capture_id, chunk_count, mime_type)
        location = f"/v1/captures/{capture_id}"
        return JSONResponse(
            status_code=202,
            content=resource,
            headers={"Location": location, "Retry-After": "1"},
        )

    return app


def mount_reference_client(app: FastAPI, static_dir: Path) -> FastAPI:
    """Serve the reference web client from the same application as the API.

    Deliberately unauthenticated: this only serves the static client shell, never `/v1` data, so
    it has no privileged path into the backend -- see docs/spec/phase-0-1.md.
    """
    from fastapi.staticfiles import StaticFiles

    @app.get("/")
    def index() -> Response:
        return Response(
            content=(static_dir / "index.html").read_bytes(),
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
