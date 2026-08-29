"""The BlindSight HTTP interface: one FastAPI application, documented in full by
docs/spec/openapi.yaml. This slice wires routing, authentication, and error mapping around the
excerpt catalog; later tickets add capture and question resources to the same app.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .auth import ApiKeyMiddleware
from .errors import ApiError, InternalError, NotFound
from .excerpts import ExcerptCatalog

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "excerpts" / "manifest.json"


def create_app(*, api_key: str, manifest_path: Path = DEFAULT_MANIFEST) -> FastAPI:
    catalog = ExcerptCatalog(manifest_path)

    app = FastAPI(title="BlindSight Stage 0/1 API")
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.envelope())

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
