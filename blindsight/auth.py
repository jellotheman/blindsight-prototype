"""Shared-key authentication for every `/v1` operation.

One static key, read from configuration and compared in constant time. There are no users,
accounts, or per-client keys in Stage 0/1 -- see docs/spec/phase-0-1.md.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .errors import Unauthorized

API_KEY_HEADER = "X-API-Key"
VERSIONED_PREFIX = "/v1"


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path.startswith(VERSIONED_PREFIX):
            supplied = request.headers.get(API_KEY_HEADER, "")
            if not hmac.compare_digest(supplied, self._api_key):
                error = Unauthorized()
                return JSONResponse(status_code=error.status_code, content=error.envelope())
        return await call_next(request)
