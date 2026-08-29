"""The shared error envelope. Every immediate HTTP error uses this shape, never a bare detail
string, so clients can branch on `code` instead of parsing prose."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """An immediate (non-async) API failure, mapped to the documented error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}

    def envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }


class Unauthorized(ApiError):
    def __init__(self) -> None:
        super().__init__(
            401,
            "UNAUTHORIZED",
            "A valid X-API-Key header is required.",
        )


class NotFound(ApiError):
    def __init__(self, message: str = "The requested resource was not found.") -> None:
        super().__init__(404, "NOT_FOUND", message)


class InternalError(ApiError):
    def __init__(self, message: str = "The request could not be completed.") -> None:
        super().__init__(500, "INTERNAL_ERROR", message, retryable=True)
