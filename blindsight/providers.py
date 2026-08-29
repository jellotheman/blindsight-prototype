"""Provider adapters and the bounded Reka-primary Stage 0 policy."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from .prompt import build_scene_card_prompt
from .scene_card import SceneCardBody

FailureKind = Literal["invalid_output", "transport", "timeout"]


@dataclass(frozen=True)
class CaptureEvidence:
    content: bytes
    media_type: str
    media_url: str | None = None
    clock: Any | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    model: str
    attempt: int
    raw_text: str = ""
    card_body: dict[str, Any] | None = None
    failure_kind: FailureKind | None = None
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResult:
    raw_text: str
    card_body: dict[str, Any] | None
    error: str | None = None
    failure_kind: FailureKind | None = None
    provider: str | None = None
    model: str | None = None
    attempt: int | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    attempts: tuple[ProviderAttempt, ...] = ()
    prompt: str | None = None


class CaptureProvider(Protocol):
    def describe(self, evidence: CaptureEvidence) -> ProviderResult: ...


class AttemptProvider(Protocol):
    name: str
    model: str

    def describe(self, evidence: CaptureEvidence, prompt: str, attempt: int) -> ProviderAttempt: ...


class MediaPublisher(Protocol):
    def publish(self, evidence: CaptureEvidence) -> str: ...

    def revoke(self, url: str) -> None: ...


class DeterministicProvider:
    """A predictable provider double for the HTTP acceptance suite and walking deployment."""

    def __init__(self, *, card_body: dict[str, Any]) -> None:
        self._card_body = card_body

    def describe(self, evidence: CaptureEvidence) -> ProviderResult:
        return ProviderResult(raw_text=json.dumps(self._card_body), card_body=self._card_body)


class ProductionProvider:
    """Exactly two invalid Reka outputs, then exactly one schema-constrained Gemini fallback."""

    def __init__(
        self,
        *,
        reka: AttemptProvider,
        gemini: AttemptProvider,
        media_urls: MediaPublisher,
    ) -> None:
        self.reka = reka
        self.gemini = gemini
        self.media_urls = media_urls

    def describe(self, evidence: CaptureEvidence) -> ProviderResult:
        prompt = build_scene_card_prompt()
        media_url = self.media_urls.publish(evidence)
        if not media_url.startswith("https://"):
            raise RuntimeError("Production Reka media transport requires HTTPS.")
        url_evidence = replace(evidence, media_url=media_url)
        attempts: list[ProviderAttempt] = []

        try:
            for number in (1, 2):
                started = _mark(evidence, f"reka_{number}_started_ms")
                attempt = self.reka.describe(url_evidence, prompt, number)
                completed = _mark(evidence, f"reka_{number}_completed_ms")
                attempt = _canonicalize(
                    replace(
                        attempt,
                        timings={"started_ms": started, "completed_ms": completed},
                    )
                )
                attempts.append(attempt)
                if attempt.card_body is not None:
                    return _selected(attempt, attempts, prompt=prompt)
                if attempt.failure_kind != "invalid_output":
                    # A transport error or timeout means Reka never responded at all, so a
                    # second identical call is unlikely to help. Don't spend it -- go straight
                    # to Gemini instead of forfeiting the fallback for a recoverable capture.
                    break
        finally:
            self.media_urls.revoke(media_url)

        started = _mark(evidence, "gemini_1_started_ms")
        fallback = self.gemini.describe(evidence, prompt, 1)
        completed = _mark(evidence, "gemini_1_completed_ms")
        fallback = _canonicalize(replace(
            fallback, timings={"started_ms": started, "completed_ms": completed}
        ))
        attempts.append(fallback)
        return (
            _selected(fallback, attempts, prompt=prompt)
            if fallback.card_body is not None
            else _failed(fallback, attempts, prompt=prompt)
        )


class SingleProvider:
    """Run one chosen adapter once, primarily for immutable evidence replay."""

    def __init__(self, adapter: AttemptProvider, *, media_urls: MediaPublisher | None = None) -> None:
        self.adapter = adapter
        self.media_urls = media_urls
        self.name = adapter.name
        self.model = adapter.model

    def describe(self, evidence: CaptureEvidence) -> ProviderResult:
        media_url: str | None = None
        if self.adapter.name == "reka":
            if self.media_urls is None:
                raise RuntimeError("Reka replay requires a published provider-media URL.")
            media_url = self.media_urls.publish(evidence)
            evidence = replace(evidence, media_url=media_url)
        prompt = build_scene_card_prompt()
        try:
            attempt = _canonicalize(self.adapter.describe(evidence, prompt, 1))
        finally:
            if self.adapter.name == "reka":
                assert self.media_urls is not None
                assert media_url is not None
                self.media_urls.revoke(media_url)
        return (
            _selected(attempt, [attempt], prompt=prompt)
            if attempt.card_body is not None
            else _failed(attempt, [attempt], prompt=prompt)
        )


class RekaChatAdapter:
    name = "reka"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.environ.get("BLINDSIGHT_REKA_MODEL", "reka-flash")
        self.timeout_seconds = timeout_seconds or float(
            os.environ.get("BLINDSIGHT_REKA_TIMEOUT_SECONDS", "40")
        )
        if client is not None:
            self.client = client
            return
        key = api_key or os.environ.get("REKA_API_KEY")
        if not key:
            raise RuntimeError("REKA_API_KEY is required for the production provider.")
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://api.reka.ai/v1", api_key=key, timeout=self.timeout_seconds
        )

    def describe(self, evidence: CaptureEvidence, prompt: str, attempt: int) -> ProviderAttempt:
        if evidence.media_url is None:
            raise ValueError("Reka Chat requires a published media URL")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": evidence.media_url}},
                            {"type": "text", "text": prompt.removesuffix("{")},
                        ],
                    },
                    {"role": "assistant", "content": "{"},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            kind: FailureKind = "timeout" if _is_timeout(exc) else "transport"
            return ProviderAttempt(
                provider=self.name,
                model=self.model,
                attempt=attempt,
                failure_kind=kind,
                error=str(exc),
                usage={"timeout_seconds": self.timeout_seconds},
            )

        content = response.choices[0].message.content or ""
        raw_text = content.strip()
        if not raw_text.startswith("{"):
            raw_text = "{" + raw_text
        usage = _native_usage(getattr(response, "usage", None))
        usage["timeout_seconds"] = self.timeout_seconds
        response_usage = getattr(response, "usage", None)
        if response_usage is not None:
            for name in (
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = getattr(response_usage, name, None)
                if value is not None:
                    usage[name] = value
        return _parse_attempt(self.name, self.model, attempt, raw_text, usage)


class GeminiAdapter:
    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.environ.get("BLINDSIGHT_GEMINI_MODEL", "gemini-3.7-flash")
        self.timeout_seconds = timeout_seconds or float(
            os.environ.get("BLINDSIGHT_GEMINI_TIMEOUT_SECONDS", "40")
        )
        if client is not None:
            self.client = client
            return
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for the production fallback.")
        from google import genai
        from google.genai import types

        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )

    def describe(self, evidence: CaptureEvidence, prompt: str, attempt: int) -> ProviderAttempt:
        try:
            response = self.client.interactions.create(
                model=self.model,
                input=[
                    {"type": "text", "text": prompt.removesuffix("{")},
                    {
                        "type": "video",
                        "data": base64.b64encode(evidence.content).decode("ascii"),
                        "mime_type": evidence.media_type,
                    },
                ],
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": SceneCardBody.model_json_schema(),
                },
            )
        except Exception as exc:
            kind: FailureKind = "timeout" if _is_timeout(exc) else "transport"
            return ProviderAttempt(
                provider=self.name,
                model=self.model,
                attempt=attempt,
                failure_kind=kind,
                error=str(exc),
                usage={"timeout_seconds": self.timeout_seconds, "schema_mode": "response_schema"},
            )

        raw_text = (getattr(response, "output_text", None) or "").strip()
        usage: dict[str, Any] = _native_usage(
            getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
        )
        usage.update({
            "timeout_seconds": self.timeout_seconds,
            "schema_mode": "response_schema",
        })
        response_usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
        if response_usage is not None:
            for name in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "prompt_token_count",
                "candidates_token_count",
                "total_token_count",
            ):
                value = getattr(response_usage, name, None)
                if value is not None:
                    usage[name] = value
        return _parse_attempt(self.name, self.model, attempt, raw_text, usage)


def _parse_attempt(
    provider: str,
    model: str,
    attempt: int,
    raw_text: str,
    usage: dict[str, Any],
) -> ProviderAttempt:
    try:
        payload = json.loads(raw_text)
        body = SceneCardBody.model_validate(payload).model_dump(mode="json")
    except (json.JSONDecodeError, ValidationError) as exc:
        return ProviderAttempt(
            provider=provider,
            model=model,
            attempt=attempt,
            raw_text=raw_text,
            failure_kind="invalid_output",
            error=str(exc),
            usage=usage,
        )
    return ProviderAttempt(
        provider=provider,
        model=model,
        attempt=attempt,
        raw_text=raw_text,
        card_body=body,
        usage=usage,
    )


def _canonicalize(attempt: ProviderAttempt) -> ProviderAttempt:
    if attempt.failure_kind in {"transport", "timeout"}:
        return replace(attempt, card_body=None)
    parsed = _parse_attempt(
        attempt.provider, attempt.model, attempt.attempt, attempt.raw_text, attempt.usage
    )
    return replace(parsed, timings=attempt.timings)


def _selected(
    attempt: ProviderAttempt,
    attempts: list[ProviderAttempt],
    *,
    prompt: str | None = None,
) -> ProviderResult:
    return ProviderResult(
        raw_text=attempt.raw_text,
        card_body=attempt.card_body,
        provider=attempt.provider,
        model=attempt.model,
        attempt=attempt.attempt,
        usage=attempt.usage,
        timings=attempt.timings,
        attempts=tuple(attempts),
        prompt=prompt,
    )


def _failed(
    attempt: ProviderAttempt,
    attempts: list[ProviderAttempt],
    *,
    prompt: str | None = None,
) -> ProviderResult:
    return ProviderResult(
        raw_text=attempt.raw_text,
        card_body=None,
        error=attempt.error,
        failure_kind=attempt.failure_kind or "invalid_output",
        provider=attempt.provider,
        model=attempt.model,
        attempt=attempt.attempt,
        usage=attempt.usage,
        timings=attempt.timings,
        attempts=tuple(attempts),
        prompt=prompt,
    )


def _is_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text or "deadline exceeded" in text


def _mark(evidence: CaptureEvidence, name: str) -> float:
    if evidence.clock is None:
        return 0.0
    return evidence.clock.mark(name)


def _native_usage(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}
