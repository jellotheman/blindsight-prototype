from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from blindsight.app import create_app
from blindsight.evidence import FileEvidenceStore
from blindsight.media_urls import MemoryMediaUrlStore
from blindsight.prompt import build_scene_card_prompt
from blindsight.providers import (
    CaptureEvidence,
    ProductionProvider,
    ProviderAttempt,
    ProviderResult,
    GeminiAdapter,
    RekaChatAdapter,
)

from .test_captures import VALID_CARD_BODY, wait_for_capture


class AcceptingMediaValidator:
    def is_decodable(self, evidence: CaptureEvidence) -> bool:
        return True


class ScriptedAdapter:
    def __init__(self, name: str, model: str, attempts: list[ProviderAttempt]) -> None:
        self.name = name
        self.model = model
        self._attempts = iter(attempts)
        self.calls: list[CaptureEvidence] = []

    def describe(self, evidence: CaptureEvidence, prompt: str, attempt: int) -> ProviderAttempt:
        self.calls.append(evidence)
        scripted = next(self._attempts)
        return ProviderAttempt(
            provider=self.name,
            model=self.model,
            attempt=attempt,
            raw_text=scripted.raw_text,
            card_body=scripted.card_body,
            failure_kind=scripted.failure_kind,
            error=scripted.error,
            usage=scripted.usage,
        )


def invalid(provider: str, attempt: int, raw: str) -> ProviderAttempt:
    return ProviderAttempt(
        provider=provider,
        model=f"{provider}-model",
        attempt=attempt,
        raw_text=raw,
        card_body=None,
        failure_kind="invalid_output",
        error="not canonical JSON",
    )


def valid(provider: str, attempt: int) -> ProviderAttempt:
    return ProviderAttempt(
        provider=provider,
        model=f"{provider}-model",
        attempt=attempt,
        raw_text=json.dumps(VALID_CARD_BODY),
        card_body=VALID_CARD_BODY,
        usage={"input_tokens": 123, "output_tokens": 45},
    )


def transport(provider: str, attempt: int, error: str) -> ProviderAttempt:
    return ProviderAttempt(
        provider=provider,
        model=f"{provider}-model",
        attempt=attempt,
        failure_kind="transport",
        error=error,
    )


def timeout(provider: str, attempt: int, error: str) -> ProviderAttempt:
    return ProviderAttempt(
        provider=provider,
        model=f"{provider}-model",
        attempt=attempt,
        failure_kind="timeout",
        error=error,
    )


def test_two_invalid_reka_attempts_fall_through_to_one_recorded_gemini_call(
    tmp_path: Path, api_key: str, auth_headers: dict[str, str]
) -> None:
    reka = ScriptedAdapter("reka", "reka-flash", [invalid("reka", 1, "no"), invalid("reka", 2, "{")])
    gemini = ScriptedAdapter("gemini", "gemini-3.7-flash", [valid("gemini", 1)])
    media_urls = MemoryMediaUrlStore("https://testserver")
    provider = ProductionProvider(reka=reka, gemini=gemini, media_urls=media_urls)
    evidence = FileEvidenceStore(tmp_path / "runs")
    client = TestClient(
        create_app(
            api_key=api_key,
            provider=provider,
            media_validator=AcceptingMediaValidator(),
            evidence_store=evidence,
            media_urls=media_urls,
        )
    )

    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-001-entry-02"}},
    )
    settled = wait_for_capture(client, created.headers["location"])

    assert settled["status"] == "succeeded"
    assert len(reka.calls) == 2
    assert len(gemini.calls) == 1
    assert reka.calls[0].media_url is not None
    assert reka.calls[0].media_url.startswith("https://testserver/_provider-media/")
    media_token = reka.calls[0].media_url.rsplit("/", 1)[-1]
    assert media_urls.resolve(media_token) is None

    run = json.loads((tmp_path / "runs" / settled["capture_id"] / "run.json").read_text())
    assert [attempt["provider"] for attempt in run["attempts"]] == ["reka", "reka", "gemini"]
    assert run["selection"] == {
        "provider": "gemini",
        "model": "gemini-3.7-flash",
        "attempt": 1,
    }
    assert run["attempts"][0]["raw_text"] == "no"
    assert run["attempts"][2]["usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert run["attempts"][0]["timings"]["started_ms"] <= run["attempts"][0]["timings"]["completed_ms"]
    assert run["attempts"][2]["timings"]["completed_ms"] <= run["timings"]["completed_ms"]
    assert run["prompt"] == build_scene_card_prompt()
    assert run["scene_card_schema"]["additionalProperties"] is False
    assert (tmp_path / "runs" / settled["capture_id"] / "capture.mp4").exists()
    assert (tmp_path / "runs" / settled["capture_id"] / "card.json").exists()


def test_two_transport_reka_attempts_fall_through_to_one_recorded_gemini_call(
    tmp_path: Path, api_key: str, auth_headers: dict[str, str]
) -> None:
    reka = ScriptedAdapter(
        "reka",
        "reka-flash",
        [
            transport("reka", 1, "upstream 400"),
            transport("reka", 2, "upstream 400"),
        ],
    )
    gemini = ScriptedAdapter("gemini", "gemini-3.7-flash", [valid("gemini", 1)])
    media_urls = MemoryMediaUrlStore("https://testserver")
    provider = ProductionProvider(reka=reka, gemini=gemini, media_urls=media_urls)
    evidence = FileEvidenceStore(tmp_path / "runs")
    client = TestClient(
        create_app(
            api_key=api_key,
            provider=provider,
            media_validator=AcceptingMediaValidator(),
            evidence_store=evidence,
            media_urls=media_urls,
        )
    )

    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-014-exit-01"}},
    )
    settled = wait_for_capture(client, created.headers["location"])

    assert settled["status"] == "succeeded"
    assert settled["card"]["card"] == VALID_CARD_BODY
    assert len(reka.calls) == 2, "a transport failure must not grant Reka extra attempts"
    assert len(gemini.calls) == 1

    run = json.loads((tmp_path / "runs" / settled["capture_id"] / "run.json").read_text())
    assert [attempt["provider"] for attempt in run["attempts"]] == ["reka", "reka", "gemini"]
    assert [attempt["failure_kind"] for attempt in run["attempts"]] == [
        "transport",
        "transport",
        None,
    ]
    assert run["attempts"][0]["error"] == "upstream 400"
    assert run["attempts"][2]["timings"]["completed_ms"] <= run["timings"]["completed_ms"]
    assert run["selection"] == {
        "provider": "gemini",
        "model": "gemini-3.7-flash",
        "attempt": 1,
    }
    assert (tmp_path / "runs" / settled["capture_id"] / "card.json").exists()


@pytest.mark.parametrize(
    ("gemini_failure", "gemini_kind", "expected_code"),
    [
        (transport, "transport", "PROVIDER_UNAVAILABLE"),
        (timeout, "timeout", "PROVIDER_TIMEOUT"),
    ],
)
def test_timeout_reka_attempts_fall_through_once_and_gemini_failure_stays_distinct(
    tmp_path: Path,
    api_key: str,
    auth_headers: dict[str, str],
    gemini_failure,
    gemini_kind: str,
    expected_code: str,
) -> None:
    reka = ScriptedAdapter(
        "reka",
        "reka-flash",
        [timeout("reka", 1, "deadline exceeded"), timeout("reka", 2, "deadline exceeded")],
    )
    gemini = ScriptedAdapter(
        "gemini",
        "gemini-3.7-flash",
        [gemini_failure("gemini", 1, "fallback also failed")],
    )
    provider = ProductionProvider(
        reka=reka,
        gemini=gemini,
        media_urls=MemoryMediaUrlStore("https://testserver"),
    )
    client = TestClient(
        create_app(
            api_key=api_key,
            provider=provider,
            media_validator=AcceptingMediaValidator(),
            evidence_store=FileEvidenceStore(tmp_path / "runs"),
        )
    )

    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-014-exit-01"}},
    )
    settled = wait_for_capture(client, created.headers["location"])

    assert settled["status"] == "failed"
    assert len(reka.calls) == 2
    assert len(gemini.calls) == 1, "the fallback runs exactly once even after transport failures"
    assert settled["failure"]["code"] == expected_code
    assert settled["failure"]["retryable"] is True
    run = json.loads((tmp_path / "runs" / settled["capture_id"] / "run.json").read_text())
    assert [attempt["provider"] for attempt in run["attempts"]] == ["reka", "reka", "gemini"]
    assert [attempt["failure_kind"] for attempt in run["attempts"]] == [
        "timeout",
        "timeout",
        gemini_kind,
    ]
    assert run["selection"] is None


def test_overall_deadline_settles_a_wedged_provider_as_timeout(
    tmp_path: Path, api_key: str, auth_headers: dict[str, str]
) -> None:
    class WedgedProvider:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            time.sleep(0.25)
            return ProviderResult(raw_text=json.dumps(VALID_CARD_BODY), card_body=VALID_CARD_BODY)

    client = TestClient(
        create_app(
            api_key=api_key,
            provider=WedgedProvider(),
            media_validator=AcceptingMediaValidator(),
            evidence_store=FileEvidenceStore(tmp_path / "runs"),
            processing_deadline_seconds=0.02,
        )
    )
    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-027-entry-05"}},
    )

    settled = wait_for_capture(client, created.headers["location"], deadline_seconds=0.2)
    assert settled["status"] == "failed"
    assert settled["failure"]["code"] == "PROVIDER_TIMEOUT"


def test_media_validation_consumes_the_overall_processing_budget(
    tmp_path: Path, api_key: str, auth_headers: dict[str, str]
) -> None:
    class SlowMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            time.sleep(0.03)
            return True

    class ProviderThatMustNotRun:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            raise AssertionError("the overall deadline was already exhausted")

    client = TestClient(
        create_app(
            api_key=api_key,
            provider=ProviderThatMustNotRun(),
            media_validator=SlowMediaValidator(),
            evidence_store=FileEvidenceStore(tmp_path / "runs"),
            processing_deadline_seconds=0.01,
        )
    )
    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-001-entry-02"}},
    )

    settled = wait_for_capture(client, created.headers["location"], deadline_seconds=0.2)
    assert settled["status"] == "failed"
    assert settled["failure"]["code"] == "PROVIDER_TIMEOUT"


def test_short_lived_media_url_is_unguessable_and_serves_without_the_api_key(api_key: str) -> None:
    media_urls = MemoryMediaUrlStore("https://testserver", ttl_seconds=60)
    client = TestClient(create_app(api_key=api_key, media_urls=media_urls))
    url = media_urls.publish(CaptureEvidence(content=b"video", media_type="video/mp4"))
    token = url.rsplit("/", 1)[-1]

    assert len(token) >= 40
    response = client.get(f"/_provider-media/{token}")
    assert response.status_code == 200
    assert response.content == b"video"
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/_provider-media/not-a-real-token").status_code == 404


def test_prompt_contains_the_complete_canonical_schema_and_json_prefill() -> None:
    prompt = build_scene_card_prompt()
    assert '"place_type"' in prompt
    assert '"uncertainties"' in prompt
    assert '"additionalProperties":false' in prompt
    assert prompt.endswith("Assistant response begins now:\n{")


def test_reka_adapter_uses_documented_short_video_chat_shape() -> None:
    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(VALID_CARD_BODY)))],
                usage=SimpleNamespace(input_tokens=20, output_tokens=10),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    adapter = RekaChatAdapter(client=client)
    result = adapter.describe(
        CaptureEvidence(
            content=b"not-sent-inline",
            media_type="video/mp4",
            media_url="https://example.test/_provider-media/unguessable",
        ),
        build_scene_card_prompt(),
        1,
    )

    assert result.card_body == VALID_CARD_BODY
    assert calls[0]["model"] == "reka-flash"
    assert calls[0]["messages"][0]["content"][0] == {
        "type": "video_url",
        "video_url": {"url": "https://example.test/_provider-media/unguessable"},
    }
    assert len(calls[0]["messages"]) == 1, "no assistant prefill is sent"
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["name"] == "SceneCardBody"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert result.usage["input_tokens"] == 20


def test_gemini_fallback_sends_inline_video_with_response_schema() -> None:
    calls: list[dict] = []

    class Interactions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(VALID_CARD_BODY),
                usage=SimpleNamespace(input_tokens=30, output_tokens=15),
            )

    adapter = GeminiAdapter(client=SimpleNamespace(interactions=Interactions()))
    result = adapter.describe(
        CaptureEvidence(content=b"video", media_type="video/webm"),
        build_scene_card_prompt(),
        1,
    )

    assert result.card_body == VALID_CARD_BODY
    assert calls[0]["response_format"]["mime_type"] == "application/json"
    assert calls[0]["response_format"]["schema"]["additionalProperties"] is False
    assert calls[0]["input"][1]["type"] == "video"
    assert calls[0]["input"][1]["mime_type"] == "video/webm"
    assert result.usage["schema_mode"] == "response_schema"
