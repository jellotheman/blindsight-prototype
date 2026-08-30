from __future__ import annotations

import copy
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from blindsight.app import create_app
from blindsight.providers import (
    CaptureEvidence,
    DeterministicCapturedViewProvider,
    DeterministicCardAnswerProvider,
    DeterministicProvider,
    QuestionAnswer,
)
from blindsight.storage import MemoryCaptureStore
from tests.conftest import SchemaValidator

VALID_CARD_BODY: dict[str, object] = {
    "place_type": "shared kitchen",
    "place_type_confidence": "high",
    "overview": (
        "The captured view showed a shared kitchen with a long counter beside two sinks and "
        "several stools facing it. No people were visible. White cabinets and bright ceiling "
        "lights made the room look evenly lit."
    ),
    "layout": [
        {
            "thing": "long counter",
            "relationship": "beside two sinks",
            "distance": "middle",
            "confidence": "high",
        }
    ],
    "open_space": "A clear aisle was visible between the counter and the cabinets.",
    "people": [],
    "visual_character": "White cabinets, steel sinks, and bright ceiling lighting were visible.",
    "uncertainties": None,
}


def wait_for(client: TestClient, location: str, headers: dict[str, str], deadline_seconds: float = 3) -> dict:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        response = client.get(location, headers=headers)
        if response.json()["status"] != "processing":
            return response.json()
        time.sleep(0.01)
    raise AssertionError("resource did not settle")


def open_scene_session(
    client: TestClient, auth_headers: dict[str, str], deadline_seconds: float = 3
) -> dict:
    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-001-entry-02"}},
    )
    return wait_for(client, created.headers["location"], auth_headers, deadline_seconds)


def make_client(
    api_key: str,
    *,
    card_provider=None,
    captured_view_provider=None,
    question_processing_deadline_seconds: float = 60.0,
) -> TestClient:
    return TestClient(
        create_app(
            api_key=api_key,
            store=MemoryCaptureStore(),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
            card_provider=card_provider,
            captured_view_provider=captured_view_provider,
            question_processing_deadline_seconds=question_processing_deadline_seconds,
        )
    )


def test_card_grounded_question_is_answered_without_a_clip_check(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = make_client(
        api_key, card_provider=DeterministicCardAnswerProvider(answer="White cabinets.")
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]

    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What colour were the cabinets?"},
    )
    assert created.status_code == 202
    schema.assert_json_response(
        "/v1/scene-sessions/{scene_session_id}/questions", "post", created
    )
    assert created.json()["status"] == "processing"
    assert created.headers["location"] == (
        f"/v1/scene-sessions/{session_id}/questions/{created.json()['question_id']}"
    )

    settled = wait_for(client, created.headers["location"], auth_headers)
    schema.assert_valid("QuestionResource", settled)
    assert settled["status"] == "answered"
    assert settled["answer"] == "White cabinets."
    assert settled["source"] == "scene_card"
    assert settled["failure"] is None


def test_card_miss_enters_needs_clip_consent_without_invoking_video_provider(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    class ExplodingCapturedViewProvider:
        def answer(self, evidence, conversation):
            raise AssertionError("the video provider must not run before explicit consent")

    client = make_client(
        api_key,
        card_provider=DeterministicCardAnswerProvider(answer=None),
        captured_view_provider=ExplodingCapturedViewProvider(),
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]

    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What colour was the mug on the counter?"},
    )
    settled = wait_for(client, created.headers["location"], auth_headers)

    assert settled["status"] == "needs_clip_consent"
    assert settled["answer"] is None
    assert settled["source"] is None
    assert settled["failure"] is None


def test_clip_check_is_valid_only_from_needs_clip_consent(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = make_client(
        api_key, card_provider=DeterministicCardAnswerProvider(answer="An answer.")
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "Anything?"},
    )
    settled = wait_for(client, created.headers["location"], auth_headers)
    assert settled["status"] == "answered"

    conflict = client.post(
        f"{created.headers['location']}/clip-check", headers=auth_headers
    )
    assert conflict.status_code == 409
    schema.assert_json_response(
        "/v1/scene-sessions/{scene_session_id}/questions/{question_id}/clip-check",
        "post",
        conflict,
    )
    assert conflict.json()["error"]["code"] == "INVALID_STATE"


def test_clip_check_answers_from_the_captured_view_after_consent(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = make_client(
        api_key,
        card_provider=DeterministicCardAnswerProvider(answer=None),
        captured_view_provider=DeterministicCapturedViewProvider(answer="A blue mug."),
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What colour was the mug?"},
    )
    needs_consent = wait_for(client, created.headers["location"], auth_headers)
    assert needs_consent["status"] == "needs_clip_consent"

    clip_check = client.post(f"{created.headers['location']}/clip-check", headers=auth_headers)
    assert clip_check.status_code == 202
    schema.assert_json_response(
        "/v1/scene-sessions/{scene_session_id}/questions/{question_id}/clip-check",
        "post",
        clip_check,
    )
    assert clip_check.json()["status"] == "processing"

    settled = wait_for(client, created.headers["location"], auth_headers)
    assert settled["status"] == "answered"
    assert settled["answer"] == "A blue mug."
    assert settled["source"] == "captured_view"


def test_second_miss_is_unanswerable_and_distinguishable_from_a_provider_failure(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    client = make_client(
        api_key,
        card_provider=DeterministicCardAnswerProvider(answer=None),
        captured_view_provider=DeterministicCapturedViewProvider(answer=None),
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What colour was the mug?"},
    )
    wait_for(client, created.headers["location"], auth_headers)

    clip_check = client.post(f"{created.headers['location']}/clip-check", headers=auth_headers)
    settled = wait_for(client, clip_check.headers["location"], auth_headers)

    assert settled["status"] == "unanswerable"
    assert settled["answer"] is None
    assert settled["source"] == "captured_view"
    assert settled["failure"] is None


def test_captured_view_provider_failure_settles_as_a_distinct_failed_state(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class UnavailableCapturedViewProvider:
        def answer(self, evidence, conversation):
            return QuestionAnswer(answer=None, failure_kind="transport", error="boom")

    client = make_client(
        api_key,
        card_provider=DeterministicCardAnswerProvider(answer=None),
        captured_view_provider=UnavailableCapturedViewProvider(),
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What colour was the mug?"},
    )
    wait_for(client, created.headers["location"], auth_headers)
    clip_check = client.post(f"{created.headers['location']}/clip-check", headers=auth_headers)
    settled = wait_for(client, clip_check.headers["location"], auth_headers)

    assert settled["status"] == "failed"
    assert settled["answer"] is None
    assert settled["failure"]["code"] == "PROVIDER_UNAVAILABLE"
    assert settled["failure"]["retryable"] is True


def test_conversation_context_accumulates_within_a_session(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    seen_conversations: list[list[dict[str, str]]] = []

    class SpyCardAnswerProvider:
        def answer(self, card_body, conversation):
            seen_conversations.append(copy.deepcopy(conversation))
            return QuestionAnswer(answer=f"answer-{len(seen_conversations)}")

    client = make_client(api_key, card_provider=SpyCardAnswerProvider())
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]

    first = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What is on the counter?"},
    )
    wait_for(client, first.headers["location"], auth_headers)

    second = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "What colour is it?"},
    )
    wait_for(client, second.headers["location"], auth_headers)

    assert seen_conversations[0] == [{"role": "user", "content": "What is on the counter?"}]
    assert seen_conversations[1] == [
        {"role": "user", "content": "What is on the counter?"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "What colour is it?"},
    ]


def test_new_capture_starts_a_fresh_session_with_no_inherited_conversation(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    seen_conversations: list[list[dict[str, str]]] = []

    class SpyCardAnswerProvider:
        def answer(self, card_body, conversation):
            seen_conversations.append(copy.deepcopy(conversation))
            return QuestionAnswer(answer="ok")

    client = make_client(api_key, card_provider=SpyCardAnswerProvider())
    first_session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    asked = client.post(
        f"/v1/scene-sessions/{first_session_id}/questions",
        headers=auth_headers,
        json={"question": "What is on the counter?"},
    )
    wait_for(client, asked.headers["location"], auth_headers)

    second_session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    assert second_session_id != first_session_id
    asked_again = client.post(
        f"/v1/scene-sessions/{second_session_id}/questions",
        headers=auth_headers,
        json={"question": "What is on the counter?"},
    )
    wait_for(client, asked_again.headers["location"], auth_headers)

    assert seen_conversations[-1] == [{"role": "user", "content": "What is on the counter?"}]


def test_deleting_a_session_ends_the_conversation_but_leaves_the_capture_readable(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    client = make_client(api_key, card_provider=DeterministicCardAnswerProvider(answer="ok"))
    capture = open_scene_session(client, auth_headers)
    session_id = capture["scene_session_id"]
    capture_id = capture["capture_id"]
    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "Anything?"},
    )
    wait_for(client, created.headers["location"], auth_headers)

    deleted = client.delete(f"/v1/scene-sessions/{session_id}", headers=auth_headers)
    assert deleted.status_code == 204

    later_question = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "Anything else?"},
    )
    assert later_question.status_code == 404
    assert later_question.json()["error"]["code"] == "NOT_FOUND"

    later_poll = client.get(created.headers["location"], headers=auth_headers)
    assert later_poll.status_code == 404

    still_readable = client.get(f"/v1/captures/{capture_id}", headers=auth_headers)
    assert still_readable.status_code == 200
    assert still_readable.json()["card"] is not None


def test_question_without_a_scene_card_yet_is_a_conflict(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    client = TestClient(
        create_app(
            api_key=api_key,
            store=MemoryCaptureStore(),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
        )
    )
    opened = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    )
    session_id = opened.json()["scene_session_id"]

    response = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "Anything?"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_STATE"


def test_question_processing_deadline_settles_as_a_timeout(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class WedgedCardAnswerProvider:
        def answer(self, card_body, conversation):
            time.sleep(1)
            return QuestionAnswer(answer="too late")

    client = make_client(
        api_key,
        card_provider=WedgedCardAnswerProvider(),
        question_processing_deadline_seconds=0.05,
    )
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]
    created = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "Anything?"},
    )
    settled = wait_for(client, created.headers["location"], auth_headers, deadline_seconds=3)

    assert settled["status"] == "failed"
    assert settled["failure"]["code"] == "PROVIDER_TIMEOUT"
    assert settled["failure"]["retryable"] is True


def test_question_request_validation_uses_the_shared_error_envelope(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = make_client(api_key)
    session_id = open_scene_session(client, auth_headers)["scene_session_id"]

    missing = client.post(
        f"/v1/scene-sessions/{session_id}/questions", headers=auth_headers, json={}
    )
    assert missing.status_code == 400
    schema.assert_json_response(
        "/v1/scene-sessions/{scene_session_id}/questions", "post", missing
    )
    assert missing.json()["error"]["code"] == "INVALID_REQUEST"

    too_long = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "x" * 1001},
    )
    assert too_long.status_code == 400
    assert too_long.json()["error"]["code"] == "INVALID_REQUEST"

    extra_property = client.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "ok", "extra": True},
    )
    assert extra_property.status_code == 400
    assert extra_property.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("api_key_value", [None, "wrong-key"])
@pytest.mark.parametrize(
    ("method", "url", "path_template", "request_kwargs"),
    [
        (
            "POST",
            "/v1/scene-sessions/ses_00000000/questions",
            "/v1/scene-sessions/{scene_session_id}/questions",
            {"json": {"question": "Anything?"}},
        ),
        (
            "GET",
            "/v1/scene-sessions/ses_00000000/questions/que_00000000",
            "/v1/scene-sessions/{scene_session_id}/questions/{question_id}",
            {},
        ),
        (
            "POST",
            "/v1/scene-sessions/ses_00000000/questions/que_00000000/clip-check",
            "/v1/scene-sessions/{scene_session_id}/questions/{question_id}/clip-check",
            {},
        ),
        (
            "DELETE",
            "/v1/scene-sessions/ses_00000000",
            "/v1/scene-sessions/{scene_session_id}",
            {},
        ),
    ],
)
def test_every_question_operation_requires_shared_api_key(
    client: TestClient,
    schema: SchemaValidator,
    api_key_value: str | None,
    method: str,
    url: str,
    path_template: str,
    request_kwargs: dict[str, Any],
) -> None:
    kwargs = copy.deepcopy(request_kwargs)
    headers = kwargs.setdefault("headers", {})
    assert isinstance(headers, dict)
    if api_key_value is not None:
        headers["X-API-Key"] = api_key_value

    response = client.request(method, url, **kwargs)

    assert response.status_code == 401
    schema.assert_json_response(path_template, method, response)
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize(
    ("method", "url", "path_template", "request_kwargs"),
    [
        (
            "POST",
            "/v1/scene-sessions/ses_missing0/questions",
            "/v1/scene-sessions/{scene_session_id}/questions",
            {"json": {"question": "Anything?"}},
        ),
        (
            "GET",
            "/v1/scene-sessions/ses_missing0/questions/que_missing0",
            "/v1/scene-sessions/{scene_session_id}/questions/{question_id}",
            {},
        ),
        (
            "POST",
            "/v1/scene-sessions/ses_missing0/questions/que_missing0/clip-check",
            "/v1/scene-sessions/{scene_session_id}/questions/{question_id}/clip-check",
            {},
        ),
        (
            "DELETE",
            "/v1/scene-sessions/ses_missing0",
            "/v1/scene-sessions/{scene_session_id}",
            {},
        ),
    ],
)
def test_unknown_question_operations_use_documented_not_found(
    client: TestClient,
    auth_headers: dict[str, str],
    schema: SchemaValidator,
    method: str,
    url: str,
    path_template: str,
    request_kwargs: dict[str, Any],
) -> None:
    kwargs = copy.deepcopy(request_kwargs)
    headers = kwargs.setdefault("headers", {})
    assert isinstance(headers, dict)
    headers.update(auth_headers)

    response = client.request(method, url, **kwargs)

    assert response.status_code == 404
    schema.assert_json_response(path_template, method, response)
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_question_resource_settles_across_app_instances_sharing_one_store(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    store = MemoryCaptureStore()
    card_provider = DeterministicCardAnswerProvider(answer="Shared across instances.")
    first = TestClient(
        create_app(
            api_key=api_key,
            store=store,
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
            card_provider=card_provider,
        )
    )
    second = TestClient(
        create_app(
            api_key=api_key,
            store=store,
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
            card_provider=card_provider,
        )
    )

    session_id = open_scene_session(first, auth_headers)["scene_session_id"]
    created = first.post(
        f"/v1/scene-sessions/{session_id}/questions",
        headers=auth_headers,
        json={"question": "Anything?"},
    )
    settled = wait_for(second, created.headers["location"], auth_headers)

    assert settled["status"] == "answered"
    assert settled["answer"] == "Shared across instances."
