from __future__ import annotations

import copy
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from blindsight.app import create_app
from blindsight.providers import CaptureEvidence, DeterministicProvider, ProviderResult
from blindsight.storage import MemoryCaptureStore, ModalCaptureStore
from tests.conftest import SchemaValidator
from tests.test_remux import (
    _ffprobe_duration,
    _ffprobe_video_codec,
    _synthetic_hevc_mp4,
    _synthetic_hevc_quicktime,
)


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


def wait_for_capture(client: TestClient, location: str, deadline_seconds: float = 3) -> dict:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        response = client.get(location, headers={"X-API-Key": "test-shared-key-0123456789"})
        if response.json()["status"] != "processing":
            return response.json()
        time.sleep(0.01)
    raise AssertionError("capture did not settle")


def test_excerpt_capture_is_pollable_from_another_app_instance(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    release_provider = threading.Event()

    class BlockingProvider:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            release_provider.wait(timeout=2)
            return ProviderResult(raw_text="valid", card_body=VALID_CARD_BODY)

    store = MemoryCaptureStore()
    provider = BlockingProvider()
    first = TestClient(create_app(api_key=api_key, store=store, provider=provider))
    second = TestClient(create_app(api_key=api_key, store=store, provider=provider))

    created = first.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-001-entry-02"}},
    )

    assert created.status_code == 201
    schema.assert_json_response("/v1/captures", "post", created)
    resource = created.json()
    assert resource["status"] == "processing"
    assert created.headers["location"] == f"/v1/captures/{resource['capture_id']}"

    processing = second.get(created.headers["location"], headers=auth_headers)
    assert processing.json()["status"] == "processing"
    assert processing.headers["retry-after"] == "1"
    schema.assert_json_response("/v1/captures/{capture_id}", "get", processing)

    release_provider.set()
    settled = wait_for_capture(second, created.headers["location"])
    assert settled["status"] == "succeeded"
    assert settled["card"]["capture_id"] == resource["capture_id"]
    assert settled["card"]["scene_session_id"] == resource["scene_session_id"]
    assert settled["card"]["card"] == VALID_CARD_BODY


def test_live_capture_opens_in_recording_state(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(
        create_app(
            api_key=api_key,
            store=MemoryCaptureStore(),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
        )
    )

    response = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    )

    assert response.status_code == 201
    schema.assert_json_response("/v1/captures", "post", response)
    assert response.json()["status"] == "recording"
    assert response.json()["source"] == {"type": "live", "mime_type": "video/webm"}


def test_live_chunks_are_indexed_idempotently_and_conflicts_are_rejected(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(
        create_app(
            api_key=api_key,
            store=MemoryCaptureStore(),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
        )
    )
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]

    for index, content in [(2, b"third"), (0, b"first"), (1, b"second")]:
        response = client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=content,
        )
        assert response.status_code == 200
        schema.assert_json_response("/v1/captures/{capture_id}/chunks/{index}", "put", response)
        assert response.json() == {
            "capture_id": capture_id,
            "index": index,
            "bytes": len(content),
            "idempotent": False,
        }

    repeated = client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"first",
    )
    assert repeated.status_code == 200
    assert repeated.json()["idempotent"] is True

    conflict = client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"different",
    )
    assert conflict.status_code == 409
    schema.assert_json_response("/v1/captures/{capture_id}/chunks/{index}", "put", conflict)
    assert conflict.json()["error"]["code"] == "CHUNK_CONFLICT"


def test_completing_live_capture_assembles_declared_chunk_order(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    class OrderCheckingProvider:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            if evidence.content == b"header-middle-tail":
                return ProviderResult(raw_text="valid", card_body=VALID_CARD_BODY)
            return ProviderResult(raw_text="wrong order", card_body=None)

    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    client = TestClient(
        create_app(
            api_key=api_key,
            store=MemoryCaptureStore(),
            provider=OrderCheckingProvider(),
            media_validator=AcceptingMediaValidator(),
        )
    )
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]
    for index, content in [(2, b"tail"), (0, b"header-"), (1, b"middle-")]:
        client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=content,
        )

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 3, "mime_type": "video/webm"},
    )

    assert completed.status_code == 202
    schema.assert_json_response("/v1/captures/{capture_id}/complete", "post", completed)
    assert completed.json()["status"] == "processing"
    assert completed.headers["location"] == f"/v1/captures/{capture_id}"
    assert completed.headers["retry-after"] == "1"
    assert wait_for_capture(client, completed.headers["location"])["status"] == "succeeded"

    late_chunk = client.put(
        f"/v1/captures/{capture_id}/chunks/3",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"late",
    )
    assert late_chunk.status_code == 409
    assert late_chunk.json()["error"]["code"] == "INVALID_STATE"


def test_missing_chunks_are_named_before_processing_starts(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(create_app(api_key=api_key))
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]
    for index in (0, 2):
        client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=f"chunk-{index}".encode(),
        )

    response = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 3, "mime_type": "video/webm"},
    )

    assert response.status_code == 409
    schema.assert_json_response("/v1/captures/{capture_id}/complete", "post", response)
    assert response.json()["error"] == {
        "code": "CAPTURE_INCOMPLETE",
        "message": "One or more declared chunks have not arrived.",
        "retryable": True,
        "details": {"missing_indices": [1]},
    }


def test_undecodable_capture_fails_without_invoking_provider(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class ExplodingProvider:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            raise AssertionError("provider must not receive undecodable media")

    class RejectingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return False

    client = TestClient(
        create_app(
            api_key=api_key,
            provider=ExplodingProvider(),
            media_validator=RejectingMediaValidator(),
        )
    )
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]
    client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"not video",
    )

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 1, "mime_type": "video/webm"},
    )
    settled = wait_for_capture(client, completed.headers["location"])

    assert settled["status"] == "failed"
    assert settled["failure"]["code"] == "CAPTURE_UNDECODABLE"


def test_capture_resource_is_shared_through_modal_dict_adapter(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class FakeModalDict:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def get(self, key: str, default: object = None) -> object:
            return self.values.get(key, default)

        def put(self, key: str, value: object, *, skip_if_exists: bool = False) -> bool:
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    remote_dict = FakeModalDict()
    store = ModalCaptureStore(remote_dict)
    provider = DeterministicProvider(card_body=VALID_CARD_BODY)
    first = TestClient(
        create_app(
            api_key=api_key,
            store=store,
            provider=provider,
            media_validator=AcceptingMediaValidator(),
        )
    )
    second = TestClient(
        create_app(
            api_key=api_key,
            store=ModalCaptureStore(remote_dict),
            provider=provider,
            media_validator=AcceptingMediaValidator(),
        )
    )

    created = first.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-014-exit-01"}},
    )
    settled = wait_for_capture(second, created.headers["location"])

    assert settled["status"] == "succeeded"


def test_capture_requests_reject_unsupported_media_and_contract_drift(
    client: TestClient, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    unsupported = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/avi"}},
    )
    assert unsupported.status_code == 415
    schema.assert_json_response("/v1/captures", "post", unsupported)
    assert unsupported.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    unsupported_complete = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/x-msvideo"}},
    )
    assert unsupported_complete.status_code == 415
    schema.assert_json_response("/v1/captures", "post", unsupported_complete)
    assert unsupported_complete.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    extra_property = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={
            "source": {"type": "live", "mime_type": "video/webm"},
            "undocumented": True,
        },
    )
    assert extra_property.status_code == 400
    schema.assert_json_response("/v1/captures", "post", extra_property)
    assert extra_property.json()["error"]["code"] == "INVALID_REQUEST"


def test_live_capture_accepts_quicktime_mime_type(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    client = TestClient(
        create_app(
            api_key=api_key,
            store=MemoryCaptureStore(),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
            media_validator=AcceptingMediaValidator(),
        )
    )

    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/quicktime"}},
    )
    assert created.status_code == 201
    schema.assert_json_response("/v1/captures", "post", created)
    assert created.json()["source"] == {"type": "live", "mime_type": "video/quicktime"}

    capture_id = created.json()["capture_id"]
    uploaded = client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"quicktime-chunk",
    )
    assert uploaded.status_code == 200

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 1, "mime_type": "video/quicktime"},
    )
    assert completed.status_code == 202
    schema.assert_json_response("/v1/captures/{capture_id}/complete", "post", completed)
    assert wait_for_capture(client, completed.headers["location"])["status"] == "succeeded"


def test_complete_mime_type_mismatch_is_rejected(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(create_app(api_key=api_key))
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/quicktime"}},
    ).json()["capture_id"]
    client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"chunk",
    )

    response = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 1, "mime_type": "video/webm"},
    )

    assert response.status_code == 400
    schema.assert_json_response("/v1/captures/{capture_id}/complete", "post", response)
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert response.json()["error"]["message"] == "Completion MIME type must match the capture."


def test_chunk_and_accumulated_capture_size_limits_use_documented_error(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(
        create_app(
            api_key=api_key,
            max_chunk_bytes=4,
            max_capture_bytes=6,
        )
    )

    def open_capture() -> str:
        return client.post(
            "/v1/captures",
            headers=auth_headers,
            json={"source": {"type": "live", "mime_type": "video/webm"}},
        ).json()["capture_id"]

    capture_id = open_capture()
    oversized_chunk = client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"12345",
    )
    assert oversized_chunk.status_code == 413
    schema.assert_json_response(
        "/v1/captures/{capture_id}/chunks/{index}", "put", oversized_chunk
    )
    assert oversized_chunk.json()["error"]["code"] == "CAPTURE_TOO_LARGE"

    capture_id = open_capture()
    first = client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"1234",
    )
    second = client.put(
        f"/v1/captures/{capture_id}/chunks/1",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"567",
    )
    assert first.status_code == 200
    assert second.status_code == 413
    assert second.json()["error"]["code"] == "CAPTURE_TOO_LARGE"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("place_type", ""),
        ("overview", "word " * 51),
        ("uncertainties", []),
        (
            "layout",
            [
                {
                    "thing": "",
                    "relationship": "beside the sink",
                    "distance": "middle",
                    "confidence": "high",
                }
            ],
        ),
        ("undocumented", True),
    ],
)
def test_invalid_provider_cards_settle_as_model_output_failure(
    api_key: str,
    auth_headers: dict[str, str],
    field: str,
    invalid_value: object,
) -> None:
    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    invalid_card = copy.deepcopy(VALID_CARD_BODY)
    invalid_card[field] = invalid_value
    client = TestClient(
        create_app(
            api_key=api_key,
            provider=DeterministicProvider(card_body=invalid_card),
            media_validator=AcceptingMediaValidator(),
        )
    )

    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-027-entry-05"}},
    )
    settled = wait_for_capture(client, created.headers["location"])

    assert settled["status"] == "failed"
    assert settled["card"] is None
    assert settled["failure"]["code"] == "MODEL_OUTPUT_INVALID"


def test_null_and_empty_collection_scene_card_meanings_remain_distinct(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    card = copy.deepcopy(VALID_CARD_BODY)
    card["layout"] = None
    card["people"] = []
    card["uncertainties"] = None
    client = TestClient(
        create_app(
            api_key=api_key,
            provider=DeterministicProvider(card_body=card),
            media_validator=AcceptingMediaValidator(),
        )
    )

    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-001-entry-02"}},
    )
    settled = wait_for_capture(client, created.headers["location"])

    assert settled["status"] == "succeeded"
    assert settled["card"]["card"]["layout"] is None
    assert settled["card"]["card"]["people"] == []
    assert settled["card"]["card"]["uncertainties"] is None


def test_malformed_provider_result_becomes_failure_instead_of_raising(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class MalformedProvider:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            return ProviderResult(raw_text='{"overview": "truncated"', card_body=None)

    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    client = TestClient(
        create_app(
            api_key=api_key,
            provider=MalformedProvider(),
            media_validator=AcceptingMediaValidator(),
        )
    )
    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-014-exit-01"}},
    )

    settled = wait_for_capture(client, created.headers["location"])
    assert settled["status"] == "failed"
    assert settled["failure"]["code"] == "MODEL_OUTPUT_INVALID"


def test_malformed_json_uses_shared_invalid_request_envelope(
    client: TestClient, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    response = client.post(
        "/v1/captures",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=b'{"source":',
    )

    assert response.status_code == 400
    schema.assert_json_response("/v1/captures", "post", response)
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_completion_and_path_validation_use_documented_bad_request(
    client: TestClient, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]
    client.put(
        f"/v1/captures/{capture_id}/chunks/0",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"capture",
    )

    extra_property = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 1, "mime_type": "video/webm", "extra": True},
    )
    assert extra_property.status_code == 400
    schema.assert_json_response(
        "/v1/captures/{capture_id}/complete", "post", extra_property
    )
    assert extra_property.json()["error"]["code"] == "INVALID_REQUEST"

    invalid_index = client.put(
        f"/v1/captures/{capture_id}/chunks/not-an-index",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"capture",
    )
    assert invalid_index.status_code == 400
    schema.assert_json_response(
        "/v1/captures/{capture_id}/chunks/{index}", "put", invalid_index
    )
    assert invalid_index.json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize("api_key_value", [None, "wrong-key"])
@pytest.mark.parametrize(
    ("method", "url", "path_template", "request_kwargs"),
    [
        (
            "POST",
            "/v1/captures",
            "/v1/captures",
            {"json": {"source": {"type": "live", "mime_type": "video/webm"}}},
        ),
        ("GET", "/v1/captures/cap_00000000", "/v1/captures/{capture_id}", {}),
        (
            "PUT",
            "/v1/captures/cap_00000000/chunks/0",
            "/v1/captures/{capture_id}/chunks/{index}",
            {"content": b"chunk", "headers": {"Content-Type": "application/octet-stream"}},
        ),
        (
            "POST",
            "/v1/captures/cap_00000000/complete",
            "/v1/captures/{capture_id}/complete",
            {"json": {"chunk_count": 1, "mime_type": "video/webm"}},
        ),
    ],
)
def test_every_capture_operation_requires_shared_api_key(
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
        ("GET", "/v1/captures/cap_missing0", "/v1/captures/{capture_id}", {}),
        (
            "PUT",
            "/v1/captures/cap_missing0/chunks/0",
            "/v1/captures/{capture_id}/chunks/{index}",
            {"content": b"chunk", "headers": {"Content-Type": "application/octet-stream"}},
        ),
        (
            "POST",
            "/v1/captures/cap_missing0/complete",
            "/v1/captures/{capture_id}/complete",
            {"json": {"chunk_count": 1, "mime_type": "video/webm"}},
        ),
    ],
)
def test_unknown_capture_operations_use_documented_not_found(
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


def test_real_bundled_video_decodes_and_completes_through_live_path(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required for real media validation")
    client = TestClient(
        create_app(
            api_key=api_key,
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
        )
    )
    content = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "excerpts"
        / "via-001-entry-02.mp4"
    ).read_bytes()
    cut_one = len(content) // 3
    cut_two = cut_one * 2
    chunks = [content[:cut_one], content[cut_one:cut_two], content[cut_two:]]
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/mp4"}},
    ).json()["capture_id"]
    for index in (2, 0, 1):
        uploaded = client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=chunks[index],
        )
        assert uploaded.status_code == 200

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 3, "mime_type": "video/mp4"},
    )
    settled = wait_for_capture(client, completed.headers["location"])

    schema.assert_valid("CaptureResource", settled)
    assert settled["status"] == "succeeded"


def test_live_capture_completes_when_streamed_webm_chunks_need_remux(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    """Reproduces the real defect: MediaRecorder-style chunks concatenate into a clip whose
    duration/seek metadata was never patched and whose VP8/VP9 bitstream Reka cannot decode.
    `ffprobe`'s codec/dimension check accepts that clip -- Reka's ingestion does not. The backend
    must transcode it to H.264 MP4 before a provider ever sees it.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required to reproduce a streamed-chunk capture")
    if subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, timeout=30
    ).stdout.count(b"libx264") == 0:
        pytest.skip("libx264 is required to reproduce the Reka-incompatible WebM transcode")

    streamed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=5",
            "-c:v",
            "libvpx",
            "-b:v",
            "150k",
            "-f",
            "webm",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    assert _ffprobe_duration(ffprobe, streamed) is None, (
        "fixture no longer reproduces the unpatched-duration defect"
    )

    class RekaLikeProvider:
        """Stands in for Reka: rejects WebM bitstreams and clips with no readable duration."""

        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            if _ffprobe_duration(ffprobe, evidence.content) is None:
                return ProviderResult(
                    raw_text="",
                    card_body=None,
                    failure_kind="transport",
                    error="Invalid video metadata None",
                )
            if evidence.media_type != "video/mp4" or _ffprobe_video_codec(
                ffprobe, evidence.content
            ) != "h264":
                return ProviderResult(
                    raw_text="",
                    card_body=None,
                    failure_kind="transport",
                    error="Expected 6 frames, got 0 None",
                )
            return ProviderResult(raw_text="valid", card_body=VALID_CARD_BODY)

    client = TestClient(create_app(api_key=api_key, provider=RekaLikeProvider()))
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]

    cut_one = len(streamed) // 3
    cut_two = cut_one * 2
    chunks = [streamed[:cut_one], streamed[cut_one:cut_two], streamed[cut_two:]]
    for index in (2, 0, 1):
        uploaded = client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=chunks[index],
        )
        assert uploaded.status_code == 200

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 3, "mime_type": "video/webm"},
    )
    settled = wait_for_capture(client, completed.headers["location"], deadline_seconds=15)

    assert settled["status"] == "succeeded"


def test_live_capture_completes_when_hevc_mp4_chunks_need_transcode(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    """Reproduces the iOS defect: an HEVC clip inside an MP4 container passes ffprobe's codec
    check -- Reka's ingestion does not, yielding zero frames exactly like VP9 WebM. The backend
    must transcode any non-H.264 codec to H.264 MP4 before a provider ever sees it.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required to reproduce a streamed-chunk capture")
    if subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, timeout=30
    ).stdout.count(b"libx265") == 0:
        pytest.skip("libx265 is required to reproduce the Reka-incompatible HEVC transcode")

    streamed = _synthetic_hevc_mp4(ffmpeg)

    class RekaLikeProvider:
        """Stands in for Reka: yields zero decoded frames from anything but H.264 MP4."""

        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            if evidence.media_type != "video/mp4" or _ffprobe_video_codec(
                ffprobe, evidence.content
            ) != "h264":
                return ProviderResult(
                    raw_text="",
                    card_body=None,
                    failure_kind="transport",
                    error="Expected 6 frames, got 0 None",
                )
            return ProviderResult(raw_text="valid", card_body=VALID_CARD_BODY)

    client = TestClient(create_app(api_key=api_key, provider=RekaLikeProvider()))
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/mp4"}},
    ).json()["capture_id"]

    cut_one = len(streamed) // 3
    cut_two = cut_one * 2
    chunks = [streamed[:cut_one], streamed[cut_one:cut_two], streamed[cut_two:]]
    for index in (2, 0, 1):
        uploaded = client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=chunks[index],
        )
        assert uploaded.status_code == 200

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 3, "mime_type": "video/mp4"},
    )
    settled = wait_for_capture(client, completed.headers["location"], deadline_seconds=15)

    assert settled["status"] == "succeeded"


def test_live_capture_completes_when_hevc_quicktime_chunks_need_transcode(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    """Reproduces the iOS defect: a native client records QuickTime/MOV chunks declared as
    `video/quicktime`. An HEVC clip inside a MOV container passes ffprobe's codec check --
    Reka's ingestion does not, yielding zero frames exactly like VP9 WebM. The backend must
    transcode it to H.264 MP4 before a provider ever sees it.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required to reproduce a streamed-chunk capture")
    if subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, timeout=30
    ).stdout.count(b"libx265") == 0:
        pytest.skip("libx265 is required to reproduce the Reka-incompatible HEVC transcode")

    streamed = _synthetic_hevc_quicktime(ffmpeg)

    class RekaLikeProvider:
        """Stands in for Reka: yields zero decoded frames from anything but H.264 MP4."""

        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            if evidence.media_type != "video/mp4" or _ffprobe_video_codec(
                ffprobe, evidence.content
            ) != "h264":
                return ProviderResult(
                    raw_text="",
                    card_body=None,
                    failure_kind="transport",
                    error="Expected 6 frames, got 0 None",
                )
            return ProviderResult(raw_text="valid", card_body=VALID_CARD_BODY)

    client = TestClient(create_app(api_key=api_key, provider=RekaLikeProvider()))
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/quicktime"}},
    ).json()["capture_id"]

    cut_one = len(streamed) // 3
    cut_two = cut_one * 2
    chunks = [streamed[:cut_one], streamed[cut_one:cut_two], streamed[cut_two:]]
    for index in (2, 0, 1):
        uploaded = client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=chunks[index],
        )
        assert uploaded.status_code == 200

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": 3, "mime_type": "video/quicktime"},
    )
    settled = wait_for_capture(client, completed.headers["location"], deadline_seconds=15)

    assert settled["status"] == "succeeded"


def test_unexpected_provider_exception_settles_as_internal_error(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class ExplodingProvider:
        def describe(self, evidence: CaptureEvidence) -> ProviderResult:
            raise RuntimeError("simulated provider bug")

    class AcceptingMediaValidator:
        def is_decodable(self, evidence: CaptureEvidence) -> bool:
            return True

    client = TestClient(
        create_app(
            api_key=api_key,
            provider=ExplodingProvider(),
            media_validator=AcceptingMediaValidator(),
        )
    )
    created = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "excerpt", "excerpt_id": "via-001-entry-02"}},
    )

    settled = wait_for_capture(client, created.headers["location"], deadline_seconds=0.5)
    assert settled["status"] == "failed"
    assert settled["failure"]["code"] == "INTERNAL_ERROR"
