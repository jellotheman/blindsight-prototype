"""Live chunk upload must cost the same whatever else the shared store already holds.

These drive the real `ModalCaptureStore` through the public HTTP interface against a Dict double
that fails loudly on a whole-Dict scan and accounts for the bytes each operation transfers. A
scan-based byte accounting made every `PUT .../chunks/{index}` download every value in the shared
Dict, so upload cost grew with the deployment's history until requests stopped completing.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from blindsight.app import create_app
from blindsight.providers import CaptureEvidence, DeterministicProvider
from blindsight.storage import ModalCaptureStore

from tests.test_captures import VALID_CARD_BODY, wait_for_capture

CHUNK = b"0123456789abcdef" * 64
CHUNK_COUNT = 8


class ScanRefusingDict:
    """A `modal.Dict` double whose bulk readers raise, and that measures per-call transfer."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.transferred_bytes = 0

    def get(self, key: str, default: Any = None) -> Any:
        value = self.values.get(key, default)
        self.transferred_bytes += _size(value)
        return value

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.transferred_bytes += _size(value)
        self.values[key] = value
        return True

    def pop(self, key: str, default: Any = None) -> Any:
        return self.values.pop(key, default)

    def items(self) -> Any:
        raise AssertionError("the shared Dict must never be scanned on a request path")

    def keys(self) -> Any:
        raise AssertionError("the shared Dict must never be scanned on a request path")

    def values_(self) -> Any:  # pragma: no cover - defensive
        raise AssertionError("the shared Dict must never be scanned on a request path")

    def prefix_keys(self, prefix: str) -> list[str]:
        return [key for key in self.values if key.startswith(prefix)]


def _size(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, tuple):
        return sum(_size(item) for item in value)
    return 0


class AcceptingMediaValidator:
    def is_decodable(self, evidence: CaptureEvidence) -> bool:
        return True


def _client(api_key: str, remote_dict: ScanRefusingDict) -> TestClient:
    return TestClient(
        create_app(
            api_key=api_key,
            store=ModalCaptureStore(remote_dict),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
            media_validator=AcceptingMediaValidator(),
        )
    )


def _record_live_capture(
    client: TestClient, auth_headers: dict[str, str], remote_dict: ScanRefusingDict
) -> tuple[str, int]:
    """Run one whole live capture and return its id and the bytes the chunk uploads moved."""
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]

    before = remote_dict.transferred_bytes
    for index in range(CHUNK_COUNT):
        uploaded = client.put(
            f"/v1/captures/{capture_id}/chunks/{index}",
            headers={**auth_headers, "Content-Type": "application/octet-stream"},
            content=CHUNK,
        )
        assert uploaded.status_code == 200, uploaded.text
    upload_bytes = remote_dict.transferred_bytes - before

    completed = client.post(
        f"/v1/captures/{capture_id}/complete",
        headers=auth_headers,
        json={"chunk_count": CHUNK_COUNT, "mime_type": "video/webm"},
    )
    assert completed.status_code == 202, completed.text
    settled = wait_for_capture(client, completed.headers["location"])
    assert settled["status"] == "succeeded", settled
    return capture_id, upload_bytes


def test_chunk_upload_cost_does_not_grow_with_the_shared_store(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    remote_dict = ScanRefusingDict()
    client = _client(api_key, remote_dict)

    _, first_upload_bytes = _record_live_capture(client, auth_headers, remote_dict)
    for _ in range(4):
        _record_live_capture(client, auth_headers, remote_dict)
    _, last_upload_bytes = _record_live_capture(client, auth_headers, remote_dict)

    # Uploading the same eight chunks costs the same after five earlier captures as it did on an
    # empty store. A scan-based accounting would have made the last capture many times dearer.
    assert last_upload_bytes == first_upload_bytes
    assert first_upload_bytes < 2 * CHUNK_COUNT * len(CHUNK)


def test_assembled_chunks_are_dropped_from_the_shared_store(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    remote_dict = ScanRefusingDict()
    client = _client(api_key, remote_dict)

    capture_id, _ = _record_live_capture(client, auth_headers, remote_dict)

    assert remote_dict.prefix_keys(f"chunk:{capture_id}:") == []
    assert remote_dict.prefix_keys(f"chunksizes:{capture_id}") == []
    assert remote_dict.prefix_keys("chunk:") == []
    # The scene card and the clip the Stage 1 captured-view check needs both survive.
    assert remote_dict.prefix_keys(f"capture:{capture_id}") != []
    assert remote_dict.prefix_keys(f"media:{capture_id}") != []


def test_accumulated_chunk_limit_still_rejects_an_oversized_capture(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    remote_dict = ScanRefusingDict()
    client = TestClient(
        create_app(
            api_key=api_key,
            store=ModalCaptureStore(remote_dict),
            provider=DeterministicProvider(card_body=VALID_CARD_BODY),
            media_validator=AcceptingMediaValidator(),
            max_chunk_bytes=4,
            max_capture_bytes=6,
        )
    )
    capture_id = client.post(
        "/v1/captures",
        headers=auth_headers,
        json={"source": {"type": "live", "mime_type": "video/webm"}},
    ).json()["capture_id"]

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
    # The rejected chunk leaves nothing behind, so a retry within the limit still succeeds.
    assert remote_dict.prefix_keys(f"chunk:{capture_id}:1") == []
    retry = client.put(
        f"/v1/captures/{capture_id}/chunks/1",
        headers={**auth_headers, "Content-Type": "application/octet-stream"},
        content=b"56",
    )
    assert retry.status_code == 200


def test_deleting_a_scene_session_releases_its_clip_and_questions(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    remote_dict = ScanRefusingDict()
    client = _client(api_key, remote_dict)

    capture_id, _ = _record_live_capture(client, auth_headers, remote_dict)
    capture = client.get(f"/v1/captures/{capture_id}", headers=auth_headers).json()
    scene_session_id = capture["scene_session_id"]

    asked = client.post(
        f"/v1/scene-sessions/{scene_session_id}/questions",
        headers=auth_headers,
        json={"question": "Where is the counter?"},
    )
    assert asked.status_code == 202
    question_id = asked.json()["question_id"]
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        polled = client.get(asked.headers["location"], headers=auth_headers)
        if polled.json()["status"] != "processing":
            break
        time.sleep(0.01)

    deleted = client.delete(f"/v1/scene-sessions/{scene_session_id}", headers=auth_headers)

    assert deleted.status_code == 204
    assert remote_dict.prefix_keys(f"media:{capture_id}") == []
    assert remote_dict.prefix_keys(f"question:{question_id}") == []
    # The capture resource stays readable so a client can re-read the card it already spoke.
    assert client.get(f"/v1/captures/{capture_id}", headers=auth_headers).status_code == 200
