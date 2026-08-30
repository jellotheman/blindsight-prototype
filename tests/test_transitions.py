"""Stage 3 transition sessions, verified only through the public HTTP contract."""

from __future__ import annotations

import time
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from blindsight.app import create_app
from blindsight.transitions import (
    InMemoryTransitionAdapter,
    ModalTransitionAdapter,
    ModalTransitionSessionStore,
    TransitionObservation,
)

from tests.conftest import SchemaValidator


class SharedModalDict:
    """Small thread-safe Modal Dict double for cross-container HTTP tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Any] = {}

    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool:
        with self._lock:
            if skip_if_exists and key in self._items:
                return False
            self._items[key] = value
            return True

    def get(self, key: str) -> Any:
        with self._lock:
            return self._items.get(key)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._items.pop(key, default)


def wait_for_session(
    client: TestClient, location: str, headers: dict[str, str], *, cursor: str | None = None
) -> dict[str, Any]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        params = {"cursor": cursor} if cursor is not None else None
        response = client.get(location, headers=headers, params=params)
        assert response.status_code == 200
        resource = response.json()
        if resource["status"] != "starting":
            return resource
        time.sleep(0.01)
    raise AssertionError("Transition session did not leave starting state.")


def create_session(
    client: TestClient, auth_headers: dict[str, str], schema: SchemaValidator
) -> tuple[str, dict[str, Any]]:
    created = client.post("/v1/transition-sessions", headers=auth_headers)
    assert created.status_code == 201
    schema.assert_json_response("/v1/transition-sessions", "post", created)
    assert created.headers["location"] == (
        f"/v1/transition-sessions/{created.json()['transition_session_id']}"
    )
    return created.headers["location"], created.json()


def test_client_can_create_poll_and_delete_a_transition_session_from_the_contract(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(create_app(api_key=api_key))
    location, created = create_session(client, auth_headers, schema)
    assert created["status"] == "starting"
    assert created["events"] == []
    assert created["next_event_cursor"] is None

    active = wait_for_session(client, location, auth_headers)
    assert active["status"] == "active"
    schema.assert_json_response(
        "/v1/transition-sessions/{transition_session_id}", "get", client.get(location, headers=auth_headers)
    )

    deleted = client.delete(location, headers=auth_headers)
    assert deleted.status_code == 204


def test_chunk_retries_and_out_of_order_delivery_preserve_the_continuous_sequence(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    adapter = InMemoryTransitionAdapter(
        observations_by_index={1: [TransitionObservation(observed_at="2026-08-30T06:30:00Z")]}
    )
    client = TestClient(create_app(api_key=api_key, transition_adapter=adapter))
    location, session = create_session(client, auth_headers, schema)
    wait_for_session(client, location, auth_headers)

    chunk_headers = {**auth_headers, "Content-Type": "video/webm"}
    delayed = client.put(f"{location}/chunks/1", headers=chunk_headers, content=b"second")
    assert delayed.status_code == 200
    schema.assert_json_response(
        "/v1/transition-sessions/{transition_session_id}/chunks/{index}", "put", delayed
    )
    # Chunk 1 is retained but cannot produce an externally visible event until the missing
    # contiguous prefix arrives.
    assert client.get(location, headers=auth_headers).json()["events"] == []
    first = client.put(f"{location}/chunks/0", headers=chunk_headers, content=b"first")
    assert first.status_code == 200
    retry = client.put(f"{location}/chunks/0", headers=chunk_headers, content=b"first")
    assert retry.status_code == 200
    assert retry.json()["idempotent"] is True
    conflict = client.put(f"{location}/chunks/0", headers=chunk_headers, content=b"different")
    assert conflict.status_code == 409
    schema.assert_json_response(
        "/v1/transition-sessions/{transition_session_id}/chunks/{index}", "put", conflict
    )
    assert conflict.json()["error"]["code"] == "CHUNK_CONFLICT"

    deadline = time.monotonic() + 3
    resource: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resource = client.get(location, headers=auth_headers).json()
        if resource["events"]:
            break
        time.sleep(0.01)
    assert len(resource["events"]) == 1
    event = resource["events"][0]
    assert event["observed_at"] == "2026-08-30T06:30:00Z"
    assert event["transition_event_id"].startswith("tev_")
    assert resource["next_event_cursor"] == event["transition_event_id"]

    repeated_cursor = client.get(location, headers=auth_headers)
    assert repeated_cursor.json()["events"] == resource["events"]
    later = client.get(location, headers=auth_headers, params={"cursor": event["transition_event_id"]})
    assert later.status_code == 200
    assert later.json()["events"] == []
    assert later.json()["next_event_cursor"] == event["transition_event_id"]
    assert session["transition_session_id"] == resource["transition_session_id"]


def test_invalid_and_missing_transition_requests_use_the_documented_envelopes(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    client = TestClient(create_app(api_key=api_key))

    missing = client.get("/v1/transition-sessions/trs_missing0", headers=auth_headers)
    assert missing.status_code == 404
    schema.assert_json_response("/v1/transition-sessions/{transition_session_id}", "get", missing)
    assert missing.json()["error"]["code"] == "NOT_FOUND"

    location, _ = create_session(client, auth_headers, schema)
    invalid = client.put(
        f"{location}/chunks/0",
        headers={**auth_headers, "Content-Type": "video/webm"},
        content=b"",
    )
    assert invalid.status_code == 400
    schema.assert_json_response(
        "/v1/transition-sessions/{transition_session_id}/chunks/{index}", "put", invalid
    )
    assert invalid.json()["error"]["code"] == "INVALID_REQUEST"


def test_transition_failures_settle_to_a_stable_terminal_resource(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    adapter = InMemoryTransitionAdapter(fail_on_start=RuntimeError("unavailable"))
    client = TestClient(create_app(api_key=api_key, transition_adapter=adapter))
    location, _ = create_session(client, auth_headers, schema)

    failed = wait_for_session(client, location, auth_headers)
    assert failed["status"] == "failed"
    assert failed["failure"]["code"] == "TRANSITION_ADAPTER_FAILED"
    response = client.get(location, headers=auth_headers)
    schema.assert_json_response("/v1/transition-sessions/{transition_session_id}", "get", response)


def test_deletion_stops_the_adapter_once_and_prevents_later_ingestion(
    api_key: str, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    adapter = InMemoryTransitionAdapter()
    client = TestClient(create_app(api_key=api_key, transition_adapter=adapter))
    location, session = create_session(client, auth_headers, schema)
    wait_for_session(client, location, auth_headers)

    assert client.delete(location, headers=auth_headers).status_code == 204
    assert adapter.stopped_session_ids == [session["transition_session_id"]]
    rejected = client.put(
        f"{location}/chunks/0",
        headers={**auth_headers, "Content-Type": "video/webm"},
        content=b"later",
    )
    assert rejected.status_code == 404
    schema.assert_json_response(
        "/v1/transition-sessions/{transition_session_id}/chunks/{index}", "put", rejected
    )
    assert rejected.json()["error"]["code"] == "NOT_FOUND"


def test_modal_backed_sessions_coordinate_queue_limits_and_deletion_across_apps(
    api_key: str, auth_headers: dict[str, str]
) -> None:
    class DelayedStartAdapter(InMemoryTransitionAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def start(self, transition_session_id: str) -> None:
            self.started.set()
            assert self.release.wait(timeout=3)

    remote_dict = SharedModalDict()
    adapter = DelayedStartAdapter()
    first = TestClient(
        create_app(
            api_key=api_key,
            transition_store=ModalTransitionSessionStore(remote_dict),
            transition_adapter=adapter,
            max_transition_queued_bytes=5,
        )
    )
    second = TestClient(
        create_app(
            api_key=api_key,
            transition_store=ModalTransitionSessionStore(remote_dict),
            transition_adapter=adapter,
            max_transition_queued_bytes=5,
        )
    )
    created = first.post("/v1/transition-sessions", headers=auth_headers)
    location = created.headers["location"]
    assert adapter.started.wait(timeout=3)

    barrier = threading.Barrier(2)
    responses: list[Any] = []

    def upload(client: TestClient, index: int) -> None:
        barrier.wait()
        responses.append(
            client.put(
                f"{location}/chunks/{index}",
                headers={**auth_headers, "Content-Type": "video/webm"},
                content=b"1234",
            )
        )

    first_upload = threading.Thread(target=upload, args=(first, 0))
    second_upload = threading.Thread(target=upload, args=(second, 1))
    first_upload.start()
    second_upload.start()
    first_upload.join(timeout=3)
    second_upload.join(timeout=3)
    assert sorted(response.status_code for response in responses) == [200, 413]

    adapter.release.set()
    wait_for_session(first, location, auth_headers)

    delete_barrier = threading.Barrier(2)
    delete_responses: list[Any] = []

    def delete(client: TestClient) -> None:
        delete_barrier.wait()
        delete_responses.append(client.delete(location, headers=auth_headers))

    first_delete = threading.Thread(target=delete, args=(first,))
    second_delete = threading.Thread(target=delete, args=(second,))
    first_delete.start()
    second_delete.start()
    first_delete.join(timeout=3)
    second_delete.join(timeout=3)
    assert sorted(response.status_code for response in delete_responses) == [204, 404]
    assert adapter.stopped_session_ids == [created.json()["transition_session_id"]]


@pytest.mark.parametrize("api_key_value", [None, "wrong-key"])
@pytest.mark.parametrize(
    ("method", "url", "path_template", "request_kwargs"),
    [
        ("POST", "/v1/transition-sessions", "/v1/transition-sessions", {}),
        (
            "PUT",
            "/v1/transition-sessions/trs_00000000/chunks/0",
            "/v1/transition-sessions/{transition_session_id}/chunks/{index}",
            {"headers": {"Content-Type": "video/webm"}, "content": b"chunk"},
        ),
        (
            "GET",
            "/v1/transition-sessions/trs_00000000",
            "/v1/transition-sessions/{transition_session_id}",
            {},
        ),
        (
            "DELETE",
            "/v1/transition-sessions/trs_00000000",
            "/v1/transition-sessions/{transition_session_id}",
            {},
        ),
    ],
)
def test_every_transition_operation_requires_the_shared_api_key(
    client: TestClient,
    schema: SchemaValidator,
    api_key_value: str | None,
    method: str,
    url: str,
    path_template: str,
    request_kwargs: dict[str, Any],
) -> None:
    kwargs = dict(request_kwargs)
    headers = dict(kwargs.pop("headers", {}))
    if api_key_value is not None:
        headers["X-API-Key"] = api_key_value

    response = client.request(method, url, headers=headers, **kwargs)
    assert response.status_code == 401
    schema.assert_json_response(path_template, method, response)
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_in_memory_and_modal_adapters_share_the_transition_adapter_interface() -> None:
    class FakeModalWorker:
        def start(self, transition_session_id: str) -> None:
            return None

        def process(
            self, transition_session_id: str, index: int, content: bytes, media_type: str
        ) -> list[TransitionObservation]:
            return []

        def stop(self, transition_session_id: str) -> None:
            return None

    for adapter in (InMemoryTransitionAdapter(), ModalTransitionAdapter(FakeModalWorker())):
        adapter.start("trs_00000000")
        assert adapter.process("trs_00000000", 0, b"chunk", "video/webm") == []
        adapter.stop("trs_00000000")
