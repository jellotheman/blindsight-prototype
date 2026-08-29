from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blindsight.app import create_app, mount_reference_client

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"


@pytest.fixture()
def client_with_reference_client(api_key: str) -> TestClient:
    app = create_app(api_key=api_key)
    mount_reference_client(app, static_dir=STATIC_DIR)
    return TestClient(app)


def test_index_page_is_served_unauthenticated(client_with_reference_client: TestClient) -> None:
    response = client_with_reference_client.get("/")
    assert response.status_code == 200
    assert b"BlindSight" in response.content


def test_static_assets_are_served_unauthenticated(client_with_reference_client: TestClient) -> None:
    response = client_with_reference_client.get("/static/app.js")
    assert response.status_code == 200


def test_v1_routes_still_require_the_api_key_when_client_is_mounted(
    client_with_reference_client: TestClient,
) -> None:
    response = client_with_reference_client.get("/v1/excerpts")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_reference_client_never_reaches_backend_state_directly(client_with_reference_client: TestClient) -> None:
    # The client shell has no route that returns excerpt data without going through /v1 -- there
    # is no in-process shortcut for it to use.
    response = client_with_reference_client.get("/excerpts")
    assert response.status_code == 404
