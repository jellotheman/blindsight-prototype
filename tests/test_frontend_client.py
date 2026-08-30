from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blindsight.app import create_app, mount_frontend_client, mount_reference_client

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_STATIC_DIR = REPO_ROOT / "static"


@pytest.fixture()
def client_with_both_clients(tmp_path: Path, api_key: str) -> TestClient:
    frontend_dist = tmp_path / "dist"
    asset_dir = frontend_dist / "_expo"
    asset_dir.mkdir(parents=True)
    (frontend_dist / "index.html").write_text(
        "<!doctype html><title>Expo BlindSight</title><main>New Expo client</main>",
        encoding="utf-8",
    )
    (frontend_dist / "orient.html").write_text(
        "<!doctype html><title>Orient</title><main>Orient route</main>",
        encoding="utf-8",
    )
    (asset_dir / "bundle.js").write_text("console.log('expo');", encoding="utf-8")

    app = create_app(api_key=api_key)
    mount_reference_client(app, static_dir=REFERENCE_STATIC_DIR)
    mount_frontend_client(app, frontend_dist_dir=frontend_dist)
    return TestClient(app)


def test_expo_client_is_served_at_the_root(client_with_both_clients: TestClient) -> None:
    response = client_with_both_clients.get("/")
    assert response.status_code == 200
    assert "New Expo client" in response.text


def test_expo_assets_are_served_unauthenticated(client_with_both_clients: TestClient) -> None:
    response = client_with_both_clients.get("/_expo/bundle.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_expo_static_routes_support_direct_navigation(client_with_both_clients: TestClient) -> None:
    response = client_with_both_clients.get("/orient")
    assert response.status_code == 200
    assert "Orient route" in response.text


def test_reference_client_remains_available(client_with_both_clients: TestClient) -> None:
    response = client_with_both_clients.get("/reference/")
    assert response.status_code == 200
    assert "BlindSight" in response.text


def test_api_route_precedes_the_root_static_mount(client_with_both_clients: TestClient) -> None:
    response = client_with_both_clients.get("/v1/excerpts")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_mount_fails_loudly_when_the_export_is_missing(tmp_path: Path, api_key: str) -> None:
    app = create_app(api_key=api_key)
    with pytest.raises(FileNotFoundError, match="npm run build:web"):
        mount_frontend_client(app, frontend_dist_dir=tmp_path / "missing")
