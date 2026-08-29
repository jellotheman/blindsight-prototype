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


def test_primary_surface_is_the_documented_full_page_tap_target(
    client_with_reference_client: TestClient,
) -> None:
    page = client_with_reference_client.get("/")
    assert b"Tap the center of the screen to record." in page.content


def test_client_script_implements_the_documented_audio_ladder(
    client_with_reference_client: TestClient,
) -> None:
    # There is no JS test harness in this repo (see docs/spec/phase-0-1.md's HTTP-seam testing
    # decision); these markers guard the client-observable contract text and behavior from
    # accidental regression without standing up a browser.
    script = client_with_reference_client.get("/static/app.js").text

    assert "Look around at what you'd like described." in script
    assert "Still working." in script
    for function_name in (
        "playReadyEarcon",
        "playMetronomeTick",
        "playCapturedEarcon",
        "playPulseEarcon",
        "playSettledEarcon",
        "playFailureBuzz",
        "unlockSpeech",
    ):
        assert function_name in script


def test_client_script_never_invents_a_client_timings_endpoint(
    client_with_reference_client: TestClient,
) -> None:
    # docs/spec/phase-0-1.md: the client uses only documented HTTP operations and never posts
    # client timing. REFERENCE.local.md explicitly warns against reproducing the retired
    # /scan/{id}/timings route.
    script = client_with_reference_client.get("/static/app.js").text
    assert "timings" not in script.lower()


def test_client_script_uploads_chunks_and_completes_through_documented_v1_routes(
    client_with_reference_client: TestClient,
) -> None:
    script = client_with_reference_client.get("/static/app.js").text
    assert "/v1/captures" in script
    assert "/chunks/${" in script
    assert "/complete" in script
    assert '"type": "excerpt"' in script or "'type': 'excerpt'" in script or "type: \"excerpt\"" in script


def test_posters_are_still_fetched_with_the_api_key_header_not_a_bare_image_source(
    client_with_reference_client: TestClient,
) -> None:
    script = client_with_reference_client.get("/static/app.js").text
    assert "X-API-Key" in script
    assert "createObjectURL" in script
