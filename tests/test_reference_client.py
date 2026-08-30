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


# --- Stage 1: follow-up questions and the consent offer ---------------------------------------
# Same guard style and the same reason as the audio-ladder test above: there is no JS harness in
# this repo, so these pin the client-observable Stage 1 contract -- the documented routes, the
# mandated spoken lines, and the structural fact that the clip check hangs off consent -- without
# standing up a browser. The behavior they guard is specified in docs/spec/phase-0-1.md
# ("Stage 1 follow-up") and docs/spec/openapi.yaml.


def _client_script(client: TestClient) -> str:
    return client.get("/static/app.js").text


def _function_body(script: str, name: str) -> str:
    start = script.index(f"function {name}(")
    rest = script[start + 1 :]
    ends = [i for i in (rest.find("\nfunction "), rest.find("\nasync function ")) if i != -1]
    return rest[: min(ends)] if ends else rest


def test_page_offers_a_follow_up_question_control_and_a_done_control(
    client_with_reference_client: TestClient,
) -> None:
    page = client_with_reference_client.get("/").text
    assert 'id="question"' in page
    assert 'id="ask"' in page
    assert 'id="done"' in page


def test_client_asks_follow_up_questions_against_the_active_scene_session(
    client_with_reference_client: TestClient,
) -> None:
    script = _client_script(client_with_reference_client)
    assert "/v1/scene-sessions/${" in script
    assert "/questions" in script
    assert "scene_session_id" in script


def test_client_speaks_a_consent_offer_with_a_wait_warning(
    client_with_reference_client: TestClient,
) -> None:
    script = _client_script(client_with_reference_client)
    assert "needs_clip_consent" in script
    assert "several seconds" in script


def test_clip_check_is_requested_only_from_the_consent_agreement_path(
    client_with_reference_client: TestClient,
) -> None:
    # Only the explicit captured-view-check operation may invoke the video provider, and only
    # after the user agrees to the spoken offer. One call site, inside the consent handler.
    script = _client_script(client_with_reference_client)
    assert script.count("/clip-check") == 1
    assert "/clip-check" in _function_body(script, "handleConsentAgreement")
    assert (
        'getElementById("consent-yes").addEventListener("click", handleConsentAgreement)' in script
    )


def test_a_second_miss_speaks_the_plain_abstention_and_never_a_confident_negative(
    client_with_reference_client: TestClient,
) -> None:
    script = _client_script(client_with_reference_client)
    assert "I couldn't tell from the capture." in script
    assert "unanswerable" in script


def test_done_deletes_the_active_scene_session(
    client_with_reference_client: TestClient,
) -> None:
    script = _client_script(client_with_reference_client)
    assert '"DELETE"' in script
    assert "endActiveSceneSession" in script


def test_starting_a_new_capture_explicitly_ends_the_previous_scene_session(
    client_with_reference_client: TestClient,
) -> None:
    # The shared key carries no client identity, so the backend cannot infer which earlier scene
    # session a new capture supersedes; ending it is a client action on both capture paths.
    script = _client_script(client_with_reference_client)
    for handler in ("handleTap", "handleExcerptTap"):
        assert "endActiveSceneSession" in _function_body(script, handler)


def test_a_capture_that_never_produced_a_card_does_not_orphan_its_scene_session(
    client_with_reference_client: TestClient,
) -> None:
    # Every capture creates a scene session, including one that fails. The backend cannot infer
    # that nobody will ask about it, so the client that opened it closes it.
    body = _function_body(client_with_reference_client.get("/static/app.js").text, "runToSettlement")
    assert body.count("deleteSceneSession") == 2


def test_backgrounding_the_page_does_not_end_the_scene_session(
    client_with_reference_client: TestClient,
) -> None:
    # pagehide also fires when a phone backgrounds the page into the back/forward cache. Only a
    # real teardown ends the session; a restore with no session behind it closes the surface
    # rather than leaving controls that swallow taps.
    script = _client_script(client_with_reference_client)
    assert "event.persisted ||" in script
    assert '"pageshow"' in script
    assert "closeConversationSurface" in script
