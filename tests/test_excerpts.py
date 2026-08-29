from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blindsight.excerpts import resolve_manifest_path
from blindsight.identifiers import is_valid_identifier
from tests.conftest import SchemaValidator, VERSIONED_GET_ROUTES


def test_listing_excerpts_matches_documented_shape(
    client: TestClient, auth_headers: dict[str, str], schema: SchemaValidator
) -> None:
    response = client.get("/v1/excerpts", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    schema.assert_json_response("/v1/excerpts", "get", response)
    assert body["items"], "expected at least one real excerpt entry"
    for item in body["items"]:
        schema.assert_valid("Excerpt", item)
        assert is_valid_identifier(item["excerpt_id"])


def test_excerpt_ids_are_contract_identifiers_not_storage_paths(client: TestClient, auth_headers) -> None:
    response = client.get("/v1/excerpts", headers=auth_headers)
    for item in response.json()["items"]:
        excerpt_id = item["excerpt_id"]
        assert "/" not in excerpt_id
        assert not excerpt_id.endswith((".mp4", ".jpg", ".json"))


def test_poster_returns_jpeg_for_a_listed_excerpt(client: TestClient, auth_headers) -> None:
    listed = client.get("/v1/excerpts", headers=auth_headers).json()["items"]
    excerpt_id = listed[0]["excerpt_id"]

    response = client.get(f"/v1/excerpts/{excerpt_id}/poster", headers=auth_headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_poster_for_every_listed_excerpt_is_retrievable(client: TestClient, auth_headers) -> None:
    listed = client.get("/v1/excerpts", headers=auth_headers).json()["items"]
    for item in listed:
        response = client.get(item["poster_url"], headers=auth_headers)
        assert response.status_code == 200
        assert response.content[:2] == b"\xff\xd8"


def test_unknown_excerpt_poster_returns_documented_not_found_envelope(
    client: TestClient, auth_headers, schema: SchemaValidator
) -> None:
    response = client.get("/v1/excerpts/nonexistent-excerpt-id/poster", headers=auth_headers)

    assert response.status_code == 404
    body = response.json()
    schema.assert_json_response("/v1/excerpts/{excerpt_id}/poster", "get", response)
    assert body["error"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize("path", VERSIONED_GET_ROUTES)
def test_missing_api_key_is_rejected(client: TestClient, path: str, schema: SchemaValidator) -> None:
    response = client.get(path)

    assert response.status_code == 401
    body = response.json()
    schema.assert_json_response(path if path == "/v1/excerpts" else "/v1/excerpts/{excerpt_id}/poster", "get", response)
    assert body["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize("path", VERSIONED_GET_ROUTES)
def test_incorrect_api_key_is_rejected(client: TestClient, path: str, schema: SchemaValidator) -> None:
    response = client.get(path, headers={"X-API-Key": "definitely-the-wrong-key"})

    assert response.status_code == 401
    body = response.json()
    schema.assert_json_response(path if path == "/v1/excerpts" else "/v1/excerpts/{excerpt_id}/poster", "get", response)
    assert body["error"]["code"] == "UNAUTHORIZED"


def test_non_versioned_routes_are_not_gated_by_the_api_key(client: TestClient) -> None:
    # The reference client shell (index page, static assets) has no data of its own to protect;
    # only /v1 operations require the key. This app doesn't mount the client, so assert the
    # narrower claim directly: an unrelated unversioned path is never caught by the middleware.
    response = client.get("/healthz")
    assert response.status_code == 404  # unmatched route, not 401 -- middleware didn't intercept it


def test_resolve_manifest_path_picks_the_first_present_candidate(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled" / "manifest.json"
    volume = tmp_path / "volume" / "manifest.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text('{"items": []}', encoding="utf-8")

    # Only the bundled fallback exists: it is selected even when listed second.
    assert resolve_manifest_path([volume, bundled]) == bundled

    # Once the volume manifest exists it wins because it is listed first.
    volume.parent.mkdir(parents=True)
    volume.write_text('{"items": []}', encoding="utf-8")
    assert resolve_manifest_path([volume, bundled]) == volume


def test_resolve_manifest_path_raises_when_no_candidate_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_manifest_path([tmp_path / "missing1.json", tmp_path / "missing2.json"])
