from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from blindsight.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent
OPENAPI_PATH = REPO_ROOT / "docs" / "spec" / "openapi.yaml"
TEST_API_KEY = "test-shared-key-0123456789"


@pytest.fixture(scope="session")
def openapi_doc() -> dict[str, Any]:
    return yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))


class SchemaValidator:
    """Validates a value against a named component schema in the OpenAPI document."""

    _BASE_URI = "urn:blindsight:openapi"

    def __init__(self, doc: dict[str, Any]) -> None:
        self._doc = doc
        resource = Resource.from_contents(doc, default_specification=DRAFT202012)
        self._registry = Registry().with_resource(uri=self._BASE_URI, resource=resource)

    def assert_valid(self, schema_name: str, instance: Any) -> None:
        schema = {"$ref": f"{self._BASE_URI}#/components/schemas/{schema_name}"}
        self._assert_schema(schema, instance, schema_name)

    def assert_schema(self, raw_schema: dict[str, Any], instance: Any, label: str) -> None:
        """Validate against an arbitrary schema fragment taken directly from the OpenAPI document."""
        self._assert_schema(raw_schema, instance, label)

    def assert_json_response(self, path: str, method: str, response: Any) -> None:
        """Validate status, media type, declared headers, and JSON body from OpenAPI."""
        operation = self._doc["paths"][path][method.lower()]
        status = str(response.status_code)
        assert status in operation["responses"], (
            f"OpenAPI does not document {response.status_code} for {method.upper()} {path}"
        )

        response_spec = operation["responses"][status]
        response_ref = response_spec.get("$ref")
        if "$ref" in response_spec:
            response_spec = self._resolve_local_ref(response_spec["$ref"])

        for header in response_spec.get("headers", {}):
            assert header in response.headers, f"documented response header {header!r} is missing"

        content = response_spec.get("content", {})
        assert "application/json" in response.headers.get("content-type", "")
        assert "application/json" in content, "OpenAPI does not document a JSON response"
        if response_ref:
            schema_pointer = response_ref
        else:
            escaped_path = path.replace("~", "~0").replace("/", "~1")
            schema_pointer = f"#/paths/{escaped_path}/{method.lower()}/responses/{status}"
        schema = {
            "$ref": (
                f"{self._BASE_URI}{schema_pointer}"
                "/content/application~1json/schema"
            )
        }
        self._assert_schema(schema, response.json(), f"{method.upper()} {path} {status}")

    def _resolve_local_ref(self, ref: str) -> dict[str, Any]:
        assert ref.startswith("#/"), f"unsupported non-local OpenAPI reference: {ref}"
        value: Any = self._doc
        for token in ref[2:].split("/"):
            value = value[token.replace("~1", "/").replace("~0", "~")]
        return value

    def _assert_schema(self, schema: dict[str, Any], instance: Any, label: str) -> None:
        if "$ref" in schema and schema["$ref"].startswith("#/"):
            schema = {"$ref": f"{self._BASE_URI}{schema['$ref']}"}
        validator = Draft202012Validator(schema, registry=self._registry)
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        if errors:
            messages = "\n".join(f"  - {'.'.join(map(str, e.path))}: {e.message}" for e in errors)
            raise AssertionError(f"{label} schema violations:\n{messages}")


@pytest.fixture(scope="session")
def schema(openapi_doc: dict[str, Any]) -> SchemaValidator:
    return SchemaValidator(openapi_doc)


@pytest.fixture()
def api_key() -> str:
    return TEST_API_KEY


@pytest.fixture()
def client(api_key: str) -> TestClient:
    app = create_app(api_key=api_key)
    return TestClient(app)


@pytest.fixture()
def auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}


# Every documented /v1 route. New tickets append here so the "every versioned operation rejects
# a missing/incorrect key" acceptance criterion keeps covering the whole surface, not just the
# routes this ticket implements.
VERSIONED_GET_ROUTES = [
    "/v1/excerpts",
    "/v1/excerpts/via-001-entry-02/poster",
]
