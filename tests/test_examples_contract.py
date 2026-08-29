"""Every worked example in docs/spec/examples.md must stay valid against the OpenAPI document.

This is the permanent regression for the "contract-only client" acceptance bar folded into issue
#10: a client built from the OpenAPI document and these examples alone must never have to guess a
shape the examples themselves get wrong. Rather than pin each example to a hand-picked schema name,
this walks the markdown in document order, resolves each curl call's path against the OpenAPI
`paths` object, and validates request bodies (``-d '...'``) and the JSON response blocks that follow
against whatever that operation actually documents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tests.conftest import SchemaValidator

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_PATH = REPO_ROOT / "docs" / "spec" / "examples.md"

_FENCE_RE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
_METHOD_RE = re.compile(r"-X\s+(\w+)")
_URL_RE = re.compile(r'"\$BLINDSIGHT_URL([^"]*)"')
_DATA_RE = re.compile(r"-d\s+'((?:[^'\\]|\\.)*)'", re.DOTALL)


def _fences(text: str) -> list[tuple[str, str]]:
    return _FENCE_RE.findall(text)


def _path_template(url_path: str, doc_paths: list[str]) -> str:
    segments = [s for s in url_path.split("?")[0].split("/") if s]
    for template in doc_paths:
        template_segments = [s for s in template.split("/") if s]
        if len(template_segments) != len(segments):
            continue
        if all(
            (t.startswith("{") and t.endswith("}")) or t == s
            for t, s in zip(template_segments, segments)
        ):
            return template
    raise AssertionError(f"no OpenAPI path matches example URL path {url_path!r}")


def _curl_operation(body: str, doc_paths: list[str]) -> tuple[str, str] | None:
    url_match = _URL_RE.search(body)
    if url_match is None:
        return None
    method_match = _METHOD_RE.search(body)
    method = (method_match.group(1) if method_match else "GET").lower()
    return method, _path_template(url_match.group(1), doc_paths)


def test_every_curl_request_body_matches_its_documented_request_schema(
    openapi_doc: dict[str, Any], schema: SchemaValidator
) -> None:
    text = EXAMPLES_PATH.read_text(encoding="utf-8")
    doc_paths = list(openapi_doc["paths"].keys())
    checked = 0
    for language, body in _fences(text):
        if language != "bash":
            continue
        operation_key = _curl_operation(body, doc_paths)
        data_match = _DATA_RE.search(body)
        if operation_key is None or data_match is None:
            continue
        method, path_template = operation_key
        payload = json.loads(data_match.group(1))
        operation = openapi_doc["paths"][path_template][method]
        request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
        schema.assert_schema(request_schema, payload, f"{method.upper()} {path_template} request body")
        checked += 1
    assert checked >= 4, "expected to validate every documented JSON request body in the examples"


def test_every_json_response_block_matches_its_documented_response_schema(
    openapi_doc: dict[str, Any], schema: SchemaValidator
) -> None:
    text = EXAMPLES_PATH.read_text(encoding="utf-8")
    doc_paths = list(openapi_doc["paths"].keys())
    last_operation: tuple[str, str] | None = None
    checked = 0
    for language, body in _fences(text):
        if language == "bash":
            operation_key = _curl_operation(body, doc_paths)
            if operation_key is not None:
                last_operation = operation_key
            continue
        if language != "json":
            continue
        payload = json.loads(body)
        if not isinstance(payload, dict) or "error" in payload:
            continue
        assert last_operation is not None, "a JSON example has no preceding curl call to anchor it"
        method, path_template = last_operation
        operation = openapi_doc["paths"][path_template][method]
        success_status = next(status for status in operation["responses"] if status.startswith("2"))
        response_content = operation["responses"][success_status].get("content", {})
        assert "application/json" in response_content, (
            f"a JSON example block followed {method.upper()} {path_template}, whose {success_status} "
            "response isn't documented as JSON -- check it's anchored to the right curl call"
        )
        response_schema = response_content["application/json"]["schema"]
        schema.assert_schema(
            response_schema, payload, f"{method.upper()} {path_template} {success_status} example"
        )
        checked += 1
    assert checked >= 3, "expected to validate every documented JSON response example"
