"""Tests for the one-command local dev launcher (tools/local_dev.py).

These cover the testable seams: tunnel URL extraction from noisy cloudflared output, the
tool-missing and port-occupied guard checks, and the bounded wait for a published tunnel URL.
Spawning real subprocesses (uvicorn, cloudflared) is integration behavior exercised by hand
against a real phone, per docs/spec/phase-0-1.md's testing decisions.
"""

from __future__ import annotations

import socket
from queue import Queue

import pytest
from fastapi.testclient import TestClient

from tools.local_dev import (
    PortOccupiedError,
    ToolMissingError,
    TunnelTimeoutError,
    build_app,
    find_tunnel_url,
    require_free_port,
    require_tool,
    wait_for_tunnel_url,
)


def test_find_tunnel_url_extracts_the_https_url_from_noisy_cloudflared_output() -> None:
    line = (
        "2026-08-29T12:00:00Z INF |  https://quiet-otter-42.trycloudflare.com  | "
        "2026-08-29T12:00:00Z INF Registered tunnel connection"
    )
    assert find_tunnel_url(line) == "https://quiet-otter-42.trycloudflare.com"


def test_find_tunnel_url_returns_none_for_unrelated_lines() -> None:
    assert find_tunnel_url("2026-08-29T12:00:00Z INF Starting tunnel") is None


def test_require_tool_raises_actionably_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(ToolMissingError, match="cloudflared"):
        require_tool("cloudflared", hint="Install cloudflared and try again.")


def test_require_tool_returns_the_resolved_path_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert require_tool("cloudflared", hint="unused") == "/usr/bin/cloudflared"


def test_require_free_port_raises_when_the_port_is_already_bound() -> None:
    occupying_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupying_socket.bind(("127.0.0.1", 0))
    occupying_socket.listen(1)
    port = occupying_socket.getsockname()[1]
    try:
        with pytest.raises(PortOccupiedError, match=str(port)):
            require_free_port("127.0.0.1", port)
    finally:
        occupying_socket.close()


def test_require_free_port_succeeds_for_an_unused_port() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    free_port = probe.getsockname()[1]
    probe.close()

    require_free_port("127.0.0.1", free_port)


def test_wait_for_tunnel_url_skips_non_matching_lines_and_returns_the_url() -> None:
    lines: Queue[str | None] = Queue()
    lines.put("starting tunnel")
    lines.put("connecting...")
    lines.put("published at https://quiet-otter-42.trycloudflare.com now")

    url = wait_for_tunnel_url(lines, timeout_seconds=1.0)

    assert url == "https://quiet-otter-42.trycloudflare.com"


def test_wait_for_tunnel_url_times_out_when_nothing_arrives() -> None:
    lines: Queue[str | None] = Queue()

    with pytest.raises(TunnelTimeoutError):
        wait_for_tunnel_url(lines, timeout_seconds=0.05)


def test_wait_for_tunnel_url_treats_process_exit_sentinel_as_a_failure() -> None:
    lines: Queue[str | None] = Queue()
    lines.put("starting tunnel")
    lines.put(None)

    with pytest.raises(TunnelTimeoutError, match="exited"):
        wait_for_tunnel_url(lines, timeout_seconds=1.0)


def test_build_app_serves_the_expo_client_and_the_authenticated_v1_interface() -> None:
    app = build_app(api_key="local-dev-key-0123456789")
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert b"/_expo/static/js/web/entry-" in page.content

    reference = client.get("/reference/")
    assert reference.status_code == 200
    assert b"Tap the center of the screen to record." in reference.content

    unauthenticated = client.get("/v1/excerpts")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/v1/excerpts", headers={"X-API-Key": "local-dev-key-0123456789"})
    assert authenticated.status_code == 200
