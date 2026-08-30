"""One-command local dev workflow: serve the BlindSight API locally and publish it through a
Cloudflare quick tunnel, so a phone can exercise the identical `/v1` HTTP interface it would reach
on Modal. See docs/spec/phase-0-1.md's "Keep local development and deployment equivalent" decision.

Run with:

    python -m tools.local_dev --api-key <shared key>

This is developer convenience; it blocks nothing else and uses the same in-memory store and
deterministic provider as the walking-skeleton defaults in blindsight.app.create_app. Live
provider wiring belongs to the deployed Modal application, not this tool.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import IO

from fastapi import FastAPI

from blindsight.app import create_app, mount_frontend_client, mount_reference_client

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
FRONTEND_DIST_DIR = REPO_ROOT / "frontend" / "dist"

TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
DEFAULT_TUNNEL_TIMEOUT_SECONDS = 30.0


class LauncherError(RuntimeError):
    """Base class for launcher failures the caller should surface loudly and exit non-zero for."""


class ToolMissingError(LauncherError):
    pass


class PortOccupiedError(LauncherError):
    pass


class TunnelTimeoutError(LauncherError):
    pass


def find_tunnel_url(text: str) -> str | None:
    match = TUNNEL_URL_PATTERN.search(text)
    return match.group(0) if match else None


def require_tool(name: str, *, hint: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolMissingError(f"Required tool {name!r} was not found on PATH. {hint}")
    return path


def require_free_port(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise PortOccupiedError(
                f"Port {port} on {host} is already in use: {exc}. "
                "Stop whatever is using it, or pass --port with a free one."
            ) from exc


def wait_for_tunnel_url(lines: Queue[str | None], *, timeout_seconds: float) -> str:
    """Read cloudflared output lines from `lines` until a tunnel URL appears or time runs out.

    `None` on the queue is the sentinel for "the process exited" -- pushed by the reader thread
    that owns the subprocess pipe.
    """
    import time

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TunnelTimeoutError(
                f"cloudflared did not publish a tunnel URL within {timeout_seconds:.0f}s."
            )
        try:
            line = lines.get(timeout=remaining)
        except Empty:
            raise TunnelTimeoutError(
                f"cloudflared did not publish a tunnel URL within {timeout_seconds:.0f}s."
            ) from None
        if line is None:
            raise TunnelTimeoutError("cloudflared exited before publishing a tunnel URL.")
        url = find_tunnel_url(line)
        if url:
            return url


def build_app(*, api_key: str) -> FastAPI:
    app = create_app(api_key=api_key)
    mount_reference_client(app, static_dir=STATIC_DIR)
    mount_frontend_client(app, frontend_dist_dir=FRONTEND_DIST_DIR)
    return app


def _pump_output(pipe: "IO[str]", sink: "Queue[str | None]") -> None:
    try:
        for line in pipe:
            sink.put(line)
    finally:
        sink.put(None)


def _start_cloudflared(cloudflared_path: str, port: int) -> tuple[subprocess.Popen, "Queue[str | None]"]:
    process = subprocess.Popen(
        [cloudflared_path, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: Queue[str | None] = Queue()
    assert process.stdout is not None
    threading.Thread(target=_pump_output, args=(process.stdout, lines), daemon=True).start()
    return process, lines


def _run_server(app: FastAPI, host: str, port: int) -> tuple["object", threading.Thread]:
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BLINDSIGHT_API_KEY"),
        help="Shared key the local server requires. Prompts when omitted.",
    )
    parser.add_argument(
        "--generate-api-key",
        action="store_true",
        help="Generate and print a temporary key instead of prompting for one.",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help="Serve only on this computer instead of opening a Cloudflare quick tunnel.",
    )
    parser.add_argument("--tunnel-timeout-seconds", type=float, default=DEFAULT_TUNNEL_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    generated_api_key = False
    if args.generate_api_key:
        args.api_key = secrets.token_urlsafe(24)
        generated_api_key = True
    elif not args.api_key:
        try:
            args.api_key = getpass.getpass(
                "Choose a temporary API key (enter the same key in BlindSight Settings): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("An API key is required.", file=sys.stderr)
            return 1
        if not args.api_key:
            print("An API key is required.", file=sys.stderr)
            return 1

    try:
        require_free_port(args.host, args.port)
        cloudflared_path = None
        if not args.no_tunnel:
            cloudflared_path = require_tool(
                "cloudflared",
                hint=(
                    "Install it from https://developers.cloudflare.com/cloudflared/downloads/ "
                    "(or `winget install cloudflare.cloudflared`), then try again."
                ),
            )
    except LauncherError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    app = build_app(api_key=args.api_key)
    server, server_thread = _run_server(app, args.host, args.port)
    if args.no_tunnel:
        print(f"BlindSight is available on this computer at: http://{args.host}:{args.port}")
        if generated_api_key:
            print(f"Temporary API key for BlindSight Settings: {args.api_key}")
        print("Press Ctrl+C to stop the server.")
        try:
            while server_thread.is_alive():
                server_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            server.should_exit = True  # type: ignore[attr-defined]
        return 0

    assert cloudflared_path is not None
    tunnel_process, tunnel_lines = _start_cloudflared(cloudflared_path, args.port)

    try:
        tunnel_url = wait_for_tunnel_url(tunnel_lines, timeout_seconds=args.tunnel_timeout_seconds)
    except LauncherError as exc:
        print(str(exc), file=sys.stderr)
        _shutdown(server, tunnel_process)
        return 1

    print(f"BlindSight is reachable from a phone at: {tunnel_url}")
    if generated_api_key:
        print(f"Temporary API key for BlindSight Settings: {args.api_key}")
    print(f"Local server: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop both the server and the tunnel.")

    try:
        while server_thread.is_alive() and tunnel_process.poll() is None:
            server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(server, tunnel_process)

    return 0


def _shutdown(server: object, tunnel_process: subprocess.Popen) -> None:
    server.should_exit = True  # type: ignore[attr-defined]
    if tunnel_process.poll() is None:
        tunnel_process.terminate()
        try:
            tunnel_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel_process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
