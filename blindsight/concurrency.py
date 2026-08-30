"""Bounded-deadline execution shared by capture and question processing."""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any, Callable


def run_with_deadline(operation: Callable[[], Any], timeout_seconds: float) -> tuple[str, Any]:
    """Run `operation` in a worker thread.

    Returns `("result", value)`, `("error", exc)`, or `("timeout", None)` if `operation` does not
    complete within `timeout_seconds`. No job may remain in `processing` indefinitely -- see
    docs/spec/phase-0-1.md.
    """
    results: Queue[tuple[str, Any]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            results.put(("result", operation()))
        except Exception as exc:
            results.put(("error", exc))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        return results.get(timeout=timeout_seconds)
    except Empty:
        return ("timeout", None)
