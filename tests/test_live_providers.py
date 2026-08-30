"""Opt-in live-provider smoke tests: real Reka and Gemini through the deployed Modal app's
Modal-hosted short-lived media URL.

CLAUDE.md's Test seam section is explicit that "live provider calls are smoke tests, not the
acceptance suite," and phase-0-1.md flags this exact flow -- Reka fetching a clip from a real
Modal-hosted URL -- as "the only unmeasured provider risk accepted by this specification." These
tests are how that risk gets measured. They hit real, billed provider APIs and a real Modal
deployment, so `python -m pytest` never runs them by default: every test here is skipped unless
every environment variable below is set.

Run against a live deployment with:

    export REKA_API_KEY=...
    export GEMINI_API_KEY=...
    export BLINDSIGHT_LIVE_PUBLIC_BASE_URL="https://<workspace>--blindsight-api-web.modal.run"
    python -m pytest tests/test_live_providers.py -v -m live

`BLINDSIGHT_LIVE_PUBLIC_BASE_URL` must name a currently-deployed instance of modal_app.py -- Reka
fetches the published media URL over the real internet, so this cannot run against an in-process
TestClient. Reading the shared Modal Dict additionally requires a local Modal identity (`modal
token new`, or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET), exactly like tools/replay.py's Reka path.

Each run is retained through the same FileEvidenceStore production uses, under
`runs/smoke/<capture_id>/`, including the append-only `index.jsonl` this module reads back to
report the validated-card rate. Nothing here should ever be cited as a measured Reka quality or
latency claim in the specification without a human first reading the recorded runs -- an automated
green check on this file is not that citation.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from blindsight.evidence import FileEvidenceStore, RunClock
from blindsight.excerpts import ExcerptCatalog, resolve_manifest_path
from blindsight.media_urls import ModalMediaUrlStore
from blindsight.providers import (
    CaptureEvidence,
    GeminiAdapter,
    ProductionProvider,
    ProviderResult,
    RekaChatAdapter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MANIFEST = REPO_ROOT / "data" / "excerpts" / "manifest.json"

PUBLIC_BASE_URL = os.environ.get("BLINDSIGHT_LIVE_PUBLIC_BASE_URL")
MODAL_DICT_NAME = os.environ.get("BLINDSIGHT_LIVE_MODAL_DICT", "blindsight-capture-state")
EXCERPT_VOLUME_NAME = os.environ.get("BLINDSIGHT_LIVE_EXCERPT_VOLUME", "blindsight-excerpts")
EXCERPT_SAMPLE_SIZE = int(os.environ.get("BLINDSIGHT_LIVE_EXCERPT_SAMPLE", "3"))
EVIDENCE_ROOT = Path(
    os.environ.get("BLINDSIGHT_LIVE_EVIDENCE_ROOT", str(REPO_ROOT / "runs" / "smoke"))
)

_REQUIRED_ENV = ("REKA_API_KEY", "GEMINI_API_KEY", "BLINDSIGHT_LIVE_PUBLIC_BASE_URL")
_MISSING = [name for name in _REQUIRED_ENV if not os.environ.get(name)]

_FAILURE_CODE_BY_KIND = {
    "timeout": "PROVIDER_TIMEOUT",
    "transport": "PROVIDER_UNAVAILABLE",
    "invalid_output": "MODEL_OUTPUT_INVALID",
}

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"opt-in live-provider smoke test: set {', '.join(_REQUIRED_ENV)} to run it",
    ),
]


@dataclass(frozen=True)
class LiveClip:
    excerpt_id: str
    content: bytes
    media_type: str


def _real_excerpt_volume_clips(sample_size: int) -> list[LiveClip] | None:
    """Sample real clips from the production `blindsight-excerpts` Modal Volume, if reachable.

    Returns None (never raises) on any failure -- unreachable volume, missing manifest, or a
    version-specific `Volume.read_file` API mismatch -- so callers always have a safe fallback to
    the small bundled dev manifest instead of failing the whole smoke run.
    """
    import modal

    try:
        volume = modal.Volume.from_name(EXCERPT_VOLUME_NAME, create_if_missing=False)
        manifest = json.loads(b"".join(volume.read_file("manifest.json")).decode("utf-8"))
        clips = []
        for item in manifest["items"][:sample_size]:
            content = b"".join(volume.read_file(item["clip_file"]))
            clips.append(
                LiveClip(excerpt_id=item["excerpt_id"], content=content, media_type="video/mp4")
            )
        return clips
    except Exception:
        return None


def _local_manifest_clips() -> list[LiveClip]:
    catalog = ExcerptCatalog(resolve_manifest_path([LOCAL_MANIFEST]))
    clips = []
    for entry in catalog.list_excerpts():
        excerpt_id = entry["excerpt_id"]
        assert isinstance(excerpt_id, str)
        content = catalog.evidence_bytes(excerpt_id)
        assert content is not None, f"bundled excerpt {excerpt_id!r} has no clip on disk"
        clips.append(LiveClip(excerpt_id=excerpt_id, content=content, media_type="video/mp4"))
    return clips


def _excerpt_clips() -> list[LiveClip]:
    """Prefer the real demonstration library; fall back to the bundled synthetic dev manifest.

    The bundled `data/excerpts/manifest.json` is documented in blindsight/excerpts.py as the
    "local-development fallback" -- it still exercises the real Reka/Gemini/media-url flow, just
    against placeholder clips rather than the real 74-excerpt library production actually serves.
    """
    return _real_excerpt_volume_clips(EXCERPT_SAMPLE_SIZE) or _local_manifest_clips()


def _live_media_urls() -> ModalMediaUrlStore:
    import modal

    assert PUBLIC_BASE_URL is not None  # guaranteed by the module skipif above
    try:
        dictionary = modal.Dict.from_name(MODAL_DICT_NAME, create_if_missing=False)
    except Exception as exc:  # pragma: no cover - depends on local Modal auth/deployment state
        pytest.skip(f"Modal Dict {MODAL_DICT_NAME!r} is not reachable: {exc}")
    return ModalMediaUrlStore(dictionary, PUBLIC_BASE_URL)


class BrokenMediaUrlStore:
    """Publishes for real, then hands Reka a same-host URL guaranteed to 404.

    Used only to force a genuine, unmocked Reka failure on the media fetch itself, so the real
    Gemini fallback path is exercised deterministically instead of hoping a good demo clip happens
    to trip Reka's real API on its own. Reka's own evidence -- not the original valid bytes -- is
    what's broken here: ProductionProvider.describe() always hands Gemini the original `evidence`
    object untouched, so Gemini still sees the real clip regardless of what Reka was given.
    """

    def __init__(self, real: ModalMediaUrlStore) -> None:
        self._real = real
        self.base_url = real.base_url

    def publish(self, evidence: CaptureEvidence) -> str:
        self._real.publish(evidence)  # published for realism; the URL below is deliberately wrong
        return f"{self.base_url}/_provider-media/smoke-test-unpublished-token"

    def revoke(self, url: str) -> None:
        pass


def _failure_envelope(result: ProviderResult) -> dict[str, object] | None:
    if result.card_body is not None:
        return None
    failure_kind = result.failure_kind or "invalid_output"
    return {
        "code": _FAILURE_CODE_BY_KIND[failure_kind],
        "message": result.error or "the live smoke run produced no validated card",
        "retryable": failure_kind in {"timeout", "transport"},
    }


def _run_and_retain(
    provider: ProductionProvider, evidence_store: FileEvidenceStore, clip: LiveClip, capture_id: str
) -> ProviderResult:
    clock = RunClock()
    evidence = CaptureEvidence(content=clip.content, media_type=clip.media_type, clock=clock)
    evidence_store.retain_capture(
        capture_id, f"smoke_{capture_id}", {"type": "excerpt", "excerpt_id": clip.excerpt_id}, evidence
    )
    clock.mark("evidence_retained_ms")
    result = provider.describe(evidence)
    clock.mark("completed_ms")
    evidence_store.finish(
        capture_id,
        result,
        accepted_card=result.card_body,
        failure=_failure_envelope(result),
        timings=clock.as_dict(),
    )
    return result


def test_live_produces_validated_cards() -> None:
    media_urls = _live_media_urls()
    provider = ProductionProvider(reka=RekaChatAdapter(), gemini=GeminiAdapter(), media_urls=media_urls)
    evidence_store = FileEvidenceStore(EVIDENCE_ROOT)

    clips = _excerpt_clips()
    assert clips, "no excerpts were available to measure"
    run_prefix = f"smoke{int(time.time())}{secrets.token_hex(4)}"
    results: list[tuple[LiveClip, ProviderResult, float]] = []
    for index, clip in enumerate(clips):
        capture_id = f"{run_prefix}{index:02d}"
        started = time.perf_counter()
        result = _run_and_retain(provider, evidence_store, clip, capture_id)
        results.append((clip, result, time.perf_counter() - started))

    for clip, result, latency_seconds in results:
        assert result.attempts, f"no provider attempts were recorded for {clip.excerpt_id!r}"
        assert result.provider in {"reka", "gemini", None}
        assert latency_seconds > 0

    # Filter by this run's own capture_id prefix rather than tailing the last N lines: index.jsonl
    # is an append-only log shared with any other smoke run against the same evidence root, and a
    # positional tail slice would silently mix in a concurrent run's rows.
    index_path = EVIDENCE_ROOT / "index.jsonl"
    recorded = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").strip().splitlines()
        if json.loads(line)["capture_id"].startswith(run_prefix)
    ]
    assert len(recorded) == len(clips), "did not find every run's own index.jsonl entry"
    validated_card_rate = sum(1 for row in recorded if row["succeeded"]) / len(recorded)

    print(  # noqa: T201 -- this is the human-facing measurement this smoke suite exists to produce
        f"\nlive Reka/Gemini smoke run: validated_card_rate={validated_card_rate:.2f} "
        f"over {len(recorded)} excerpts; providers used = "
        f"{sorted({row['provider'] for row in recorded if row['provider']})}"
    )
    assert 0.0 <= validated_card_rate <= 1.0


def test_live_transport_failure_falls_through_to_gemini() -> None:
    """A dead media URL is a Reka transport failure that still reaches the Gemini fallback.

    ProductionProvider lets every failure kind (invalid_output, transport, timeout) consume a
    Reka attempt and fall through to the single Gemini fallback (providers.py). Reka's real API
    may surface the dead URL as any of those kinds; either way both Reka attempts are consumed
    and Gemini is invoked exactly once with the real inline clip. Whether Gemini then succeeds
    depends on Reka's live error shape, so only the bounded attempt structure is asserted.
    """
    provider = ProductionProvider(
        reka=RekaChatAdapter(),
        gemini=GeminiAdapter(),
        media_urls=BrokenMediaUrlStore(_live_media_urls()),
    )
    clip = _excerpt_clips()[0]

    result = provider.describe(CaptureEvidence(content=clip.content, media_type=clip.media_type))

    providers = [attempt.provider for attempt in result.attempts]
    assert providers == ["reka", "reka", "gemini"]
    assert result.attempts[0].failure_kind in {"transport", "timeout", "invalid_output"}
    if result.card_body is not None:
        assert result.provider == "gemini"
    else:
        assert result.failure_kind in {"transport", "timeout", "invalid_output"}
