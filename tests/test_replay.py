from __future__ import annotations

import json
from pathlib import Path

from blindsight.evidence import FileEvidenceStore
from blindsight.providers import CaptureEvidence, ProviderResult
from blindsight.replay import ReplayService

from .test_captures import VALID_CARD_BODY


class ReplayProvider:
    name = "chosen-provider"
    model = "chosen-model"

    def describe(self, evidence: CaptureEvidence) -> ProviderResult:
        return ProviderResult(
            raw_text=json.dumps(VALID_CARD_BODY),
            card_body=VALID_CARD_BODY,
            provider=self.name,
            model=self.model,
            attempt=1,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def seed_run(root: Path, capture_id: str) -> Path:
    evidence = FileEvidenceStore(root)
    captured = CaptureEvidence(content=b"retained-video", media_type="video/mp4")
    evidence.retain_capture(capture_id, "ses_original", {"type": "live"}, captured)
    evidence.finish(
        capture_id,
        ProviderResult(raw_text=json.dumps(VALID_CARD_BODY), card_body=VALID_CARD_BODY),
        accepted_card={"immutable": True},
        failure=None,
        timings={"completed_ms": 1.0},
    )
    return root / capture_id


def test_replay_writes_beside_original_without_mutating_accepted_record(tmp_path: Path) -> None:
    run_dir = seed_run(tmp_path / "runs", "cap_original")
    original_run = (run_dir / "run.json").read_bytes()
    original_card = (run_dir / "card.json").read_bytes()

    output = ReplayService(ReplayProvider()).replay(run_dir)

    assert output.parent == run_dir / "replays"
    replay = json.loads(output.read_text())
    assert replay["provider"] == "chosen-provider"
    assert replay["valid"] is True
    assert replay["card"] == VALID_CARD_BODY
    assert (run_dir / "run.json").read_bytes() == original_run
    assert (run_dir / "card.json").read_bytes() == original_card


def test_replay_set_processes_each_selected_retained_capture(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    seed_run(root, "cap_one")
    seed_run(root, "cap_two")

    outputs = ReplayService(ReplayProvider()).replay_many(root, ["cap_two", "cap_one"])

    assert [path.parents[1].name for path in outputs] == ["cap_two", "cap_one"]
