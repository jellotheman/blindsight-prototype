"""Stage 1 follow-up conversation lifecycle, independent of HTTP, storage, and provider
implementations.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .captures import ThreadRunner
from .concurrency import run_with_deadline
from .errors import ApiError, NotFound
from .providers import CapturedViewProvider, CardAnswerProvider, CaptureEvidence, QuestionAnswer
from .storage import CaptureStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


_FAILURE_CODES = {
    "timeout": "PROVIDER_TIMEOUT",
    "transport": "PROVIDER_UNAVAILABLE",
    "invalid_output": "MODEL_OUTPUT_INVALID",
}
_FAILURE_MESSAGES = {
    "timeout": "The provider timed out while answering the question.",
    "transport": "The provider was unavailable while answering the question.",
    "invalid_output": "The provider returned an invalid answer.",
}


class QuestionService:
    def __init__(
        self,
        *,
        store: CaptureStore,
        card_provider: CardAnswerProvider,
        captured_view_provider: CapturedViewProvider,
        runner: ThreadRunner | None = None,
        processing_deadline_seconds: float = 60.0,
    ) -> None:
        self.store = store
        self.card_provider = card_provider
        self.captured_view_provider = captured_view_provider
        self.runner = runner or ThreadRunner()
        self.processing_deadline_seconds = processing_deadline_seconds

    def create_question(self, scene_session_id: str, question: str) -> dict[str, Any]:
        session = self.store.get_session(scene_session_id)
        if session is None:
            raise NotFound(f"No scene session with id {scene_session_id!r}.")
        capture = self.store.get_capture(session["capture_id"])
        if capture is None or capture["card"] is None:
            raise ApiError(
                409,
                "INVALID_STATE",
                "The scene session has no scene card yet.",
                details={"status": capture["status"] if capture is not None else "unknown"},
            )

        question_id = _identifier("que")
        created_at = _now()
        resource: dict[str, Any] = {
            "question_id": question_id,
            "scene_session_id": scene_session_id,
            "question": question,
            "status": "processing",
            "answer": None,
            "source": None,
            "failure": None,
            "created_at": created_at,
            "updated_at": created_at,
        }
        self.store.create_question(resource)
        session["conversation"].append({"role": "user", "content": question})
        # Recorded so deleting the scene session can drop its question resources with it.
        session.setdefault("question_ids", []).append(question_id)
        self.store.put_session(scene_session_id, session)

        self.runner.submit(lambda: self._answer_from_card(question_id))
        return resource

    def get(self, scene_session_id: str, question_id: str) -> dict[str, Any] | None:
        if self.store.get_session(scene_session_id) is None:
            return None
        resource = self.store.get_question(question_id)
        if resource is None or resource["scene_session_id"] != scene_session_id:
            return None
        return resource

    def clip_check(self, scene_session_id: str, question_id: str) -> dict[str, Any]:
        session = self.store.get_session(scene_session_id)
        if session is None:
            raise NotFound(f"No scene session with id {scene_session_id!r}.")
        resource = self.store.get_question(question_id)
        if resource is None or resource["scene_session_id"] != scene_session_id:
            raise NotFound(f"No question with id {question_id!r}.")
        if resource["status"] != "needs_clip_consent":
            raise ApiError(
                409,
                "INVALID_STATE",
                "The captured-view check is valid only from needs_clip_consent.",
                details={"status": resource["status"]},
            )
        media = self.store.get_media(session["capture_id"])
        if media is None:
            raise ApiError(
                409,
                "INVALID_STATE",
                "The captured view is no longer available to check.",
            )

        resource["status"] = "processing"
        resource["updated_at"] = _now()
        self.store.put_question(question_id, resource)

        content, media_type = media
        evidence = CaptureEvidence(content=content, media_type=media_type)
        self.runner.submit(lambda: self._answer_from_captured_view(question_id, evidence))
        return resource

    def delete_session(self, scene_session_id: str) -> bool:
        if self.store.get_session(scene_session_id) is None:
            return False
        self.store.delete_session(scene_session_id)
        return True

    def _answer_from_card(self, question_id: str) -> None:
        resource = self.store.get_question(question_id)
        if resource is None:
            return
        session = self.store.get_session(resource["scene_session_id"])
        if session is None:
            return
        capture = self.store.get_capture(session["capture_id"])
        if capture is None or capture["card"] is None:
            self._settle_internal_error(resource)
            return

        card_body = capture["card"]["card"]
        try:
            outcome, value = run_with_deadline(
                lambda: self.card_provider.answer(card_body, session["conversation"]),
                self.processing_deadline_seconds,
            )
            result = self._result_from_outcome(outcome, value)
        except Exception:
            self._settle_internal_error(resource)
            return

        self._settle(
            resource,
            session,
            result,
            source_on_answer="scene_card",
            miss_status="needs_clip_consent",
        )

    def _answer_from_captured_view(self, question_id: str, evidence: CaptureEvidence) -> None:
        resource = self.store.get_question(question_id)
        if resource is None:
            return
        session = self.store.get_session(resource["scene_session_id"])
        if session is None:
            return

        try:
            outcome, value = run_with_deadline(
                lambda: self.captured_view_provider.answer(evidence, session["conversation"]),
                self.processing_deadline_seconds,
            )
            result = self._result_from_outcome(outcome, value)
        except Exception:
            self._settle_internal_error(resource)
            return

        self._settle(
            resource,
            session,
            result,
            source_on_answer="captured_view",
            miss_status="unanswerable",
        )

    @staticmethod
    def _result_from_outcome(outcome: str, value: Any) -> QuestionAnswer:
        if outcome == "timeout":
            return QuestionAnswer(
                answer=None, failure_kind="timeout", error="The processing deadline was exceeded."
            )
        if outcome == "error":
            raise value
        if not isinstance(value, QuestionAnswer):
            raise RuntimeError("Question provider returned an invalid result type.")
        return value

    def _settle(
        self,
        resource: dict[str, Any],
        session: dict[str, Any],
        result: QuestionAnswer,
        *,
        source_on_answer: str,
        miss_status: str,
    ) -> None:
        if result.failure_kind is not None:
            resource["status"] = "failed"
            resource["failure"] = {
                "code": _FAILURE_CODES[result.failure_kind],
                "message": _FAILURE_MESSAGES[result.failure_kind],
                "retryable": result.failure_kind in {"timeout", "transport"},
            }
            resource["updated_at"] = _now()
            self.store.put_question(resource["question_id"], resource)
            return

        resource["updated_at"] = _now()
        if result.answer is not None:
            resource["status"] = "answered"
            resource["answer"] = result.answer
            resource["source"] = source_on_answer
            resource["failure"] = None
            session["conversation"].append({"role": "assistant", "content": result.answer})
            self.store.put_session(session["scene_session_id"], session)
        else:
            resource["status"] = miss_status
            resource["answer"] = None
            resource["source"] = "captured_view" if miss_status == "unanswerable" else None
            resource["failure"] = None
        self.store.put_question(resource["question_id"], resource)

    def _settle_internal_error(self, resource: dict[str, Any]) -> None:
        resource["status"] = "failed"
        resource["failure"] = {
            "code": "INTERNAL_ERROR",
            "message": "The question could not be processed.",
            "retryable": True,
        }
        resource["updated_at"] = _now()
        self.store.put_question(resource["question_id"], resource)
