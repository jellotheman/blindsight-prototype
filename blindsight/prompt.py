"""Prompts shared by live processing, evidence replay, and Stage 1 follow-up questions."""

from __future__ import annotations

import json

from .scene_card import AnswerBody, SceneCardBody


def build_scene_card_prompt() -> str:
    schema = json.dumps(SceneCardBody.model_json_schema(), separators=(",", ":"))
    return f"""You help a blind or low-vision person understand a captured view from a short video.
This is environmental understanding, not navigation or a safety assessment. Describe only visual
evidence in the completed capture. Do not imply complete coverage or current awareness.

Return exactly one JSON object matching this complete JSON Schema, with every key present and no
markdown fence or prose outside the object:
{schema}

The overview is the only content spoken unasked. Target 30-45 words and never exceed 50. Prioritize
place type when evident, occupancy and apparent activity, two or three dominant objects and their
relationships, material open-space shape, then a short directly observed colour/light/material
clause if it fits. Name objects plainly. Use null when evidence cannot determine an optional claim;
use an empty people or layout array only when the evidence supports that none were observed. Every
uncertainty must identify its affected claim and the qualification. Abstain rather than invent.
Never identify a person or infer atmosphere or style as if directly observed.

Assistant response begins now:
{{"""


def build_card_answer_prompt() -> str:
    schema = json.dumps(AnswerBody.model_json_schema(), separators=(",", ":"))
    return f"""You answer a follow-up question about a captured view for a blind or low-vision
person, using only the scene card and conversation that follow, and nothing else. This is
environmental understanding, not navigation or a safety assessment. Earlier turns may use
pronouns such as "it" that refer back to something named in the scene card or an earlier answer;
resolve them from that context. If the scene card does not support an answer, return null rather
than inventing one -- abstaining is correct, not a failure. Never state a confident negative.

Return exactly one JSON object matching this complete JSON Schema, with every key present and no
markdown fence or prose outside the object:
{schema}

Assistant response begins now:
{{"""


def build_captured_view_prompt() -> str:
    schema = json.dumps(AnswerBody.model_json_schema(), separators=(",", ":"))
    return f"""You are re-checking a previously captured view to answer one follow-up question
that the scene card alone could not answer. Use only what is directly visible in the captured
view and the conversation that follows; do not imply awareness beyond that recording. If you
still cannot determine the answer, return null rather than inventing one -- abstaining is correct,
not a failure. Never state a confident negative.

Return exactly one JSON object matching this complete JSON Schema, with every key present and no
markdown fence or prose outside the object:
{schema}

Assistant response begins now:
{{"""
