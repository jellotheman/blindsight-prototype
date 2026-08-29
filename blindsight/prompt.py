"""The single Stage 0 prompt shared by live processing and evidence replay."""

from __future__ import annotations

import json

from .scene_card import SceneCardBody


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
