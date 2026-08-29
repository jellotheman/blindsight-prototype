"""The canonical scene-card validator shared by every capture source and provider."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Confidence = Literal["high", "medium", "low"]
Distance = Literal["close", "middle", "far"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LayoutItem(ContractModel):
    thing: str = Field(min_length=1)
    relationship: str | None = Field(min_length=1)
    distance: Distance | None
    confidence: Confidence


class PersonObservation(ContractModel):
    count_description: str = Field(min_length=1)
    relationship: str | None = Field(min_length=1)
    activity: str | None = Field(min_length=1)
    confidence: Confidence


class ClaimUncertainty(ContractModel):
    claim: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class SceneCardBody(ContractModel):
    place_type: str | None = Field(min_length=1)
    place_type_confidence: Confidence | None
    overview: str = Field(min_length=1)
    layout: list[LayoutItem] | None
    open_space: str | None = Field(min_length=1)
    people: list[PersonObservation] | None
    visual_character: str | None = Field(min_length=1)
    uncertainties: list[ClaimUncertainty] | None

    @field_validator("overview")
    @classmethod
    def overview_has_at_most_fifty_words(cls, value: str) -> str:
        if len(value.split()) > 50:
            raise ValueError("overview must contain at most 50 words")
        return value

    @field_validator("uncertainties")
    @classmethod
    def uncertainty_array_is_never_empty(
        cls, value: list[ClaimUncertainty] | None
    ) -> list[ClaimUncertainty] | None:
        if value == []:
            raise ValueError("uncertainties must be null rather than an empty array")
        return value


class SceneCard(ContractModel):
    capture_id: str
    scene_session_id: str
    revision: int = Field(ge=1)
    evidence: list[str] = Field(min_length=1)
    card: SceneCardBody
