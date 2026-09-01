from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1)
    owner: str | None = None

    @field_validator("task")
    @classmethod
    def strip_task(cls, value: str) -> str:
        return value.strip()

    @field_validator("owner")
    @classmethod
    def normalize_owner(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]


class FinalSessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_summary: str = Field(min_length=1)
    main_topics: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]


class GeneratedCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    checkpoint: Checkpoint
    raw_response: str
    generation_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> GeneratedCheckpoint:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class GeneratedSummaryBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoints: tuple[GeneratedCheckpoint, ...]
    final_summary: FinalSessionSummary
    final_raw_response: str
    processing_seconds: float = Field(ge=0)


class StoredSummaryCheckpoint(Checkpoint):
    id: int
    session_id: int
    chunk_index: int
    start_seconds: float
    end_seconds: float
    model_name: str
    generation_seconds: float | None = None
    created_at: datetime


class StoredFinalSummary(FinalSessionSummary):
    id: int
    session_id: int
    model_name: str
    checkpoint_count: int
    processing_seconds: float | None = None
    created_at: datetime

