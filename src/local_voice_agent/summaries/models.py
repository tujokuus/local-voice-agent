from __future__ import annotations

from datetime import datetime
from typing import Literal

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


class EvidenceItem(BaseModel):
    """A transcript-supported observation with its narrowest useful evidence range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    segment_ids: list[int] = Field(default_factory=list)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def end_must_follow_start(self) -> EvidenceItem:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class EvidenceReference(BaseModel):
    """LLM-facing evidence that cites transcript rows instead of calculating time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    segment_ids: list[int] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("segment_ids")
    @classmethod
    def normalize_segment_ids(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("segment_ids must contain positive database IDs")
        return list(dict.fromkeys(values))


class KeyConcept(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    term: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    segment_ids: list[int] = Field(default_factory=list)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @field_validator("term", "explanation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def end_must_follow_start(self) -> KeyConcept:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class KeyConceptReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    term: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    segment_ids: list[int] = Field(min_length=1)

    @field_validator("term", "explanation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("segment_ids")
    @classmethod
    def normalize_segment_ids(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("segment_ids must contain positive database IDs")
        return list(dict.fromkeys(values))


class TermToVerify(BaseModel):
    """A likely transcription error that should be checked against the audio or a source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_form: str = Field(min_length=1)
    suggested_form: str | None = None
    reason: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @field_validator("transcript_form", "reason")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("suggested_form")
    @classmethod
    def normalize_suggested_form(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def end_must_follow_start(self) -> TermToVerify:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    key_claims: list[EvidenceItem] = Field(default_factory=list)
    key_concepts: list[KeyConcept] = Field(default_factory=list)
    uncertainties_and_debates: list[EvidenceItem] = Field(default_factory=list)
    terms_to_verify: list[TermToVerify] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class CheckpointResponse(BaseModel):
    """Small LLM-facing schema for one transcript chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    notes: list[str] = Field(min_length=1)


class FinalSessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_summary: str = Field(min_length=1)
    important_notes: list[str] = Field(default_factory=list)
    main_topics: list[str] = Field(default_factory=list)
    key_claims: list[EvidenceItem] = Field(default_factory=list)
    key_concepts: list[KeyConcept] = Field(default_factory=list)
    uncertainties_and_debates: list[EvidenceItem] = Field(default_factory=list)
    terms_to_verify: list[TermToVerify] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class FinalSessionSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_summary: str = Field(min_length=1)
    important_notes: list[str] = Field(min_length=1)
    main_topics: list[str] = Field(min_length=1)


ContentMode = Literal["auto", "informational", "meeting"]


class GeneratedCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    checkpoint: Checkpoint
    raw_response: str
    generation_seconds: float | None = Field(default=None, ge=0)
    attempt_seconds: tuple[float, ...] = ()

    @model_validator(mode="after")
    def end_must_follow_start(self) -> GeneratedCheckpoint:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    @field_validator("attempt_seconds")
    @classmethod
    def attempts_must_be_non_negative(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value < 0 for value in values):
            raise ValueError("attempt_seconds values must be non-negative")
        return values


class GeneratedSummaryBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoints: tuple[GeneratedCheckpoint, ...]
    final_summary: FinalSessionSummary
    final_raw_response: str
    final_generation_seconds: float = Field(ge=0)
    final_attempt_seconds: tuple[float, ...]
    processing_seconds: float = Field(ge=0)

    @field_validator("final_attempt_seconds")
    @classmethod
    def attempts_must_be_non_negative(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values:
            raise ValueError("final_attempt_seconds must contain at least one attempt")
        if any(value < 0 for value in values):
            raise ValueError("final_attempt_seconds values must be non-negative")
        return values

    @property
    def retry_count(self) -> int:
        checkpoint_retries = sum(
            max(0, len(item.attempt_seconds) - 1) for item in self.checkpoints
        )
        return checkpoint_retries + max(0, len(self.final_attempt_seconds) - 1)

    @property
    def retry_processing_seconds(self) -> float:
        checkpoint_retry_seconds = sum(
            sum(item.attempt_seconds[1:]) for item in self.checkpoints
        )
        return checkpoint_retry_seconds + sum(self.final_attempt_seconds[1:])


class StoredSummaryCheckpoint(Checkpoint):
    id: int
    summary_run_id: int
    session_id: int
    chunk_index: int
    start_seconds: float
    end_seconds: float
    model_name: str
    generation_seconds: float | None = None
    attempt_seconds: tuple[float, ...] = ()
    created_at: datetime


class StoredFinalSummary(FinalSessionSummary):
    id: int
    session_id: int
    label: str | None = None
    model_name: str
    chunk_seconds: float | None = None
    content_mode: ContentMode = "auto"
    checkpoint_count: int
    processing_seconds: float | None = None
    final_generation_seconds: float | None = None
    final_attempt_seconds: tuple[float, ...] = ()
    created_at: datetime


class StoredSummaryRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    session_id: int
    label: str | None = None
    model_name: str
    chunk_seconds: float | None = None
    content_mode: ContentMode = "auto"
    checkpoint_count: int
    processing_seconds: float | None = None
    final_generation_seconds: float | None = None
    created_at: datetime
