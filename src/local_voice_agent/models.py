from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> TranscriptSegment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    segments: tuple[TranscriptSegment, ...]
    duration_seconds: float | None = Field(default=None, ge=0)
    detected_language: str | None = None
    detected_language_probability: float | None = Field(default=None, ge=0, le=1)


class Session(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    source_audio_path: Path
    status: SessionStatus
    language: str
    transcription_model: str
    created_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    detected_language: str | None = None
    detected_language_probability: float | None = None
    error_message: str | None = None


class StoredTranscriptSegment(TranscriptSegment):
    id: int
    session_id: int

