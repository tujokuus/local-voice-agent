"""Validated human-authored gold data and a session preflight without model calls."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from local_voice_agent.models import StoredTranscriptSegment

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveId = Annotated[int, Field(strict=True, gt=0)]


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GoldEvidenceRange(DatasetModel):
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    end_seconds: float = Field(gt=0, allow_inf_nan=False)
    segment_ids: list[PositiveId] = Field(min_length=1)
    explanation: NonEmptyText

    @model_validator(mode="after")
    def validate_range(self) -> GoldEvidenceRange:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if len(set(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("gold segment_ids must not contain duplicates")
        return self


class EvaluationCase(DatasetModel):
    id: NonEmptyText
    question: NonEmptyText
    answerable: bool
    category: Literal["focused", "synthesis", "unanswerable"]
    difficulty: Literal["easy", "medium", "hard"]
    reference_answer: NonEmptyText
    required_points: list[NonEmptyText]
    gold_evidence_ranges: list[GoldEvidenceRange]
    tags: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_answerability(self) -> EvaluationCase:
        if self.answerable != bool(self.gold_evidence_ranges):
            raise ValueError("answerable cases require gold evidence; unanswerable cases forbid it")
        if self.answerable == (self.category == "unanswerable"):
            raise ValueError("category must agree with answerable")
        if self.answerable and not self.required_points:
            raise ValueError("answerable cases require required_points for human review")
        bounds = [(item.start_seconds, item.end_seconds) for item in self.gold_evidence_ranges]
        if len(set(bounds)) != len(bounds):
            raise ValueError("gold evidence ranges must not contain duplicates")
        if bounds != sorted(bounds):
            raise ValueError("gold evidence ranges must be ordered by timestamp")
        return self


class EvaluationDataset(DatasetModel):
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    id: NonEmptyText
    description: NonEmptyText
    source_session_id: PositiveId
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> EvaluationDataset:
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("dataset contains duplicate case IDs")
        return self


def load_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(path.read_bytes())


def _boundary_value(value: float, gold_boundary: float) -> float:
    # CLI transcript timestamps are floored to whole seconds. Only preflight uses
    # this normalization; metric intervals always retain the original timestamps.
    return float(math.floor(value)) if gold_boundary.is_integer() else value


def validate_dataset_session(
    dataset: EvaluationDataset,
    transcript: list[StoredTranscriptSegment],
    *,
    session_id: int,
    remap_segment_ids: bool = False,
) -> dict[str, list[list[int]]]:
    """Resolve all gold rows, rejecting incomplete or mismatched audit IDs.

    Whole-second gold boundaries match the application's displayed timestamps.
    Opt-in remapping preserves gold timestamps and resolves IDs in another copy
    of the recording. Callers retain both the original dataset and this mapping.
    """

    if not transcript:
        raise ValueError(f"Session {session_id} has no transcript segments")
    if any(segment.session_id != session_id for segment in transcript):
        raise ValueError("transcript contains segments from a different session")
    resolved: dict[str, list[list[int]]] = {}
    for case in dataset.cases:
        resolved[case.id] = []
        for gold in case.gold_evidence_ranges:
            expected = [
                segment for segment in transcript
                if _boundary_value(segment.start_seconds, gold.end_seconds) < gold.end_seconds
                and _boundary_value(segment.end_seconds, gold.start_seconds) > gold.start_seconds
            ]
            context = f"Case {case.id}, gold range {gold.start_seconds:g}-{gold.end_seconds:g}"
            if not expected:
                raise ValueError(
                    f"{context}: no matching transcript segments in session {session_id}"
                )
            start = _boundary_value(min(row.start_seconds for row in expected), gold.start_seconds)
            end = _boundary_value(max(row.end_seconds for row in expected), gold.end_seconds)
            if not (
                math.isclose(start, gold.start_seconds, abs_tol=1e-7, rel_tol=0)
                and math.isclose(end, gold.end_seconds, abs_tol=1e-7, rel_tol=0)
            ):
                raise ValueError(f"{context}: boundaries must cover complete transcript segments")
            ids = [row.id for row in expected]
            if remap_segment_ids and len(ids) != len(gold.segment_ids):
                raise ValueError(f"{context}: segment count differs from the gold transcript")
            if not remap_segment_ids and set(ids) != set(gold.segment_ids):
                raise ValueError(
                    f"{context}: listed segment IDs {gold.segment_ids} do not match all "
                    f"session {session_id} rows {ids}; fix the gold data or explicitly use "
                    "--remap-segment-ids for a timestamp-identical copy of this recording"
                )
            resolved[case.id].append(ids)
    return resolved
