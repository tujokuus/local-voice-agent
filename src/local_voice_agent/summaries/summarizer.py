from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

from pydantic import BaseModel

from local_voice_agent.llm import (
    ChatMessage,
    LLMProvider,
    StructuredOutputError,
    StructuredResponse,
)
from local_voice_agent.models import StoredTranscriptSegment
from local_voice_agent.summaries.chunking import chunk_transcript
from local_voice_agent.summaries.models import (
    Checkpoint,
    FinalSessionSummary,
    GeneratedCheckpoint,
    GeneratedSummaryBundle,
)
from local_voice_agent.summaries.prompts import (
    CHECKPOINT_SYSTEM_PROMPT,
    FINAL_SUMMARY_SYSTEM_PROMPT,
    checkpoint_user_prompt,
    final_summary_user_prompt,
)


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
ProgressCallback = Callable[[int, int], None]
FinalProgressCallback = Callable[[], None]
ResponseValidator = Callable[[ResponseModel], None]

_TIMESTAMP_TOLERANCE_SECONDS = 1.0
_MISSING_CONTENT_PHRASES = (
    "no transcript",
    "transcript was not provided",
    "transcript chunk was not provided",
    "no content was provided",
    "cannot analyze",
)


class SessionSummarizer:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        target_chunk_seconds: float = 300,
        minimum_final_chunk_seconds: float = 120,
    ) -> None:
        self.provider = provider
        self.target_chunk_seconds = target_chunk_seconds
        self.minimum_final_chunk_seconds = minimum_final_chunk_seconds

    def summarize(
        self,
        segments: list[StoredTranscriptSegment],
        *,
        progress: ProgressCallback | None = None,
        final_progress: FinalProgressCallback | None = None,
    ) -> GeneratedSummaryBundle:
        chunks = chunk_transcript(
            segments,
            target_seconds=self.target_chunk_seconds,
            minimum_final_seconds=self.minimum_final_chunk_seconds,
        )
        if not chunks:
            raise ValueError("Cannot summarize a session without transcript segments")

        started_at = perf_counter()
        generated: list[GeneratedCheckpoint] = []
        previous_checkpoint: Checkpoint | None = None

        for index, chunk in enumerate(chunks, start=1):
            if progress is not None:
                progress(index, len(chunks))
            response, attempt_seconds = self._structured_with_one_retry(
                messages=[
                    ChatMessage(role="system", content=CHECKPOINT_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=checkpoint_user_prompt(chunk, previous_checkpoint),
                    ),
                ],
                response_model=Checkpoint,
                validator=lambda checkpoint, chunk=chunk: self._validate_structured_summary(
                    checkpoint,
                    evidence_start=chunk.start_seconds,
                    evidence_end=chunk.end_seconds,
                    label=f"checkpoint {chunk.index + 1}",
                ),
            )
            checkpoint = response.data
            generated.append(
                GeneratedCheckpoint(
                    chunk_index=chunk.index,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    checkpoint=checkpoint,
                    raw_response=response.raw_content,
                    generation_seconds=sum(attempt_seconds),
                    attempt_seconds=attempt_seconds,
                )
            )
            previous_checkpoint = checkpoint

        if final_progress is not None:
            final_progress()
        final_response, final_attempt_seconds = self._structured_with_one_retry(
            messages=[
                ChatMessage(role="system", content=FINAL_SUMMARY_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=final_summary_user_prompt(
                        [item.checkpoint for item in generated]
                    ),
                ),
            ],
            response_model=FinalSessionSummary,
            validator=lambda summary: self._validate_structured_summary(
                summary,
                evidence_start=chunks[0].start_seconds,
                evidence_end=chunks[-1].end_seconds,
                label="final summary",
            ),
        )
        return GeneratedSummaryBundle(
            checkpoints=tuple(generated),
            final_summary=final_response.data,
            final_raw_response=final_response.raw_content,
            final_generation_seconds=sum(final_attempt_seconds),
            final_attempt_seconds=final_attempt_seconds,
            processing_seconds=perf_counter() - started_at,
        )

    def _structured_with_one_retry(
        self,
        *,
        messages: list[ChatMessage],
        response_model: type[ResponseModel],
        validator: ResponseValidator[ResponseModel] | None = None,
    ) -> tuple[StructuredResponse[ResponseModel], tuple[float, ...]]:
        attempt_seconds: list[float] = []

        def make_attempt(
            attempt_messages: list[ChatMessage],
        ) -> StructuredResponse[ResponseModel]:
            attempt_started_at = perf_counter()
            try:
                response = self.provider.structured_chat(
                    messages=attempt_messages,
                    response_model=response_model,
                )
                if validator is not None:
                    try:
                        validator(response.data)
                    except ValueError as exc:
                        raise StructuredOutputError(
                            f"Context validation failed: {exc}",
                            raw_content=response.raw_content,
                        ) from exc
                return response
            finally:
                attempt_seconds.append(perf_counter() - attempt_started_at)

        try:
            response = make_attempt(messages)
            return response, tuple(attempt_seconds)
        except StructuredOutputError as first_error:
            repair_context = first_error.raw_content or "No usable JSON was returned."
            retry_messages = [
                *messages,
                ChatMessage(role="assistant", content=repair_context),
                ChatMessage(
                    role="user",
                    content=(
                        "The previous response was rejected. The validation error was: "
                        f"{first_error}. Return one corrected JSON object matching the required "
                        "schema and transcript evidence exactly. Re-read the new transcript chunk, "
                        "use absolute seconds, and do not repeat the rejected mistake."
                    ),
                ),
            ]
            response = make_attempt(retry_messages)
            return response, tuple(attempt_seconds)

    @staticmethod
    def _validate_structured_summary(
        summary: Checkpoint | FinalSessionSummary,
        *,
        evidence_start: float,
        evidence_end: float,
        label: str,
    ) -> None:
        normalized_summary = (
            summary.summary if isinstance(summary, Checkpoint) else summary.overall_summary
        ).casefold()
        if any(phrase in normalized_summary for phrase in _MISSING_CONTENT_PHRASES):
            raise ValueError(f"{label} incorrectly claims that transcript content is missing")

        has_substantive_content = any(
            (
                summary.key_claims,
                summary.key_concepts,
                summary.uncertainties_and_debates,
                summary.decisions,
                summary.action_items,
                summary.open_questions,
            )
        )
        if not has_substantive_content:
            raise ValueError(f"{label} contains no substantive extracted items")

        timestamped_items = (
            *summary.key_claims,
            *summary.key_concepts,
            *summary.uncertainties_and_debates,
            *summary.terms_to_verify,
        )
        for item in timestamped_items:
            if item.start_seconds < evidence_start - _TIMESTAMP_TOLERANCE_SECONDS:
                raise ValueError(
                    f"{label} timestamp {item.start_seconds:g} starts before its "
                    f"evidence range {evidence_start:g}-{evidence_end:g} seconds"
                )
            if item.end_seconds > evidence_end + _TIMESTAMP_TOLERANCE_SECONDS:
                raise ValueError(
                    f"{label} timestamp {item.end_seconds:g} ends after its "
                    f"evidence range {evidence_start:g}-{evidence_end:g} seconds"
                )
