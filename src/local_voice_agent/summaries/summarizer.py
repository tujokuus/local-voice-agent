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
    CheckpointResponse,
    ContentMode,
    FinalSessionSummary,
    FinalSessionSummaryResponse,
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
        target_chunk_seconds: float = 600,
        minimum_final_chunk_seconds: float = 120,
    ) -> None:
        self.provider = provider
        self.target_chunk_seconds = target_chunk_seconds
        self.minimum_final_chunk_seconds = minimum_final_chunk_seconds

    def summarize(
        self,
        segments: list[StoredTranscriptSegment],
        *,
        content_mode: ContentMode = "auto",
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
                        content=checkpoint_user_prompt(
                            chunk, previous_checkpoint, content_mode
                        ),
                    ),
                ],
                response_model=CheckpointResponse,
                validator=lambda item, number=index: self._validate_checkpoint(
                    item, number
                ),
            )
            checkpoint = Checkpoint(
                summary=response.data.summary,
                notes=response.data.notes,
            )
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
                        [item.checkpoint for item in generated], content_mode
                    ),
                ),
            ],
            response_model=FinalSessionSummaryResponse,
            validator=self._validate_final_summary,
        )
        final_summary = FinalSessionSummary(
            overall_summary=final_response.data.overall_summary,
            important_notes=final_response.data.important_notes,
            main_topics=final_response.data.main_topics,
        )
        return GeneratedSummaryBundle(
            checkpoints=tuple(generated),
            final_summary=final_summary,
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
                            f"Content validation failed: {exc}",
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
                        f"{first_error}. Return one corrected JSON object matching the "
                        "required schema. Use only the supplied source content and make "
                        "every required text field and list non-empty."
                    ),
                ),
            ]
            response = make_attempt(retry_messages)
            return response, tuple(attempt_seconds)

    @staticmethod
    def _validate_checkpoint(checkpoint: CheckpointResponse, number: int) -> None:
        SessionSummarizer._reject_missing_content_claim(
            checkpoint.summary, f"checkpoint {number}"
        )
        if not any(note.strip() for note in checkpoint.notes):
            raise ValueError(f"checkpoint {number} has no useful notes")

    @staticmethod
    def _validate_final_summary(summary: FinalSessionSummaryResponse) -> None:
        SessionSummarizer._reject_missing_content_claim(
            summary.overall_summary, "final summary"
        )
        if not any(note.strip() for note in summary.important_notes):
            raise ValueError("final summary has no important notes")
        if not any(topic.strip() for topic in summary.main_topics):
            raise ValueError("final summary has no main topics")

    @staticmethod
    def _reject_missing_content_claim(text: str, label: str) -> None:
        normalized = text.casefold()
        if any(phrase in normalized for phrase in _MISSING_CONTENT_PHRASES):
            raise ValueError(f"{label} incorrectly claims that source content is missing")
