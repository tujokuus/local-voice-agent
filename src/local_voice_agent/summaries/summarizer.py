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
            response = self._structured_with_one_retry(
                messages=[
                    ChatMessage(role="system", content=CHECKPOINT_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=checkpoint_user_prompt(chunk, previous_checkpoint),
                    ),
                ],
                response_model=Checkpoint,
            )
            checkpoint = response.data
            generated.append(
                GeneratedCheckpoint(
                    chunk_index=chunk.index,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    checkpoint=checkpoint,
                    raw_response=response.raw_content,
                    generation_seconds=response.total_duration_seconds,
                )
            )
            previous_checkpoint = checkpoint

        if final_progress is not None:
            final_progress()
        final_response = self._structured_with_one_retry(
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
        )
        return GeneratedSummaryBundle(
            checkpoints=tuple(generated),
            final_summary=final_response.data,
            final_raw_response=final_response.raw_content,
            processing_seconds=perf_counter() - started_at,
        )

    def _structured_with_one_retry(
        self,
        *,
        messages: list[ChatMessage],
        response_model: type[ResponseModel],
    ) -> StructuredResponse[ResponseModel]:
        try:
            return self.provider.structured_chat(
                messages=messages,
                response_model=response_model,
            )
        except StructuredOutputError as first_error:
            repair_context = first_error.raw_content or "No usable JSON was returned."
            retry_messages = [
                *messages,
                ChatMessage(role="assistant", content=repair_context),
                ChatMessage(
                    role="user",
                    content=(
                        "The previous response failed schema validation. Return one corrected "
                        "JSON object matching the required schema exactly."
                    ),
                ),
            ]
            return self.provider.structured_chat(
                messages=retry_messages,
                response_model=response_model,
            )
