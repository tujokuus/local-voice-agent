from __future__ import annotations

import json

from local_voice_agent.summaries.chunking import TranscriptChunk
from local_voice_agent.summaries.models import (
    Checkpoint,
    CheckpointResponse,
    ContentMode,
    FinalSessionSummaryResponse,
)


CHECKPOINT_SYSTEM_PROMPT = """Create clear notes from one chronological chunk of an English
transcript. Cover the whole chunk, use only information present in it, and avoid repetition. Return
one concise paragraph in summary and the most important details as notes. Do not add citations,
timestamps, segment IDs, decisions, action items, terminology corrections, or unsupported facts.
Write concise English. The response must match the supplied JSON schema exactly."""


FINAL_SUMMARY_SYSTEM_PROMPT = """Combine chronological checkpoint notes into useful notes for one
English recording. Use only the supplied checkpoint content. Return a coherent overall summary, a
deduplicated list of the most important notes, and a short list of main topics. Represent the whole
recording rather than only its beginning or end. Do not add citations, timestamps, segment IDs,
decisions, action items, terminology corrections, or unsupported facts. Write concise English. The
response must match the supplied JSON schema exactly."""


def checkpoint_user_prompt(
    chunk: TranscriptChunk,
    previous_checkpoint: Checkpoint | None,
    content_mode: ContentMode,
) -> str:
    previous_context = (
        json.dumps(
            {
                "summary": previous_checkpoint.summary,
                "notes": previous_checkpoint.notes,
            },
            ensure_ascii=False,
            indent=2,
        )
        if previous_checkpoint is not None
        else "None. This is the first checkpoint."
    )
    schema = json.dumps(CheckpointResponse.model_json_schema(), ensure_ascii=False)
    return f"""Create checkpoint {chunk.index + 1} for transcript time range
{_timestamp(chunk.start_seconds)}-{_timestamp(chunk.end_seconds)}.

Source type: {_content_mode_description(content_mode)}

The previous checkpoint is context for continuity only. Summarize the new transcript chunk, not the
previous checkpoint:
{previous_context}

New transcript chunk (primary evidence):
{chunk.transcript_text}

Return JSON matching this schema:
{schema}"""


def final_summary_user_prompt(
    checkpoints: list[Checkpoint], content_mode: ContentMode
) -> str:
    checkpoint_json = json.dumps(
        [
            {
                "checkpoint": index + 1,
                "summary": checkpoint.summary,
                "notes": checkpoint.notes,
            }
            for index, checkpoint in enumerate(checkpoints)
        ],
        ensure_ascii=False,
        indent=2,
    )
    schema = json.dumps(FinalSessionSummaryResponse.model_json_schema(), ensure_ascii=False)
    return f"""Create the final session summary from these chronological checkpoints.

Source type: {_content_mode_description(content_mode)}

Chronological checkpoints:
{checkpoint_json}

Return JSON matching this schema:
{schema}"""


def _content_mode_description(content_mode: ContentMode) -> str:
    if content_mode == "informational":
        return "informational or educational material"
    if content_mode == "meeting":
        return "a meeting, summarized here as general notes"
    return "an unspecified recording"


def _timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
