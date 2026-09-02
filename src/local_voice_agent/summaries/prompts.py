from __future__ import annotations

import json

from local_voice_agent.summaries.chunking import TranscriptChunk
from local_voice_agent.summaries.models import Checkpoint, FinalSessionSummary


CHECKPOINT_SYSTEM_PROMPT = """You analyze one chronological chunk of an English session
transcript. Use only claims supported by the new transcript chunk. Extract its most important claims
and concepts, record genuine uncertainty or debate, and flag likely transcription errors in
specialized terms or proper names. Give each extracted item the narrowest supporting start_seconds
and end_seconds found in the transcript timestamps. Timestamp fields are absolute seconds from the
beginning of the recording, not clock strings, decimal clock values, or seconds relative to the
current chunk. For example, 00:07:19 is start_seconds 439 and 00:07:25 is end_seconds 445. Every
timestamp must fall within the current checkpoint range. Do not guess a corrected term:
suggested_form must be null unless the correction is reasonably clear from context. Distinguish
discussion from actual decisions, proposals from assigned action items, and uncertainty from facts.
Never invent an owner, decision, action item, open question, claim, concept, correction, or
timestamp.
The transcript chunk is always present after the 'New transcript chunk' heading; do not claim it was
omitted. Use empty arrays only for individual categories with no supported items, not for the entire
checkpoint. Write concise English. The response must match the supplied JSON schema exactly."""


FINAL_SUMMARY_SYSTEM_PROMPT = """You synthesize structured checkpoints from one English session.
Use only the checkpoint evidence. Select the most important claims and concepts, retain their
original supporting timestamps, deduplicate repeated items, preserve uncertainty and debates, and
retain useful term-verification warnings. Do not promote discussion or suggestions into decisions or
action items. Never invent evidence or widen an evidence range beyond the supplied checkpoint items.
Use empty arrays when needed. Write concise English. The response must match the supplied JSON
schema exactly."""


def checkpoint_user_prompt(
    chunk: TranscriptChunk,
    previous_checkpoint: Checkpoint | None,
) -> str:
    previous_context = (
        json.dumps(previous_checkpoint.model_dump(), ensure_ascii=False, indent=2)
        if previous_checkpoint is not None
        else "None. This is the first checkpoint."
    )
    schema = json.dumps(Checkpoint.model_json_schema(), ensure_ascii=False)
    return f"""Create checkpoint {chunk.index + 1} for transcript time range
{_timestamp(chunk.start_seconds)}-{_timestamp(chunk.end_seconds)}.

The previous checkpoint is context only. Every field in the new checkpoint must describe evidence in
the new transcript time range. Do not restate definitions, claims, topics, or other items merely
because they appear in the previous checkpoint. Include a previous item only when the new transcript
explicitly continues, changes, challenges, or resolves it:
{previous_context}

New transcript chunk (primary evidence):
{chunk.transcript_text}

Return JSON matching this schema:
{schema}"""


def final_summary_user_prompt(checkpoints: list[Checkpoint]) -> str:
    checkpoint_json = json.dumps(
        [checkpoint.model_dump() for checkpoint in checkpoints],
        ensure_ascii=False,
        indent=2,
    )
    schema = json.dumps(FinalSessionSummary.model_json_schema(), ensure_ascii=False)
    return f"""Create the final session summary from these chronological checkpoints:
{checkpoint_json}

Return JSON matching this schema:
{schema}"""


def _timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
