from __future__ import annotations

import json

from local_voice_agent.summaries.chunking import TranscriptChunk
from local_voice_agent.summaries.models import Checkpoint, FinalSessionSummary


CHECKPOINT_SYSTEM_PROMPT = """You analyze one chronological chunk of an English session transcript.
Use only claims supported by the supplied transcript. Distinguish discussion from actual decisions,
proposals from assigned action items, and uncertainty from facts. Never invent an owner, decision,
action item, or open question. Use empty arrays when a category has no supported items. Write concise
English. The response must match the supplied JSON schema exactly."""


FINAL_SUMMARY_SYSTEM_PROMPT = """You synthesize structured checkpoints from one English session.
Use only the checkpoint evidence. Deduplicate repeated topics and items, preserve uncertainty, and do
not promote discussion or suggestions into decisions or action items. Use empty arrays when needed.
Write concise English. The response must match the supplied JSON schema exactly."""


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

The previous checkpoint is context only. Do not copy its items unless the new transcript supports
continuation or resolution:
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

