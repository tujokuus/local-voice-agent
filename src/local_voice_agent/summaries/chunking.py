from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from local_voice_agent.models import StoredTranscriptSegment


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    index: int
    start_seconds: float
    end_seconds: float
    segments: tuple[StoredTranscriptSegment, ...]

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def transcript_text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


def chunk_transcript(
    segments: Sequence[StoredTranscriptSegment],
    *,
    target_seconds: float = 600,
    minimum_final_seconds: float = 120,
    sentence_boundary_window_seconds: float = 30,
) -> list[TranscriptChunk]:
    """Group complete transcript segments into approximately sized chunks.

    A boundary near the target prefers a transcript segment that appears to end a
    sentence. The search window is bounded, so missing punctuation cannot create
    an excessively large chunk. A short final remainder is merged backward.
    """

    if target_seconds <= 0:
        raise ValueError("target_seconds must be greater than zero")
    if minimum_final_seconds < 0:
        raise ValueError("minimum_final_seconds cannot be negative")
    if sentence_boundary_window_seconds < 0:
        raise ValueError("sentence_boundary_window_seconds cannot be negative")
    if not segments:
        return []

    ordered = sorted(
        segments, key=lambda segment: (segment.start_seconds, segment.index)
    )
    groups: list[list[StoredTranscriptSegment]] = []
    cursor = 0
    while cursor < len(ordered):
        remaining = ordered[cursor:]
        current_start = remaining[0].start_seconds
        if remaining[-1].end_seconds <= current_start + target_seconds:
            groups.append(remaining)
            break

        target = current_start + target_seconds
        boundary_offset = _choose_boundary_offset(
            remaining,
            target_seconds=target,
            window_seconds=sentence_boundary_window_seconds,
        )
        boundary = cursor + boundary_offset + 1
        groups.append(ordered[cursor:boundary])
        cursor = boundary

    if len(groups) > 1:
        final_group = groups[-1]
        final_duration = final_group[-1].end_seconds - final_group[0].start_seconds
        if final_duration < minimum_final_seconds:
            groups[-2].extend(groups.pop())

    return [
        TranscriptChunk(
            index=index,
            start_seconds=group[0].start_seconds,
            end_seconds=group[-1].end_seconds,
            segments=tuple(group),
        )
        for index, group in enumerate(groups)
    ]


def _choose_boundary_offset(
    segments: Sequence[StoredTranscriptSegment],
    *,
    target_seconds: float,
    window_seconds: float,
) -> int:
    lower = target_seconds - window_seconds
    upper = target_seconds + window_seconds
    sentence_candidates = [
        (index, segment)
        for index, segment in enumerate(segments)
        if lower <= segment.end_seconds <= upper and _ends_sentence(segment.text)
    ]
    if sentence_candidates:
        return min(
            sentence_candidates,
            key=lambda item: (abs(item[1].end_seconds - target_seconds), item[0]),
        )[0]

    return min(
        enumerate(segments),
        key=lambda item: (abs(item[1].end_seconds - target_seconds), item[0]),
    )[0]


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r'[.!?][\"\')\]]*$', text.strip()))


def _timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
