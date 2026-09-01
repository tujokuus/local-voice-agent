from __future__ import annotations

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
        return "\n".join(
            f"[{_timestamp(segment.start_seconds)}-{_timestamp(segment.end_seconds)}] "
            f"{segment.text}"
            for segment in self.segments
        )


def chunk_transcript(
    segments: Sequence[StoredTranscriptSegment],
    *,
    target_seconds: float = 300,
    minimum_final_seconds: float = 120,
) -> list[TranscriptChunk]:
    """Group complete transcript segments into approximately sized chunks.

    Boundaries follow segment timestamps rather than splitting recognized text at
    an exact clock value. A short final remainder is merged into the previous
    chunk so it does not become a low-context checkpoint on its own.
    """

    if target_seconds <= 0:
        raise ValueError("target_seconds must be greater than zero")
    if minimum_final_seconds < 0:
        raise ValueError("minimum_final_seconds cannot be negative")
    if not segments:
        return []

    ordered = sorted(segments, key=lambda segment: (segment.start_seconds, segment.index))
    groups: list[list[StoredTranscriptSegment]] = []
    current: list[StoredTranscriptSegment] = []
    current_start = ordered[0].start_seconds

    for segment in ordered:
        if current and segment.start_seconds >= current_start + target_seconds:
            groups.append(current)
            current = []
            current_start = segment.start_seconds
        current.append(segment)

    if current:
        groups.append(current)

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


def _timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

