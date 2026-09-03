from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from local_voice_agent.models import StoredTranscriptSegment


MAX_SEARCH_RESULTS = 20
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TranscriptSearchResult:
    segment_id: int
    segment_index: int
    start_seconds: float
    end_seconds: float
    text: str
    score: int
    matched_terms: tuple[str, ...]


def search_transcript(
    segments: Sequence[StoredTranscriptSegment],
    query: str,
    *,
    limit: int = 5,
) -> list[TranscriptSearchResult]:
    """Return a bounded, deterministic ranking of transcript segments."""

    if limit < 1 or limit > MAX_SEARCH_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")

    query_terms = _tokenize(query)
    if not query_terms:
        raise ValueError("search query must contain at least one word")

    unique_query_terms = tuple(dict.fromkeys(query_terms))
    results: list[TranscriptSearchResult] = []
    for segment in segments:
        text_terms = _tokenize(segment.text)
        term_counts = Counter(text_terms)
        matched_terms = tuple(
            term for term in unique_query_terms if term_counts[term] > 0
        )
        if not matched_terms:
            continue

        phrase_matches = _count_phrase_occurrences(text_terms, query_terms)
        all_terms_match = len(matched_terms) == len(unique_query_terms)
        score = (
            phrase_matches * 100
            + int(all_terms_match) * 25
            + len(matched_terms) * 5
            + sum(term_counts[term] for term in unique_query_terms)
        )
        results.append(
            TranscriptSearchResult(
                segment_id=segment.id,
                segment_index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text,
                score=score,
                matched_terms=matched_terms,
            )
        )

    results.sort(key=lambda item: (-item.score, item.start_seconds, item.segment_id))
    return results[:limit]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def _count_phrase_occurrences(text: list[str], phrase: list[str]) -> int:
    if len(phrase) > len(text):
        return 0
    width = len(phrase)
    return sum(
        text[index : index + width] == phrase
        for index in range(len(text) - width + 1)
    )
