from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from local_voice_agent.models import StoredTranscriptSegment


MAX_SEARCH_RESULTS = 20
MAX_TRANSCRIPT_RANGE_SECONDS = 600
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

# Small, deterministic groups improve lexical retrieval without introducing a
# second model or an embedding dependency. Original query terms still carry
# more ranking weight than these alternatives.
_SYNONYM_GROUPS = (
    frozenset({"write", "writes", "writing", "wrote", "written", "text", "textual"}),
    frozenset({"print", "prints", "printing", "printed", "press"}),
    frozenset(
        {
            "change", "changes", "changed", "changing", "shift", "shifts",
            "shifted", "transform", "transforms", "transformed", "alter",
            "alters", "altered", "effect", "effects", "impact", "impacts",
        }
    ),
    frozenset({"poetry", "poem", "poems", "poetic", "verse", "verses"}),
    frozenset({"reader", "readers", "audience", "audiences"}),
    frozenset({"speak", "speaks", "spoken", "speech", "oral", "orally"}),
    frozenset(
        {
            "perform", "performs", "performed", "performing", "performance",
            "performer", "performers", "recite", "recited", "recitation",
        }
    ),
    frozenset(
        {
            "preserve", "preserves", "preserved", "record", "records",
            "recorded", "fixed", "stable", "permanent",
        }
    ),
)
_SYNONYMS_BY_TERM = {
    term: group - {term}
    for group in _SYNONYM_GROUPS
    for term in group
}


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
    additional_terms: Sequence[str] = (),
) -> list[TranscriptSearchResult]:
    """Return a bounded ranking using exact, synonymous, and contextual terms."""

    if limit < 1 or limit > MAX_SEARCH_RESULTS:
        raise ValueError(f"limit must be between 1 and {MAX_SEARCH_RESULTS}")

    query_terms = _tokenize(query)
    if not query_terms:
        raise ValueError("search query must contain at least one word")

    unique_query_terms = tuple(dict.fromkeys(query_terms))
    expanded_terms = tuple(
        dict.fromkeys(
            synonym
            for term in unique_query_terms
            for synonym in sorted(_SYNONYMS_BY_TERM.get(term, ()))
            if synonym not in unique_query_terms
        )
    )
    contextual_terms = tuple(
        term
        for term in dict.fromkeys(
            token
            for value in additional_terms
            for token in _tokenize(value)
        )
        if term not in unique_query_terms and term not in expanded_terms
    )
    results: list[TranscriptSearchResult] = []
    for segment in segments:
        text_terms = _tokenize(segment.text)
        term_counts = Counter(text_terms)
        matched_query_terms = tuple(
            term for term in unique_query_terms if term_counts[term] > 0
        )
        matched_expanded_terms = tuple(
            term for term in expanded_terms if term_counts[term] > 0
        )
        matched_contextual_terms = tuple(
            term for term in contextual_terms if term_counts[term] > 0
        )
        if not matched_query_terms and not matched_expanded_terms:
            continue
        matched_terms = tuple(
            dict.fromkeys(
                (*matched_query_terms, *matched_expanded_terms, *matched_contextual_terms)
            )
        )

        phrase_matches = _count_phrase_occurrences(text_terms, query_terms)
        all_terms_match = len(matched_query_terms) == len(unique_query_terms)
        score = (
            phrase_matches * 100
            + int(all_terms_match) * 25
            + len(matched_query_terms) * 8
            + sum(term_counts[term] for term in unique_query_terms)
            + len(matched_expanded_terms) * 3
            + len(matched_contextual_terms) * 4
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


def expand_search_terms(text: str) -> tuple[str, ...]:
    """Return normalized query words plus deterministic word-form alternatives."""

    terms = tuple(dict.fromkeys(_tokenize(text)))
    return tuple(
        dict.fromkeys(
            (
                *terms,
                *(
                    synonym
                    for term in terms
                    for synonym in sorted(_SYNONYMS_BY_TERM.get(term, ()))
                ),
            )
        )
    )


def get_transcript_range(
    segments: Sequence[StoredTranscriptSegment],
    *,
    start_seconds: float,
    end_seconds: float,
    maximum_duration_seconds: float = MAX_TRANSCRIPT_RANGE_SECONDS,
) -> list[StoredTranscriptSegment]:
    """Return chronological segments overlapping one bounded time range."""

    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("range start cannot be negative")
    if not math.isfinite(end_seconds) or end_seconds <= start_seconds:
        raise ValueError("range end must be greater than its start")
    if not math.isfinite(maximum_duration_seconds) or maximum_duration_seconds <= 0:
        raise ValueError("maximum range duration must be greater than zero")
    if end_seconds - start_seconds > maximum_duration_seconds:
        raise ValueError(
            "requested range is too long; maximum duration is "
            f"{maximum_duration_seconds:g} seconds"
        )

    return sorted(
        (
            segment
            for segment in segments
            if segment.start_seconds < end_seconds
            and segment.end_seconds > start_seconds
        ),
        key=lambda segment: (segment.start_seconds, segment.index),
    )


def parse_time_value(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into seconds."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("time value cannot be empty")

    parts = normalized.split(":")
    if len(parts) == 1:
        try:
            seconds = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"invalid time value: {value}") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("time value cannot be negative")
        return seconds

    if len(parts) not in (2, 3):
        raise ValueError(f"invalid time value: {value}")
    try:
        leading_parts = [int(part) for part in parts[:-1]]
        seconds = float(parts[-1])
    except ValueError as exc:
        raise ValueError(f"invalid time value: {value}") from exc

    if (
        any(part < 0 for part in leading_parts)
        or not math.isfinite(seconds)
        or not 0 <= seconds < 60
    ):
        raise ValueError(f"invalid time value: {value}")
    if len(parts) == 2:
        minutes = leading_parts[0]
        return minutes * 60 + seconds

    hours, minutes = leading_parts
    if minutes >= 60:
        raise ValueError(f"invalid time value: {value}")
    return hours * 3600 + minutes * 60 + seconds


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
