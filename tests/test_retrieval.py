from __future__ import annotations

import unittest

from local_voice_agent.models import StoredTranscriptSegment
from local_voice_agent.retrieval import (
    get_transcript_range,
    parse_time_value,
    search_transcript,
)


class TranscriptSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        texts = (
            "Poetry began in oral traditions and song.",
            "Writing changed how poetry was preserved.",
            "Oral tradition helped performers remember long poems.",
        )
        self.segments = [
            StoredTranscriptSegment(
                id=index + 10,
                session_id=2,
                index=index,
                start_seconds=index * 10,
                end_seconds=(index + 1) * 10,
                text=text,
            )
            for index, text in enumerate(texts)
        ]

    def test_exact_phrase_ranks_before_separate_term_matches(self) -> None:
        results = search_transcript(self.segments, "oral tradition")

        self.assertEqual([item.segment_id for item in results], [12, 10])
        self.assertEqual(results[0].matched_terms, ("oral", "tradition"))

    def test_search_is_case_insensitive_and_bounded(self) -> None:
        results = search_transcript(self.segments, "POETRY", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].segment_id, 10)

    def test_no_match_returns_empty_list(self) -> None:
        self.assertEqual(search_transcript(self.segments, "astronomy"), [])

    def test_invalid_query_and_limit_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            search_transcript(self.segments, "---")
        with self.assertRaises(ValueError):
            search_transcript(self.segments, "poetry", limit=21)

    def test_range_returns_overlapping_segments_in_time_order(self) -> None:
        results = get_transcript_range(
            list(reversed(self.segments)),
            start_seconds=9,
            end_seconds=21,
        )

        self.assertEqual([item.id for item in results], [10, 11, 12])

    def test_range_rejects_invalid_or_oversized_intervals(self) -> None:
        with self.assertRaises(ValueError):
            get_transcript_range(self.segments, start_seconds=10, end_seconds=10)
        with self.assertRaises(ValueError):
            get_transcript_range(self.segments, start_seconds=0, end_seconds=601)

    def test_time_parser_supports_documented_formats(self) -> None:
        self.assertEqual(parse_time_value("90"), 90)
        self.assertEqual(parse_time_value("01:30"), 90)
        self.assertEqual(parse_time_value("00:01:30"), 90)
        with self.assertRaises(ValueError):
            parse_time_value("00:61:00")
        with self.assertRaises(ValueError):
            parse_time_value("nan")


if __name__ == "__main__":
    unittest.main()
