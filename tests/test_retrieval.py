from __future__ import annotations

import unittest

from local_voice_agent.models import StoredTranscriptSegment
from local_voice_agent.retrieval import search_transcript


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


if __name__ == "__main__":
    unittest.main()
