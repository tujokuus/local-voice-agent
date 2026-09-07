from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from local_voice_agent.agent import (
    AgentStep,
    ForcedFinalAnswerAgent,
    QuestionAnswer,
    RetrievalStep,
    SessionQuestionAgent,
)
from local_voice_agent.llm import StructuredResponse
from local_voice_agent.models import StoredTranscriptSegment
from local_voice_agent.summaries import StoredFinalSummary, StoredSummaryCheckpoint


class FakeProvider:
    model_name = "telemetry-fake"

    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self.responses = iter(responses)
        self.message_history = []

    def structured_chat(self, *, messages, response_model):
        self.message_history.append(list(messages))
        payload = next(self.responses)
        if isinstance(payload, Exception):
            raise payload
        if response_model is AgentStep:
            defaults = {
                "query": None,
                "segment_id": None,
                "answer": None,
                "evidence_insufficient": False,
            }
        elif response_model is RetrievalStep:
            defaults = {
                "query": None,
                "segment_id": None,
                "evidence_insufficient": False,
            }
        else:
            defaults = {
                "claims": [],
                "evidence_insufficient": False,
                "explanation": None,
            }
        response = response_model.model_validate({**defaults, **payload})
        return StructuredResponse(data=response, raw_content=response.model_dump_json())


def segment(
    segment_id: int, index: int, start: float, end: float, text: str
) -> StoredTranscriptSegment:
    return StoredTranscriptSegment(
        id=segment_id,
        session_id=2,
        index=index,
        start_seconds=start,
        end_seconds=end,
        text=text,
    )


class AgentTelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = [
            segment(42, 0, 100.2, 110.8, "Writing fixed poetry."),
            segment(43, 1, 110.8, 120.9, "Absent readers received a stable version."),
            segment(44, 2, 300, 310, "Writing encouraged a distant audience."),
        ]

    def make_agent(self, responses, *, agent_type=SessionQuestionAgent, **kwargs):
        return agent_type(
            provider=FakeProvider(responses),
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            **kwargs,
        )

    def test_question_answer_retains_three_argument_construction(self) -> None:
        result = QuestionAnswer("Existing answer", 1, ("search_transcript",))
        self.assertEqual(result.retrieved_segment_ids, ())
        self.assertEqual(result.cited_segment_ids, ())
        self.assertEqual(result.citation_ranges, ())
        self.assertIsNone(result.evidence_insufficient)

    def test_manual_records_search_rows_context_rows_and_exact_citation_ranges(self) -> None:
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:50].",
                },
            ]
        )

        result = agent.answer("What did writing change?")

        self.assertEqual(result.retrieved_segment_ids, (42, 43, 44))
        self.assertEqual(result.cited_segment_ids, (42,))
        self.assertEqual(result.citation_ranges, ((100.0, 110.0),))
        self.assertIs(result.evidence_insufficient, False)
        self.assertIs(agent.last_telemetry, result)

    def test_manual_citation_maps_to_overlapping_rows_without_boundary_touch(self) -> None:
        self.transcript.append(segment(45, 3, 120, 130, "Boundary segment."))
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:50] "
                    "and reached readers [00:01:50 - 00:02:00].",
                },
            ]
        )

        result = agent.answer("What did writing change?")

        self.assertEqual(result.cited_segment_ids, (42, 43))
        self.assertEqual(result.citation_ranges, ((100.0, 110.0), (110.0, 120.0)))

    def test_manual_subsecond_row_is_cited_without_expanding_rendered_range(self) -> None:
        self.transcript = [segment(42, 0, 100.2, 100.8, "Writing fixed poetry.")]
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:40].",
                },
            ]
        )

        result = agent.answer("What did writing change?")

        self.assertEqual(result.cited_segment_ids, (42,))
        self.assertEqual(result.citation_ranges, ((100.0, 100.0),))

    def test_summary_ranges_do_not_count_as_retrieved_transcript(self) -> None:
        agent = self.make_agent(
            [
                {"action": "get_session_summary"},
                RuntimeError("provider failed"),
            ]
        )
        agent.summary = StoredFinalSummary(
            id=1,
            session_id=2,
            model_name="fake",
            checkpoint_count=1,
            created_at=datetime.now(UTC),
            overall_summary="Writing affected poetry.",
        )
        agent.checkpoints = [
            StoredSummaryCheckpoint(
                id=1,
                summary_run_id=1,
                session_id=2,
                chunk_index=0,
                start_seconds=0,
                end_seconds=400,
                model_name="fake",
                created_at=datetime.now(UTC),
                summary="Writing affected poetry.",
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            agent.answer("What did writing change?")

        self.assertEqual(agent.last_telemetry.tool_calls, ("get_session_summary",))
        self.assertEqual(agent.last_telemetry.retrieved_segment_ids, ())
        self.assertEqual(agent.last_telemetry.cited_segment_ids, ())
        self.assertEqual(agent.last_telemetry.citation_ranges, ())
        self.assertIsNone(agent.last_telemetry.evidence_insufficient)

    def test_truncated_search_and_context_rows_are_excluded(self) -> None:
        for agent_type in (SessionQuestionAgent, ForcedFinalAnswerAgent):
            with self.subTest(agent_type=agent_type.__name__):
                agent = self.make_agent(
                    [
                        {"action": "search_transcript", "query": "writing"},
                        {"action": "get_transcript", "segment_id": 42},
                        RuntimeError("provider failed"),
                    ],
                    agent_type=agent_type,
                )
                with patch("local_voice_agent.agent.MAX_TOOL_RESULT_CHARS", 90):
                    with self.assertRaisesRegex(RuntimeError, "provider failed"):
                        agent.answer("What did writing change?")

                self.assertEqual(agent.last_telemetry.retrieved_segment_ids, (42,))
                self.assertEqual(agent.last_telemetry.step_count, 2)

    def test_failed_and_duplicate_calls_do_not_enter_telemetry(self) -> None:
        for agent_type in (SessionQuestionAgent, ForcedFinalAnswerAgent):
            with self.subTest(agent_type=agent_type.__name__):
                agent = self.make_agent(
                    [
                        {"action": "get_transcript", "segment_id": 999},
                        {"action": "search_transcript", "query": "writing"},
                        {"action": "search_transcript", "query": "writing"},
                        RuntimeError("provider failed"),
                    ],
                    agent_type=agent_type,
                )

                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    agent.answer("What did writing change?")

                self.assertEqual(agent.last_telemetry.tool_calls, ("search_transcript",))
                self.assertEqual(agent.last_telemetry.retrieved_segment_ids, (42, 44))
                self.assertEqual(agent.last_telemetry.step_count, 1)
                self.assertEqual(agent.last_telemetry.answer, "")
                self.assertIsNone(agent.last_telemetry.evidence_insufficient)

    def test_final_records_selected_ids_and_application_rendered_merged_range(self) -> None:
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "finish_retrieval"},
                {
                    "claims": [
                        {"text": "Writing fixed poetry for readers.", "segment_ids": [43, 42]}
                    ]
                },
            ],
            agent_type=ForcedFinalAnswerAgent,
        )

        result = agent.answer("What did writing change?")

        self.assertEqual(result.retrieved_segment_ids, (42, 43, 44))
        self.assertEqual(result.cited_segment_ids, (42, 43))
        self.assertEqual(result.citation_ranges, ((100.0, 120.0),))
        self.assertEqual(
            result.answer,
            "Writing fixed poetry for readers [00:01:40-00:02:00].",
        )
        self.assertIs(result.evidence_insufficient, False)
        self.assertIs(agent.last_telemetry, result)

    def test_search_only_rows_still_cannot_be_cited_in_final_mode(self) -> None:
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "finish_retrieval"},
                {"claims": [{"text": "Distant audience.", "segment_ids": [44]}]},
                {"claims": [{"text": "Poetry became fixed.", "segment_ids": [42]}]},
            ],
            agent_type=ForcedFinalAnswerAgent,
        )

        result = agent.answer("What did writing change?")

        self.assertEqual(result.retrieved_segment_ids, (42, 43, 44))
        self.assertEqual(result.cited_segment_ids, (42,))
        self.assertNotIn("segment_id=44", agent.provider.message_history[3][-1].content)
        self.assertIn("were not retrieved", agent.provider.message_history[4][-1].content)

    def test_final_disjoint_ranges_and_duplicate_claim_sources_match_rendered_text(self) -> None:
        self.transcript[2] = segment(44, 3, 125.4, 130.6, "Writing reached an audience.")
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "finish_retrieval"},
                {
                    "claims": [
                        {"text": "Writing reached an audience.", "segment_ids": [44, 42]},
                        {"text": "Poetry became fixed.", "segment_ids": [42]},
                    ]
                },
            ],
            agent_type=ForcedFinalAnswerAgent,
        )

        result = agent.answer("What did writing change?")

        self.assertEqual(result.retrieved_segment_ids, (42, 43, 44))
        self.assertEqual(result.cited_segment_ids, (42, 44))
        self.assertEqual(result.citation_ranges, ((100.0, 110.0), (125.0, 130.0)))
        self.assertEqual(
            result.answer,
            "Writing reached an audience [00:01:40-00:01:50] [00:02:05-00:02:10]. "
            "Poetry became fixed [00:01:40-00:01:50].",
        )

    def test_insufficient_evidence_is_explicit_in_both_modes(self) -> None:
        for agent_type in (SessionQuestionAgent, ForcedFinalAnswerAgent):
            with self.subTest(agent_type=agent_type.__name__):
                answer = {"evidence_insufficient": True}
                if agent_type is SessionQuestionAgent:
                    answer.update(action="answer", answer="No astronomy evidence was found.")
                else:
                    answer["explanation"] = "No astronomy evidence was found."
                agent = self.make_agent(
                    [
                        {"action": "search_transcript", "query": "astronomy"},
                        {"action": "search_transcript", "query": "galaxies"},
                        answer,
                    ],
                    agent_type=agent_type,
                )

                result = agent.answer("What does the recording say about astronomy?")

                self.assertIs(result.evidence_insufficient, True)
                self.assertEqual(result.retrieved_segment_ids, ())
                self.assertEqual(result.cited_segment_ids, ())
                self.assertEqual(result.citation_ranges, ())

    def test_final_provider_failure_keeps_completed_retrieval_steps(self) -> None:
        agent = self.make_agent(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "finish_retrieval"},
                RuntimeError("final provider failed"),
            ],
            agent_type=ForcedFinalAnswerAgent,
        )

        with self.assertRaisesRegex(RuntimeError, "final provider failed"):
            agent.answer("What did writing change?")

        self.assertEqual(agent.last_telemetry.step_count, 3)
        self.assertEqual(agent.last_telemetry.retrieved_segment_ids, (42, 43, 44))
        self.assertEqual(agent.last_telemetry.cited_segment_ids, ())
        self.assertIsNone(agent.last_telemetry.evidence_insufficient)

    def test_new_answer_clears_telemetry_even_if_validation_fails(self) -> None:
        for agent_type in (SessionQuestionAgent, ForcedFinalAnswerAgent):
            with self.subTest(agent_type=agent_type.__name__):
                agent = self.make_agent([], agent_type=agent_type)
                agent.last_telemetry = QuestionAnswer("Old answer", 1, ())

                with self.assertRaisesRegex(ValueError, "question cannot be empty"):
                    agent.answer(" ")

                self.assertIsNone(agent.last_telemetry)


if __name__ == "__main__":
    unittest.main()
