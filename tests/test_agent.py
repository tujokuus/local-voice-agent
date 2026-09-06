from __future__ import annotations

import unittest
from datetime import UTC, datetime

from local_voice_agent.agent import (
    AgentStep,
    FinalAnswer,
    ForcedFinalAnswerAgent,
    RetrievalStep,
    SessionQuestionAgent,
)
from local_voice_agent.llm import StructuredResponse
from local_voice_agent.models import StoredTranscriptSegment
from local_voice_agent.summaries import StoredFinalSummary


class FakeProvider:
    model_name = "fake"

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls = 0
        self.message_history = []
        self.response_models = []

    def structured_chat(self, *, messages, response_model):
        self.message_history.append(list(messages))
        self.response_models.append(response_model)
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
        payload = {**defaults, **self.responses[self.calls]}
        response = response_model.model_validate(payload)
        self.calls += 1
        return StructuredResponse(
            data=response,
            raw_content=response.model_dump_json(),
        )


class SessionQuestionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = [
            StoredTranscriptSegment(
                id=42,
                session_id=2,
                index=0,
                start_seconds=100,
                end_seconds=110,
                text="Writing fixed poetry into versions for absent readers.",
            )
        ]

    def test_agent_searches_before_returning_cited_answer(self) -> None:
        provider = FakeProvider(
            [
                {
                    "action": "search_transcript",
                    "query": "writing poetry",
                },
                {
                    "action": "get_transcript",
                    "segment_id": 42,
                },
                {
                    "action": "answer",
                    "answer": (
                        "Writing fixed poetry for absent readers "
                        "[00:01:40-00:01:50]."
                    ),
                },
            ]
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
        )

        result = agent.answer("What changed when poetry was written down?")

        self.assertEqual(result.step_count, 3)
        self.assertEqual(
            result.tool_calls,
            ("search_transcript", "get_transcript"),
        )
        self.assertIn("[00:01:40-00:01:50]", result.answer)
        self.assertIn(
            "Original question (do not change)",
            provider.message_history[1][-1].content,
        )

    def test_agent_rejects_answer_before_any_tool_call(self) -> None:
        provider = FakeProvider(
            [
                {"action": "answer", "answer": "An unsupported answer."},
                {"action": "search_transcript", "query": "writing"},
                {
                    "action": "get_transcript",
                    "segment_id": 42,
                },
                {
                    "action": "answer",
                    "answer": "Writing fixed the poem [00:01:40-00:01:50].",
                },
            ]
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=4,
        )

        result = agent.answer("What did writing do?")

        self.assertEqual(result.step_count, 3)
        self.assertEqual(provider.calls, 4)

    def test_agent_rejects_invented_timestamp(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing"},
                {
                    "action": "get_transcript",
                    "segment_id": 42,
                },
                {
                    "action": "answer",
                    "answer": "Writing changed poetry [00:00:00-00:00:05].",
                },
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:50].",
                },
            ]
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=4,
        )

        result = agent.answer("How did writing affect poetry?")

        self.assertEqual(result.step_count, 3)
        self.assertIn("[00:01:40-00:01:50]", result.answer)

    def test_rejected_answer_reason_is_traced_without_using_step_budget(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 42},
                {
                    "action": "answer",
                    "answer": "Writing changed poetry [00:00:00-00:00:05].",
                },
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:50].",
                },
            ]
        )
        trace_messages: list[str] = []
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=3,
        )

        result = agent.answer(
            "How did writing affect poetry?", trace=trace_messages.append
        )

        self.assertEqual(result.step_count, 3)
        self.assertIn("Answer repair 1/3: rejected", trace_messages[2])
        self.assertIn("timestamps not returned by tools", trace_messages[2])
        self.assertEqual(trace_messages[-1], "Step 3: answer evidence_insufficient=False")

    def test_insufficient_answer_requires_alternative_second_search(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "unmatched first"},
                {
                    "action": "answer",
                    "answer": "The first search found no evidence.",
                    "evidence_insufficient": True,
                },
                {"action": "search_transcript", "query": "unmatched alternative"},
                {
                    "action": "answer",
                    "answer": "Two searches found no evidence.",
                    "evidence_insufficient": True,
                },
            ]
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=3,
        )

        result = agent.answer("What does it say about astronomy?")

        self.assertEqual(result.step_count, 3)
        self.assertEqual(
            result.tool_calls,
            ("search_transcript", "search_transcript"),
        )
        self.assertEqual(provider.calls, 4)

    def test_session_summary_is_supplied_as_search_orientation(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing fixed readers"},
                {"action": "get_transcript", "segment_id": 42},
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:50].",
                },
            ]
        )
        summary = StoredFinalSummary(
            id=7,
            session_id=2,
            model_name="fake",
            checkpoint_count=1,
            created_at=datetime.now(UTC),
            overall_summary=(
                "Writing fixed fluid performances into content for absent readers."
            ),
            important_notes=["Printing later changed visual presentation."],
            main_topics=["Oral and written poetry"],
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=summary,
            checkpoints=[],
        )

        agent.answer("How did writing change poetry?")

        initial_prompt = provider.message_history[0][-1].content
        self.assertIn("session-summary orientation", initial_prompt)
        self.assertIn("absent readers", initial_prompt)
        self.assertIn("Potential transcript vocabulary", initial_prompt)

    def test_step_schema_rejects_incomplete_tool_arguments(self) -> None:
        with self.assertRaises(ValueError):
            AgentStep(
                action="get_transcript",
                query=None,
                segment_id=None,
                answer=None,
                evidence_insufficient=False,
            )

    def test_step_schema_rejects_unknown_tool(self) -> None:
        with self.assertRaises(ValueError):
            AgentStep(
                action="run_shell",
                query=None,
                segment_id=None,
                answer=None,
                evidence_insufficient=False,
            )

    def test_agent_stops_at_configured_step_limit(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "search_transcript", "query": "poetry"},
            ]
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=2,
        )

        with self.assertRaises(RuntimeError):
            agent.answer("What changed?")

    def test_invalid_segment_selection_uses_repair_not_valid_step(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing"},
                {"action": "get_transcript", "segment_id": 999},
                {"action": "get_transcript", "segment_id": 42},
                {
                    "action": "answer",
                    "answer": "Writing fixed poetry [00:01:40-00:01:50].",
                },
            ]
        )
        agent = SessionQuestionAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=3,
        )

        result = agent.answer("How did writing affect poetry?")

        self.assertEqual(result.step_count, 3)
        self.assertEqual(provider.calls, 4)

    def test_all_agent_step_fields_are_required_by_json_schema(self) -> None:
        required = set(AgentStep.model_json_schema()["required"])

        self.assertEqual(
            required,
            {
                "action",
                "query",
                "segment_id",
                "answer",
                "evidence_insufficient",
            },
        )

        self.assertEqual(
            set(RetrievalStep.model_json_schema()["required"]),
            {"action", "query", "segment_id", "evidence_insufficient"},
        )
        self.assertEqual(
            set(FinalAnswer.model_json_schema()["required"]),
            {"claims", "evidence_insufficient", "explanation"},
        )

    def test_forced_final_answer_uses_segment_ids_and_app_generated_citation(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing poetry"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "finish_retrieval"},
                {
                    "claims": [
                        {
                            "text": "Writing fixed poetry for absent readers.",
                            "segment_ids": [42],
                        }
                    ],
                },
            ]
        )
        agent = ForcedFinalAnswerAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
        )

        result = agent.answer("How did writing affect poetry?")

        self.assertEqual(result.step_count, 4)
        self.assertEqual(
            result.answer,
            "Writing fixed poetry for absent readers [00:01:40-00:01:50].",
        )
        self.assertEqual(provider.response_models[-1], FinalAnswer)
        final_prompt = provider.message_history[-1][-1].content
        self.assertIn("segment_id=42", final_prompt)
        self.assertIn("Tools are disabled", provider.message_history[-1][0].content)

    def test_forced_final_answer_retries_unretrieved_segment_id(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing poetry"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "finish_retrieval"},
                {
                    "claims": [
                        {"text": "Unsupported claim.", "segment_ids": [999]}
                    ],
                },
                {
                    "claims": [
                        {"text": "Writing fixed poetry.", "segment_ids": [42]}
                    ],
                },
            ]
        )
        trace_messages: list[str] = []
        agent = ForcedFinalAnswerAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
        )

        result = agent.answer(
            "How did writing affect poetry?", trace=trace_messages.append
        )

        self.assertEqual(result.step_count, 4)
        self.assertTrue(
            any(
                "FinalAnswer repair 1/3" in message
                and "were not retrieved" in message
                for message in trace_messages
            )
        )

    def test_forced_final_answer_starts_automatically_after_two_passages(self) -> None:
        transcript = [
            *self.transcript,
            StoredTranscriptSegment(
                id=43,
                session_id=2,
                index=1,
                start_seconds=110,
                end_seconds=120,
                text="Printing encouraged poets to write for the eye.",
            ),
        ]
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "writing poetry"},
                {"action": "get_transcript", "segment_id": 42},
                {"action": "search_transcript", "query": "printing visual"},
                {"action": "get_transcript", "segment_id": 43},
                {
                    "claims": [
                        {
                            "text": "Writing fixed poetry for absent readers.",
                            "segment_ids": [42],
                        },
                        {
                            "text": "Printing encouraged writing for the eye.",
                            "segment_ids": [43],
                        },
                    ],
                },
            ]
        )
        trace_messages: list[str] = []
        agent = ForcedFinalAnswerAgent(
            provider=provider,
            transcript=transcript,
            summary=None,
            checkpoints=[],
        )

        result = agent.answer(
            "How did writing affect poetry?", trace=trace_messages.append
        )

        self.assertEqual(result.step_count, 5)
        self.assertEqual(provider.calls, 5)
        self.assertIn("[00:01:50-00:02:00]", result.answer)
        self.assertIn(
            "Retrieval limit reached; forcing FinalAnswer with tools disabled",
            trace_messages,
        )

    def test_forced_final_answer_can_report_insufficient_evidence_after_two_searches(self) -> None:
        provider = FakeProvider(
            [
                {"action": "search_transcript", "query": "astronomy galaxies"},
                {"action": "search_transcript", "query": "planets stars"},
                {
                    "evidence_insufficient": True,
                    "explanation": "The recording does not discuss astronomy.",
                },
            ]
        )
        agent = ForcedFinalAnswerAgent(
            provider=provider,
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
        )

        result = agent.answer("What does the recording say about astronomy?")

        self.assertEqual(result.step_count, 3)
        self.assertEqual(result.answer, "The recording does not discuss astronomy.")

    def test_forced_final_answer_requires_room_for_search_get_and_final(self) -> None:
        agent = ForcedFinalAnswerAgent(
            provider=FakeProvider([]),
            transcript=self.transcript,
            summary=None,
            checkpoints=[],
            max_steps=2,
        )

        with self.assertRaisesRegex(ValueError, "at least 3"):
            agent.answer("What changed?")


if __name__ == "__main__":
    unittest.main()
