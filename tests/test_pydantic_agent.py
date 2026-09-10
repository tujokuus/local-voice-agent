from __future__ import annotations

import importlib.util
import builtins
import json
import unittest
from unittest.mock import patch

from local_voice_agent.llm import OllamaProvider
from local_voice_agent.agent_modes import agent_class_for
from local_voice_agent.models import StoredTranscriptSegment

HAS_FRAMEWORK = importlib.util.find_spec("pydantic_ai") is not None
if HAS_FRAMEWORK:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from local_voice_agent.pydantic_agent import PydanticQuestionAgent


class OptionalFrameworkTests(unittest.TestCase):
    def test_missing_extra_does_not_prevent_baseline_agent_selection(self):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "local_voice_agent.pydantic_agent":
                raise ImportError("simulated missing framework")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            self.assertEqual(agent_class_for("manual").__name__, "SessionQuestionAgent")
            self.assertEqual(agent_class_for("final").__name__, "ForcedFinalAnswerAgent")
            with self.assertRaisesRegex(RuntimeError, "pip install"):
                agent_class_for("pydanticai")


@unittest.skipUnless(HAS_FRAMEWORK, 'Install the ".[pydanticai]" extra for framework tests')
class PydanticAgentTests(unittest.TestCase):
    def make_agent(self, responses, *, max_steps=5):
        self.requests = []
        script = iter(responses)

        def respond(messages, info):
            self.requests.append((messages, info))
            response = next(script)
            if isinstance(response, Exception):
                raise response
            if isinstance(response, dict):
                return ModelResponse(parts=[TextPart(json.dumps(response))])
            return ModelResponse(parts=response)

        return PydanticQuestionAgent(
            provider=OllamaProvider(model_name="offline"), model=FunctionModel(respond),
            transcript=[
                StoredTranscriptSegment(id=42, session_id=2, index=0, start_seconds=100,
                                        end_seconds=110, text="Writing fixed poetry."),
                StoredTranscriptSegment(id=99, session_id=2, index=1, start_seconds=300,
                                        end_seconds=310, text="Poetry for absent readers."),
            ], summary=None, checkpoints=[], max_steps=max_steps,
        )

    @staticmethod
    def call(name, **args):
        return [ToolCallPart(name, args)]

    @staticmethod
    def answer(segment_id=42):
        return {"claims": [{"text": "Writing fixed poetry", "segment_ids": [segment_id]}],
                "evidence_insufficient": False, "explanation": None}

    def test_native_tools_then_tool_free_final_with_application_citations(self):
        agent = self.make_agent([
            self.call("search_transcript", query="poetry"),
            self.call("get_transcript", segment_id=42),
            self.call("finish_retrieval", evidence_insufficient=False),
            self.answer(),
        ])
        result = agent.answer("What was written?")
        self.assertIn("[00:01:40-00:01:50]", result.answer)
        self.assertEqual(result.step_count, 4)
        self.assertEqual(result.tool_calls, ("search_transcript", "get_transcript"))
        self.assertEqual(result.retrieved_segment_ids, (42, 99))
        self.assertEqual(result.cited_segment_ids, (42,))
        final_messages, final_info = self.requests[-1]
        self.assertFalse(final_info.function_tools)
        self.assertFalse(final_info.output_tools)
        self.assertEqual(final_info.model_request_parameters.output_mode, "native")
        self.assertNotIn("absent readers", str(final_messages))

    def test_search_only_citation_is_rejected_and_repaired(self):
        agent = self.make_agent([
            self.call("search_transcript", query="poetry"),
            self.call("get_transcript", segment_id=42),
            self.answer(99), self.answer(),
        ], max_steps=3)
        result = agent.answer("What was written?")
        self.assertEqual(result.cited_segment_ids, (42,))
        self.assertEqual(result.step_count, 3)
        self.assertIn("not retrieved", str(self.requests[-1][0]))

    def test_two_empty_searches_allow_insufficiency_without_more_retrieval(self):
        agent = self.make_agent([
            self.call("search_transcript", query="galaxies"),
            self.call("search_transcript", query="planets"),
            {"claims": [], "evidence_insufficient": True, "explanation": "No evidence."},
        ])
        result = agent.answer("Which planet?")
        self.assertTrue(result.evidence_insufficient)
        self.assertEqual(result.step_count, 3)
        self.assertEqual(len(self.requests), 3)

    def test_early_insufficiency_and_unsearched_segment_are_repaired(self):
        agent = self.make_agent([
            self.call("finish_retrieval", evidence_insufficient=True),
            self.call("get_transcript", segment_id=42),
            self.call("search_transcript", query="poetry"),
            self.call("get_transcript", segment_id=42),
            self.answer(),
        ], max_steps=3)
        self.assertEqual(agent.answer("What was written?").cited_segment_ids, (42,))

    def test_final_failure_keeps_partial_telemetry_and_limits_requests(self):
        agent = self.make_agent([
            self.call("search_transcript", query="poetry"),
            self.call("get_transcript", segment_id=42),
            *[self.answer(999)] * 4,
        ], max_steps=3)
        with self.assertRaises(RuntimeError):
            agent.answer("What was written?")
        self.assertEqual(len(self.requests), 6)
        self.assertEqual(agent.last_telemetry.tool_calls, ("search_transcript", "get_transcript"))
        self.assertEqual(agent.last_telemetry.cited_segment_ids, ())

    def test_same_response_cannot_exceed_tool_budget(self):
        agent = self.make_agent([
            self.call("search_transcript", query="poetry"),
            [ToolCallPart("get_transcript", {"segment_id": 42}, "one"),
             ToolCallPart("get_transcript", {"segment_id": 99}, "two")],
            self.answer(),
        ], max_steps=3)
        result = agent.answer("What was written?")
        self.assertEqual(result.tool_calls, ("search_transcript", "get_transcript"))
        self.assertNotIn("absent readers", str(self.requests[-1][0]))

    def test_repeated_calls_fail_with_bounded_partial_telemetry(self):
        agent = self.make_agent([self.call("search_transcript", query="poetry")] * 4)
        with self.assertRaises(RuntimeError):
            agent.answer("What was written?")
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(agent.last_telemetry.tool_calls, ("search_transcript",))

    def test_framework_repairs_invalid_tool_arguments(self):
        agent = self.make_agent([
            self.call("get_transcript", segment_id=-1),
            self.call("search_transcript", query="poetry"),
            self.call("get_transcript", segment_id=42),
            self.answer(),
        ], max_steps=3)
        self.assertEqual(agent.answer("What was written?").step_count, 3)
        self.assertIn("greater_than", str(self.requests[1][0]))

    def test_search_results_without_inspection_cannot_reach_final(self):
        agent = self.make_agent([
            self.call("search_transcript", query="poetry"),
            self.call("search_transcript", query="writing"),
        ], max_steps=3)
        with self.assertRaisesRegex(RuntimeError, "without inspected evidence"):
            agent.answer("What was written?")
        self.assertEqual(len(self.requests), 2)

    def test_ollama_transport_uses_native_tools_and_native_json_output(self):
        import httpx2
        from openai import AsyncOpenAI

        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(str(request.url), "http://127.0.0.1:11434/v1/chat/completions")
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["reasoning_effort"], "none")
            if len(requests) <= 2:
                name = "search_transcript" if len(requests) == 1 else "get_transcript"
                args = {"query": "poetry"} if len(requests) == 1 else {"segment_id": 42}
                self.assertFalse(payload["parallel_tool_calls"])
                self.assertEqual({tool["function"]["name"] for tool in payload["tools"]}, {
                    "search_transcript", "get_transcript", "get_session_summary", "finish_retrieval"
                })
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call{len(requests)}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }]}
            else:
                self.assertNotIn("tools", payload)
                self.assertEqual(payload["response_format"]["type"], "json_schema")
                message = {"role": "assistant", "content": json.dumps(self.answer())}
            return httpx2.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0, "model": "offline",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        def client_factory(**kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            self.assertEqual(kwargs["timeout"], 600)
            return AsyncOpenAI(**kwargs, http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(respond)
            ))

        agent = self.make_agent([], max_steps=3)
        agent._model = None
        agent.provider = OllamaProvider(model_name="qwen3.5:4b")
        with patch("local_voice_agent.pydantic_agent.AsyncOpenAI", side_effect=client_factory):
            result = agent.answer("What was written?")
        self.assertEqual(result.cited_segment_ids, (42,))
        self.assertEqual(len(requests), 3)
