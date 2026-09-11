from __future__ import annotations

import builtins
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_voice_agent.cli import main
from local_voice_agent.llm import OllamaProvider, StructuredResponse
from local_voice_agent.models import StoredTranscriptSegment, TranscriptSegment, TranscriptionResult
from local_voice_agent.storage import Database
from local_voice_agent.summaries.modes import summarizer_class_for
from local_voice_agent.summaries.summarizer import SessionSummarizer

HAS_FRAMEWORK = importlib.util.find_spec("pydantic_ai") is not None
if HAS_FRAMEWORK:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from local_voice_agent.summaries.pydantic_summarizer import PydanticSessionSummarizer

CHECKPOINT = {"summary": "Writing fixed poetry.", "notes": ["Poems were written down."]}
FINAL = {"overall_summary": "Writing changed poetry.",
         "important_notes": ["Writing fixed content."], "main_topics": ["Poetry"]}


class CustomProvider:
    model_name = "offline"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def structured_chat(self, *, messages, response_model):
        self.messages.append(messages)
        raw = json.dumps(next(self.responses))
        return StructuredResponse(data=response_model.model_validate_json(raw), raw_content=raw)


class SummaryModeTests(unittest.TestCase):
    def test_custom_selection_without_optional_dependency(self):
        original = builtins.__import__

        def missing(name, *args, **kwargs):
            if name == "local_voice_agent.summaries.pydantic_summarizer":
                raise ImportError("missing extra")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing):
            self.assertIs(summarizer_class_for("custom"), SessionSummarizer)
            with self.assertRaisesRegex(RuntimeError, "pip install"):
                summarizer_class_for("pydanticai")


@unittest.skipUnless(HAS_FRAMEWORK, "Install the pydanticai extra")
class PydanticNotesTests(unittest.TestCase):
    def make_model(self, responses):
        script = iter(responses)
        self.requests = []

        def respond(messages, info):
            self.requests.append((messages, info))
            payload = next(script)
            if isinstance(payload, Exception):
                raise payload
            return ModelResponse(parts=[TextPart(json.dumps(payload))])

        return FunctionModel(respond)

    def make_summarizer(self, responses):
        return PydanticSessionSummarizer(
            provider=OllamaProvider(model_name="offline"), model=self.make_model(responses),
            target_chunk_seconds=60, minimum_final_chunk_seconds=0,
        )

    def segments(self):
        return [StoredTranscriptSegment(id=i + 1, session_id=1, index=i,
                                        start_seconds=i * 100, end_seconds=i * 100 + 10,
                                        text=f"Transcript section {i} about poetry.")
                for i in range(2)]

    def test_shared_pipeline_prompts_schemas_and_raw_responses(self):
        provider = CustomProvider([CHECKPOINT, CHECKPOINT, FINAL])
        custom = SessionSummarizer(provider=provider, target_chunk_seconds=60,
                                   minimum_final_chunk_seconds=0).summarize(self.segments())
        summarizer = self.make_summarizer([CHECKPOINT, CHECKPOINT, FINAL])
        progress = []
        framework = summarizer.summarize(
            self.segments(), progress=lambda i, n: progress.append((i, n)),
            final_progress=lambda: progress.append("final"),
        )
        self.assertEqual(progress, [(1, 2), (2, 2), "final"])
        self.assertEqual(custom.final_summary, framework.final_summary)
        self.assertEqual(len(framework.checkpoints), 2)
        for old, new in zip(custom.checkpoints, framework.checkpoints, strict=True):
            self.assertEqual(old.checkpoint, new.checkpoint)
            self.assertEqual((old.start_seconds, old.end_seconds),
                             (new.start_seconds, new.end_seconds))
            self.assertEqual(old.raw_response, new.raw_response)
            self.assertEqual(len(new.attempt_seconds), 1)
            self.assertGreaterEqual(new.generation_seconds, 0)
        self.assertEqual(framework.final_raw_response, json.dumps(FINAL))
        for baseline, (messages, info) in zip(provider.messages, self.requests, strict=True):
            self.assertEqual([m.content for m in baseline],
                             [part.content for m in messages for part in m.parts])
            self.assertFalse(info.function_tools)
            self.assertFalse(info.output_tools)
            self.assertEqual(info.model_request_parameters.output_mode, "native")

    def test_schema_and_content_repairs_are_bounded_and_timed(self):
        for invalid in ({"summary": ""},
                        {"summary": "No transcript was provided", "notes": ["Missing"]}):
            with self.subTest(invalid=invalid):
                summarizer = self.make_summarizer([
                    invalid, CHECKPOINT, CHECKPOINT, {"main_topics": []}, FINAL
                ])
                result = summarizer.summarize(self.segments())
                self.assertEqual(len(result.checkpoints[0].attempt_seconds), 2)
                self.assertEqual(len(result.final_attempt_seconds), 2)
                self.assertEqual(result.retry_count, 2)
                self.assertEqual(len(self.requests), 5)

    def test_exhausted_repairs_and_transport_failure_stop_generation(self):
        for responses, expected in (([{}, {}], 2), ([RuntimeError("connection failed")], 1)):
            with self.subTest(expected=expected):
                summarizer = self.make_summarizer(responses)
                with self.assertRaisesRegex(RuntimeError, "PydanticAI note generation failed"):
                    summarizer.summarize(self.segments())
                self.assertEqual(len(self.requests), expected)

    def test_cli_storage_migration_and_failure_preserve_previous_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.db"
            db = Database(path)
            db.initialize()
            sid = db.create_session(Path("fixture.wav"), "en", "fake")
            db.store_transcription(sid, TranscriptionResult(segments=(
                TranscriptSegment(index=0, start_seconds=0, end_seconds=10, text="Poetry."),
            )))
            # Exercise the real pre-feature schema and its automatic additive upgrade.
            with db.connect() as connection:
                connection.execute("ALTER TABLE summary_runs DROP COLUMN summary_mode")
                connection.execute("PRAGMA user_version=7")
                connection.commit()

            def invoke(mode):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    return main(["--database", str(path), "summarize", str(sid),
                                 "--summary-mode", mode, "--label", "offline notes"])

            with patch("local_voice_agent.cli.OllamaProvider",
                       return_value=CustomProvider([CHECKPOINT, FINAL])):
                self.assertEqual(invoke("custom"), 0)
            old = db.get_final_summary(sid)
            self.assertEqual(old.summary_mode, "custom")
            with db.connect() as connection:
                connection.execute("ALTER TABLE summary_runs DROP COLUMN summary_mode")
                connection.execute("PRAGMA user_version=7")
                connection.commit()
            with patch("local_voice_agent.summaries.pydantic_summarizer.OllamaModel",
                       return_value=self.make_model([CHECKPOINT, FINAL])):
                self.assertEqual(invoke("pydanticai"), 0)
            latest = db.get_final_summary(sid)
            self.assertEqual(latest.summary_mode, "pydanticai")
            self.assertEqual(db.get_final_summary(sid, old.id), old)
            self.assertEqual([r.summary_mode for r in db.list_summary_runs(sid)],
                             ["pydanticai", "custom"])
            with patch("local_voice_agent.summaries.pydantic_summarizer.OllamaModel",
                       return_value=self.make_model([CHECKPOINT, {}, {}])):
                self.assertEqual(invoke("pydanticai"), 1)
            self.assertEqual(len(db.list_summary_runs(sid)), 2)
            self.assertEqual(db.get_final_summary(sid), latest)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--database", str(path), "summary", str(sid)]), 0)
            self.assertIn("Summary mode: pydanticai", output.getvalue())
