from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_voice_agent.agent import AgentStep, RetrievalStep
from local_voice_agent.cli import _build_parser, main
from local_voice_agent.config import Settings
from local_voice_agent.llm import StructuredResponse
from local_voice_agent.models import TranscriptSegment, TranscriptionResult
from local_voice_agent.storage import Database


class ScriptedProvider:
    model_name = "fake-cli-model"

    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def structured_chat(self, *, messages, response_model):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        if response_model is AgentStep:
            defaults = {
                "query": None, "segment_id": None, "answer": None,
                "evidence_insufficient": False,
            }
        elif response_model is RetrievalStep:
            defaults = {"query": None, "segment_id": None, "evidence_insufficient": False}
        else:
            defaults = {"claims": [], "explanation": None, "evidence_insufficient": False}
        data = response_model.model_validate({**defaults, **response})
        return StructuredResponse(data=data, raw_content=data.model_dump_json())


class EvaluationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.database_path = self.directory / "application.db"
        self.history_path = self.directory / "evaluations.db"
        self.dataset_path = self.directory / "cases.json"
        database = Database(self.database_path)
        database.initialize()
        self.session_id = database.create_session(Path("poetry.wav"), "en", "fake")
        database.store_transcription(
            self.session_id,
            TranscriptionResult(segments=(TranscriptSegment(
                index=0, start_seconds=100, end_seconds=110,
                text="Writing fixed poetry for absent readers.",
            ),)),
        )
        self.segment_id = database.get_transcript(self.session_id)[0].id
        self.dataset = {
            "schema_version": 1, "id": "cli-fixture", "description": "Offline CLI fixture",
            "source_session_id": self.session_id,
            "cases": [
                {
                    "id": "writing", "question": "How did writing change poetry?",
                    "answerable": True, "category": "focused", "difficulty": "easy",
                    "reference_answer": "Writing fixed poetry for absent readers.",
                    "required_points": ["Writing fixed poetry."],
                    "gold_evidence_ranges": [{
                        "start_seconds": 100, "end_seconds": 110,
                        "segment_ids": [self.segment_id], "explanation": "Writing's effect.",
                    }],
                },
                {
                    "id": "astronomy", "question": "What does it say about astronomy?",
                    "answerable": False, "category": "unanswerable", "difficulty": "easy",
                    "reference_answer": "The recording does not discuss astronomy.",
                    "required_points": ["Requested information is absent."],
                    "gold_evidence_ranges": [],
                },
            ],
        }
        self._write_dataset()

    def _write_dataset(self) -> None:
        self.dataset_path.write_text(json.dumps(self.dataset), encoding="utf-8")

    def _run(self, responses, *options):
        provider = ScriptedProvider(responses)
        output = io.StringIO()
        with (
            patch(
                "local_voice_agent.evaluation.cli.OllamaProvider", return_value=provider
            ) as factory,
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            code = main([
                "--database", str(self.database_path), "evaluate", str(self.dataset_path),
                "--session-id", str(self.session_id),
                "--evaluation-database", str(self.history_path),
                "--model", provider.model_name, *options,
            ])
        return code, output.getvalue(), provider, factory

    def test_both_agent_modes_execute_answerable_and_unanswerable_cases(self) -> None:
        for mode in ("final", "manual"):
            with self.subTest(mode=mode):
                prefix = [
                    {"action": "search_transcript", "query": "writing"},
                    {"action": "get_transcript", "segment_id": self.segment_id},
                ]
                searches = [
                    {"action": "search_transcript", "query": "astronomy"},
                    {"action": "search_transcript", "query": "galaxies"},
                ]
                if mode == "final":
                    responses = [
                        *prefix, {"action": "finish_retrieval"},
                        {"claims": [{"text": "Writing fixed poetry.",
                                     "segment_ids": [self.segment_id]}]},
                        *searches,
                        {"evidence_insufficient": True,
                         "explanation": "The recording does not discuss astronomy."},
                    ]
                else:
                    responses = [
                        *prefix,
                        {"action": "answer",
                         "answer": "Writing fixed poetry [00:01:40-00:01:50]."},
                        *searches,
                        {"action": "answer", "evidence_insufficient": True,
                         "answer": "The recording does not discuss astronomy."},
                    ]
                code, output, provider, factory = self._run(
                    responses, "--agent-mode", mode, "--timeout", "17",
                    "--label", f"fake {mode}",
                )
                self.assertEqual(code, 0, output)
                self.assertIn("2 successful, 0 failed", output)
                self.assertIn("Answerability accuracy: 100.0%", output)
                self.assertIn("Citation recall: 100.0%", output)
                self.assertEqual(provider.calls, len(responses))
                self.assertEqual(factory.call_args.kwargs["timeout_seconds"], 17)

    def test_case_failure_still_runs_next_case_and_returns_nonzero(self) -> None:
        code, output, provider, _ = self._run([
            RuntimeError("fake model failure"),
            {"action": "search_transcript", "query": "astronomy"},
            {"action": "search_transcript", "query": "galaxies"},
            {"evidence_insufficient": True,
             "explanation": "The recording does not discuss astronomy."},
        ])
        self.assertEqual(code, 1, output)
        self.assertIn("1 successful, 1 failed", output)
        self.assertIn("fake model failure", output)
        self.assertEqual(provider.calls, 4)

    def test_unknown_case_and_nonfinite_timeout_do_not_call_provider(self) -> None:
        for options in (("--case-id", "typo"), ("--timeout", "nan"), ("--limit", "0")):
            with self.subTest(options=options):
                code, output, provider, _ = self._run([], *options)
                self.assertEqual(code, 1, output)
                self.assertEqual(provider.calls, 0)
                self.assertFalse(self.history_path.exists())

    def test_invalid_gold_is_checked_even_outside_selected_subset(self) -> None:
        self.dataset["cases"][0]["gold_evidence_ranges"][0]["segment_ids"] = [999]
        self._write_dataset()
        code, output, provider, _ = self._run([], "--case-id", "astronomy")
        self.assertEqual(code, 1, output)
        self.assertEqual(provider.calls, 0)
        self.assertFalse(self.history_path.exists())

    def test_application_and_evaluation_database_must_be_separate(self) -> None:
        original = self.database_path.read_bytes()
        code, output, provider, _ = self._run(
            [], "--evaluation-database", str(self.database_path),
        )
        self.assertEqual(code, 1, output)
        self.assertIn("must differ", output)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(self.database_path.read_bytes(), original)

    def test_parser_preserves_application_database_option_and_repeated_case_ids(self) -> None:
        settings = Settings(database_path=Path("configured.db"), ollama_model="configured-model")
        args = _build_parser(settings).parse_args([
            "--database", "source.db", "evaluate", "dataset.json", "--session-id", "2",
            "--case-id", "first", "--case-id", "second", "--limit", "1",
        ])
        self.assertEqual(args.database, Path("source.db"))
        self.assertEqual(args.case_ids, ["first", "second"])
        self.assertEqual(args.model, "configured-model")
        self.assertEqual(args.agent_mode, "final")
        self.assertEqual(args.evaluation_database, Path("data/local_voice_agent_evaluations.db"))


if __name__ == "__main__":
    unittest.main()
