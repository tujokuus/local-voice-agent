from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_voice_agent.agent import AgentStep, FinalAnswer
from local_voice_agent.cli import main
from local_voice_agent.evaluation.database import EvaluationDatabase
from local_voice_agent.evaluation.history import read_history
from local_voice_agent.evaluation.matrix import run_matrix
from local_voice_agent.evaluation.reporting import compare_runs, show_run
from local_voice_agent.evaluation.runner import run_evaluation
from local_voice_agent.llm import StructuredResponse
from local_voice_agent.models import TranscriptSegment, TranscriptionResult
from local_voice_agent.storage import Database
from test_evaluation import dataset_payload


class OfflineProvider:
    def __init__(self, model_name: str = "fake", *, fail: bool = False) -> None:
        self.model_name = model_name
        self.calls = 0
        self.fail = fail

    def structured_chat(self, *, messages, response_model):
        self.calls += 1
        if self.fail:
            raise RuntimeError("offline simulated failure")
        phase = (self.calls - 1) % 3
        if response_model is FinalAnswer:
            payload = {"claims": [], "evidence_insufficient": True,
                       "explanation": "The recording does not provide that information."}
        else:
            payload = {"action": "search_transcript", "query": ("galaxies", "planets")[phase % 2],
                       "segment_id": None, "evidence_insufficient": False}
            if response_model is AgentStep:
                payload["answer"] = None
            if phase == 2:
                payload.update(action="answer", query=None, evidence_insufficient=True,
                               answer="The recording does not provide that information.")
        data = response_model.model_validate(payload)
        return StructuredResponse(data=data, raw_content=data.model_dump_json())


class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = Database(self.directory / "source.db")
        self.source.initialize()
        self.session_id = self.source.create_session(Path("fixture.wav"), "en", "fake")
        self.source.store_transcription(self.session_id, TranscriptionResult(segments=(
            TranscriptSegment(index=0, start_seconds=100, end_seconds=110,
                              text="Writing fixed poetry."),
        )))
        segment_id = self.source.get_transcript(self.session_id)[0].id
        self.dataset = self.directory / "dataset.json"
        self.dataset.write_text(json.dumps(dataset_payload(segment_id, self.session_id)),
                                encoding="utf-8")
        self.history = EvaluationDatabase(self.directory / "history.db")

    def single(self, **options):
        return run_evaluation(self.dataset, self.source, self.history, self.session_id,
                              OfflineProvider(), **options)

    def matrix(self, factory=OfflineProvider, **options):
        return run_matrix(self.dataset, self.source, self.history, self.session_id,
                          factory, **options)

    def cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main([*args, "--evaluation-database", str(self.history.path)])
        return code, output.getvalue()

    def test_matrix_runs_all_four_combinations_and_retains_history(self) -> None:
        providers = []

        def factory(model):
            provider = OfflineProvider(model, fail=not providers)
            providers.append(provider)
            return provider

        report = self.matrix(factory, limit=1, label="offline group")
        runs = read_history(self.history.path, group_id=report.group_id)
        self.assertEqual(len(report.runs), 4)
        self.assertEqual({(run["model_name"], run["agent_mode"]) for run in runs}, {
            ("qwen3.5:4b", "final"), ("qwen3.5:4b", "manual"),
            ("qwen3.5:9b", "final"), ("qwen3.5:9b", "manual"),
        })
        self.assertEqual(report.runs[0].status, "completed_with_errors")
        self.assertTrue(all(run.status == "completed" for run in report.runs[1:]))
        self.assertTrue(all(len(run["cases"]) == 1 for run in runs))
        original_bytes = self.history.path.read_bytes()
        show_run(runs[0])
        compare_runs(sorted(runs, key=lambda run: run["id"]))
        self.assertEqual(self.history.path.read_bytes(), original_bytes)
        another = self.matrix(limit=1)
        self.assertNotEqual(report.group_id, another.group_id)
        self.assertEqual(len(read_history(self.history.path)), 8)
        self.assertEqual(len(read_history(self.history.path, group_id=report.group_id)), 4)

    def test_matrix_freezes_dataset_transcript_and_summary_before_provider_calls(self) -> None:
        def factory(model):
            self.dataset.write_text("not valid JSON any more", encoding="utf-8")
            with self.source.connect() as connection:
                connection.execute("UPDATE transcript_segments SET text='Changed after preflight'")
                connection.commit()
            return OfflineProvider(model)

        with (
            patch.object(
                self.source, "get_final_summary", wraps=self.source.get_final_summary
            ) as summary,
            patch.object(
                self.source, "get_transcript", wraps=self.source.get_transcript
            ) as transcript,
        ):
            report = self.matrix(factory, limit=1)
        self.assertEqual(summary.call_count, 1)
        self.assertEqual(transcript.call_count, 1)
        runs = read_history(self.history.path, group_id=report.group_id)
        for key in ("dataset_sha256", "transcript_sha256", "summary_sha256"):
            self.assertEqual(len({run[key] for run in runs}), 1)
        saved = self.history.get_run(report.runs[-1].run_id)
        self.assertIn("Writing fixed poetry", saved["transcript_snapshot_json"])
        self.assertNotIn("Changed after", saved["transcript_snapshot_json"])

    def test_matrix_invalid_inputs_fail_before_provider_or_history_creation(self) -> None:
        for options in ({"models": []}, {"models": ["fake", "fake"]},
                        {"agent_modes": ["invalid"]}, {"max_steps": 2},
                        {"limit": 0}, {"case_ids": ["missing"]}):
            with self.subTest(options=options):
                with patch(__name__ + ".OfflineProvider") as provider:
                    with self.assertRaises(ValueError):
                        self.matrix(provider, **options)
                    provider.assert_not_called()
                self.assertFalse(self.history.path.exists())

    def test_matrix_interruption_preserves_group_and_does_not_start_next_model(self) -> None:
        with (
            patch.object(OfflineProvider, "structured_chat", side_effect=KeyboardInterrupt),
            patch(__name__ + ".OfflineProvider", wraps=OfflineProvider) as factory,
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.matrix(factory)
            self.assertEqual(factory.call_count, 1)
        runs = read_history(self.history.path)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "interrupted")
        self.assertTrue(runs[0]["configuration"]["comparison_group_id"])
        self.assertEqual(len(runs[0]["configuration"]["comparison_expected_combinations"]), 4)

    def test_history_filters_and_readonly_missing_path(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_history(self.history.path)
        self.assertFalse(self.history.path.exists())
        first = self.single(limit=1, label="first")
        second = self.single(limit=1, label="second")
        self.assertEqual(read_history(self.history.path, limit=1)[0]["id"], second.run_id)
        self.assertEqual([run["id"] for run in read_history(
            self.history.path, run_ids=[first.run_id, second.run_id]
        )], [first.run_id, second.run_id])
        self.assertEqual(read_history(self.history.path, session_id=999), [])
        with self.assertRaises(LookupError):
            read_history(self.history.path, run_ids=[999])
        with self.assertRaises(ValueError):
            read_history(self.history.path, run_ids=[first.run_id, first.run_id])
        with self.assertRaises(LookupError):
            read_history(self.history.path, group_id="missing")

    def test_show_includes_answer_reference_required_points_and_errors(self) -> None:
        report = self.single()
        run = read_history(self.history.path, run_ids=[report.run_id])[0]
        text = show_run(run, "writing")
        for fragment in ("How did writing change poetry?", "Generated answer:",
                         "does not provide", "Reference answer:", "Writing fixed poetry.",
                         "Required points:", "Gold evidence:", "Answerability correct: False"):
            self.assertIn(fragment, text)
        self.assertNotIn("Case: astronomy", text)
        with self.assertRaises(LookupError):
            show_run(run, "unknown")
        failed = run_evaluation(self.dataset, self.source, self.history, self.session_id,
                                OfflineProvider(fail=True), limit=1)
        text = show_run(read_history(self.history.path, run_ids=[failed.run_id])[0])
        self.assertIn("offline simulated failure", text)
        self.assertIn("no accepted answer", text)

    def test_partial_running_results_are_not_treated_as_completed(self) -> None:
        snapshots = []

        def progress(message):
            if message.startswith("  completed"):
                snapshots.append(read_history(self.history.path)[0])

        self.single(progress=progress)
        first = snapshots[0]
        self.assertEqual(first["status"], "running")
        self.assertEqual(first["metrics"]["attempted_cases"], 1)
        text = show_run(first, "astronomy")
        self.assertIn("No stored result yet: astronomy", text)
        self.assertIn("Partial stored results", text)

    def test_compare_uses_first_run_baseline_and_detects_input_mismatch(self) -> None:
        first, second = self.single(limit=1), self.single(limit=1)
        runs = read_history(self.history.path, run_ids=[first.run_id, second.run_id])
        runs[0]["metrics"]["citation_recall"] = 0.5
        runs[1]["metrics"]["citation_recall"] = 0.75
        text = compare_runs(runs, case_id="writing")
        self.assertIn("+25.0pp", text)
        self.assertIn("Generated answer:", text)
        for field in ("dataset_sha256", "transcript_sha256", "summary_sha256"):
            different = copy.deepcopy(runs)
            different[1][field] = "changed"
            with self.assertRaisesRegex(ValueError, field):
                compare_runs(different)
            self.assertIn("NON-EQUIVALENT", compare_runs(different, allow_mismatch=True))
        for field in ("selected_case_ids", "metric_version", "max_steps",
                      "timeout_seconds_per_request"):
            different = copy.deepcopy(runs)
            different[1]["configuration"][field] = "changed"
            with self.assertRaisesRegex(ValueError, field):
                compare_runs(different)
        with self.assertRaises(ValueError):
            compare_runs(runs[:1])

    def test_cli_matrix_reports_failure_then_history_show_and_comparison_work(self) -> None:
        count = 0

        def factory(*, model_name, **kwargs):
            nonlocal count
            count += 1
            return OfflineProvider(model_name, fail=count == 1)

        with patch("local_voice_agent.evaluation.cli.OllamaProvider", side_effect=factory):
            code, text = self.cli(
                "--database", str(self.source.path), "evaluate-matrix", str(self.dataset),
                "--session-id", str(self.session_id), "--limit", "1", "--label", "CLI group",
            )
        self.assertEqual(code, 1, text)
        self.assertIn("4 combinations x 1 cases = 4", text)
        self.assertIn("Citation recall", text)
        runs = read_history(self.history.path)
        group_id = runs[0]["configuration"]["comparison_group_id"]
        with patch("local_voice_agent.evaluation.cli.OllamaProvider") as provider:
            code, text = self.cli("evaluation-runs", "--group-id", group_id)
            self.assertEqual(code, 0, text)
            self.assertIn("CLI group", text)
            code, text = self.cli("evaluation-show", "1", "--case-id", "writing")
            self.assertEqual(code, 0, text)
            self.assertIn("Generated answer", text)
            code, text = self.cli("evaluation-compare", "--group-id", group_id)
            self.assertEqual(code, 0, text)
            self.assertIn("Deltas are relative to run 1", text)
            code, text = self.cli("evaluation-compare", "1", "2", "--case-id", "writing")
            self.assertEqual(code, 0, text)
            self.assertIn("Reference answer", text)
            provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
