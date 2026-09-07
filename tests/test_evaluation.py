from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from local_voice_agent.evaluation import (
    EvaluationDatabase,
    EvaluationDataset,
    load_dataset,
    run_evaluation,
    validate_dataset_session,
)
from local_voice_agent.evaluation.metrics import (
    aggregate_metrics,
    compute_case_metrics,
    intervals_overlap,
)
from local_voice_agent.models import StoredTranscriptSegment, TranscriptSegment, TranscriptionResult
from local_voice_agent.storage import Database
from test_evaluation_cli import ScriptedProvider


def dataset_payload(segment_id: int = 42, session_id: int = 2) -> dict:
    return {
        "schema_version": 1, "id": "fixture", "description": "Offline evaluation fixture",
        "source_session_id": session_id,
        "cases": [
            {
                "id": "writing", "question": "How did writing change poetry?",
                "answerable": True, "category": "focused", "difficulty": "easy",
                "reference_answer": "Writing fixed poetry.",
                "required_points": ["Writing fixed poetry."],
                "gold_evidence_ranges": [{
                    "start_seconds": 100, "end_seconds": 110,
                    "segment_ids": [segment_id], "explanation": "Writing fixed content.",
                }],
            },
            {
                "id": "astronomy", "question": "What does it say about astronomy?",
                "answerable": False, "category": "unanswerable", "difficulty": "medium",
                "reference_answer": "The recording does not discuss astronomy.",
                "required_points": ["Acknowledge missing information."],
                "gold_evidence_ranges": [],
            },
        ],
    }


def insufficient_responses() -> list[dict]:
    return [
        {"action": "search_transcript", "query": "astronomy"},
        {"action": "search_transcript", "query": "galaxies"},
        {"evidence_insufficient": True,
         "explanation": "The recording does not discuss astronomy."},
    ]


class DatasetValidationTests(unittest.TestCase):
    def test_rejects_duplicate_ids_and_invalid_evidence_contracts(self) -> None:
        mutations = [
            lambda data: data["cases"][1].update(id="writing"),
            lambda data: data["cases"][0].update(gold_evidence_ranges=[]),
            lambda data: data["cases"][1].update(
                gold_evidence_ranges=copy.deepcopy(data["cases"][0]["gold_evidence_ranges"])
            ),
            lambda data: data["cases"][0].update(question="   "),
            lambda data: data["cases"][0].update(required_points=[]),
            lambda data: data["cases"][0].update(answerable="true"),
            lambda data: data["cases"][0]["gold_evidence_ranges"][0].update(segment_ids=[42, 42]),
            lambda data: data["cases"][0]["gold_evidence_ranges"][0].update(segment_ids=[True]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                data = dataset_payload()
                mutate(data)
                with self.assertRaises(ValueError):
                    EvaluationDataset.model_validate(data)

    def test_rejects_nonfinite_negative_reversed_and_empty_ranges(self) -> None:
        for start, end in ((-1, 10), (10, 10), (11, 10), (float("nan"), 10),
                           (0, float("inf"))):
            with self.subTest(start=start, end=end):
                data = dataset_payload()
                data["cases"][0]["gold_evidence_ranges"][0].update(
                    start_seconds=start, end_seconds=end
                )
                with self.assertRaises(ValueError):
                    EvaluationDataset.model_validate(data)

    def test_session_preflight_handles_display_precision_and_explicit_id_remapping(self) -> None:
        dataset = EvaluationDataset.model_validate(dataset_payload())
        transcript = [StoredTranscriptSegment(
            id=42, session_id=2, index=0, start_seconds=100.2, end_seconds=110.8,
            text="Writing fixed poetry.",
        )]
        self.assertEqual(validate_dataset_session(dataset, transcript, session_id=2)["writing"],
                         [[42]])
        imported = [transcript[0].model_copy(update={"id": 700, "session_id": 9})]
        with self.assertRaisesRegex(ValueError, "segment IDs"):
            validate_dataset_session(dataset, imported, session_id=9)
        self.assertEqual(validate_dataset_session(
            dataset, imported, session_id=9, remap_segment_ids=True
        )["writing"], [[700]])
        with self.assertRaisesRegex(ValueError, "different session"):
            validate_dataset_session(dataset, transcript, session_id=9)

    def test_preflight_rejects_missing_claim_continuation_and_changed_segment_count(self) -> None:
        data = dataset_payload()
        data["cases"][0]["gold_evidence_ranges"][0]["end_seconds"] = 120
        dataset = EvaluationDataset.model_validate(data)
        transcript = [
            StoredTranscriptSegment(id=42, session_id=2, index=0,
                                    start_seconds=100, end_seconds=110, text="Writing fixed"),
            StoredTranscriptSegment(id=43, session_id=2, index=1,
                                    start_seconds=110, end_seconds=120, text="poetry."),
        ]
        with self.assertRaisesRegex(ValueError, "segment IDs"):
            validate_dataset_session(dataset, transcript, session_id=2)
        with self.assertRaisesRegex(ValueError, "segment count"):
            validate_dataset_session(dataset, transcript, session_id=2, remap_segment_ids=True)
        data["cases"][0]["gold_evidence_ranges"][0]["segment_ids"] = [42, 43]
        complete = EvaluationDataset.model_validate(data)
        self.assertEqual(validate_dataset_session(complete, transcript, session_id=2)["writing"],
                         [[42, 43]])
        data["cases"][0]["gold_evidence_ranges"][0]["end_seconds"] = 119
        with self.assertRaisesRegex(ValueError, "complete transcript"):
            validate_dataset_session(
                EvaluationDataset.model_validate(data), transcript, session_id=2
            )

    def test_committed_dataset_has_required_composition_and_complete_writing_claim(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evals" / "poetry_session_2.json"
        dataset = load_dataset(path)
        self.assertEqual(len(dataset.cases), 12)
        self.assertEqual(Counter(case.category for case in dataset.cases),
                         {"focused": 8, "synthesis": 2, "unanswerable": 2})
        ranges = [gold for case in dataset.cases for gold in case.gold_evidence_ranges]
        self.assertTrue(all(34 <= gold.start_seconds < gold.end_seconds <= 725 for gold in ranges))
        writing = next(
            case for case in dataset.cases if case.id == "oral_writing_printing_transition"
        )
        self.assertTrue(any({109, 110, 111} <= set(gold.segment_ids)
                            for gold in writing.gold_evidence_ranges))


class EvaluationMetricsTests(unittest.TestCase):
    def test_strict_overlap_handles_containment_touching_and_degenerate_intervals(self) -> None:
        for first, second, expected in (
            ((0, 10), (9, 20), True), ((0, 10), (10, 20), False),
            ((0, 10), (-10, 0), False), ((0, 10), (2, 3), True),
            ((0, 10), (5, 5), False), ((0, 10), (6, 4), False),
            ((0, 10), (0, float("inf")), False), ((float("nan"), 10), (0, 5), False),
        ):
            with self.subTest(first=first, second=second):
                self.assertIs(intervals_overlap(first, second), expected)
                self.assertIs(intervals_overlap(second, first), expected)

    def test_metrics_measure_range_hits_without_grading_answer_text(self) -> None:
        data = dataset_payload()
        data["cases"][0]["gold_evidence_ranges"].append({
            "start_seconds": 120, "end_seconds": 130,
            "segment_ids": [43], "explanation": "Another required passage.",
        })
        case = EvaluationDataset.model_validate(data).cases[0]
        metrics = compute_case_metrics(
            case, execution_success=True, evidence_insufficient=False,
            retrieved_ranges=[(105, 106), (110, 120)],
            citation_ranges=[(100, 110), (200, 210)],
        )
        self.assertEqual(metrics["retrieval_evidence_recall"], 0.5)
        self.assertEqual(metrics["citation_recall"], 0.5)
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertIs(metrics["answerability_correct"], True)

    def test_unanswerable_recall_and_no_citation_precision_are_null(self) -> None:
        case = EvaluationDataset.model_validate(dataset_payload()).cases[1]
        for citations, precision in (([], None), ([(100, 110)], 0.0)):
            metrics = compute_case_metrics(
                case, execution_success=True, evidence_insufficient=True,
                retrieved_ranges=[(100, 110)], citation_ranges=citations,
            )
            self.assertIsNone(metrics["retrieval_evidence_recall"])
            self.assertIsNone(metrics["citation_recall"])
            self.assertEqual(metrics["citation_precision"], precision)
            self.assertIs(metrics["answerability_correct"], True)

    def test_failure_and_incorrect_abstention_are_visible_in_denominators(self) -> None:
        answerable, unanswerable = EvaluationDataset.model_validate(dataset_payload()).cases
        results = []
        for case, success, insufficient, seconds in (
            (answerable, False, None, 2), (unanswerable, True, True, 4),
            (answerable, True, True, 6),
        ):
            results.append({
                "processing_seconds": seconds,
                "metrics": compute_case_metrics(
                    case, execution_success=success, evidence_insufficient=insufficient,
                    retrieved_ranges=[], citation_ranges=[],
                ),
            })
        total = aggregate_metrics(results)
        self.assertAlmostEqual(total["execution_success_rate"], 2 / 3)
        self.assertAlmostEqual(total["answerability_accuracy"], 1 / 3)
        self.assertEqual(total["answerability_known_cases"], 2)
        self.assertEqual(total["retrieval_applicable_cases"], 2)
        self.assertEqual(total["retrieval_evidence_recall"], 0)
        self.assertEqual(total["average_processing_seconds"], 4)
        self.assertEqual(total["failed_cases"], 1)
        self.assertIsNone(results[0]["metrics"]["answerability_correct"])
        self.assertIs(results[2]["metrics"]["answerability_correct"], False)


class EvaluationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.database = Database(self.directory / "application.db")
        self.database.initialize()
        self.session_id = self.database.create_session(Path("poetry.wav"), "en", "fake")
        self.database.store_transcription(self.session_id, TranscriptionResult(segments=(
            TranscriptSegment(index=0, start_seconds=100, end_seconds=110,
                              text="Writing fixed poetry."),
        )))
        self.segment_id = self.database.get_transcript(self.session_id)[0].id
        self.dataset_path = self.directory / "dataset.json"
        self.dataset_path.write_text(
            json.dumps(dataset_payload(self.segment_id, self.session_id)), encoding="utf-8"
        )
        self.history = EvaluationDatabase(self.directory / "history.db")

    def run_fixture(self, provider, **options):
        return run_evaluation(
            self.dataset_path, self.database, self.history, self.session_id, provider, **options
        )

    def test_case_failure_retains_partial_retrieval_and_continues_to_next_case(self) -> None:
        provider = ScriptedProvider([
            {"action": "search_transcript", "query": "writing"},
            RuntimeError("provider unavailable"), *insufficient_responses(),
        ])
        report = self.run_fixture(provider, label="failure comparison")
        self.assertEqual(report.status, "completed_with_errors")
        first, second = self.history.get_case_results(report.run_id)
        self.assertEqual(first["execution_success"], 0)
        self.assertIn("provider unavailable", first["error_text"])
        self.assertEqual(json.loads(first["retrieved_segment_ids_json"]), [self.segment_id])
        self.assertEqual(json.loads(first["metrics_json"])["retrieval_evidence_recall"], 1)
        self.assertEqual(json.loads(first["metrics_json"])["citation_recall"], 0)
        self.assertIsNone(first["evidence_insufficient"])
        self.assertEqual(second["execution_success"], 1)
        self.assertEqual(second["evidence_insufficient"], 1)
        self.assertEqual(report.metrics["attempted_cases"], 2)
        self.assertGreaterEqual(first["processing_seconds"], 0)
        self.assertTrue(first["started_at"] <= first["finished_at"])

    def test_runs_append_snapshots_and_finalized_history_is_immutable(self) -> None:
        first = self.run_fixture(
            ScriptedProvider(insufficient_responses()), case_ids=["astronomy"]
        )
        original = self.history.get_run(first.run_id)
        original_cases = self.history.get_case_results(first.run_id)
        second = self.run_fixture(ScriptedProvider(insufficient_responses()), case_ids=["astronomy"])
        self.assertGreater(second.run_id, first.run_id)
        self.assertEqual(self.history.get_run(first.run_id), original)
        self.assertEqual(self.history.get_case_results(first.run_id), original_cases)
        for column in ("dataset_sha256", "transcript_sha256", "summary_sha256"):
            self.assertEqual(len(original[column]), 64)
        self.assertIn("required_points", json.loads(original_cases[0]["case_snapshot_json"]))
        config = json.loads(original["configuration_json"])
        self.assertEqual(config["retrieval_strategy"], "lexical")
        self.assertIn("llm/ollama_provider.py", config["source_sha256"])
        self.assertEqual(json.loads(original["metrics_json"])["successful_cases"], 1)
        with self.assertRaises(ValueError):
            self.history.finish_run(first.run_id, status="completed", finished_at="later",
                                    duration_seconds=0, metrics={})
        with self.assertRaises(ValueError):
            self.history.append_case(first.run_id, {})
        for statement in (
            "UPDATE evaluation_runs SET label='overwritten' WHERE id=?",
            "DELETE FROM evaluation_runs WHERE id=?",
            "UPDATE evaluation_case_results SET answer='overwritten' WHERE run_id=?",
            "DELETE FROM evaluation_case_results WHERE run_id=?",
        ):
            with self.subTest(statement=statement):
                with self.history.connect() as connection:
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(statement, (first.run_id,))

    def test_subset_order_limit_and_preflight_failures_make_no_model_calls(self) -> None:
        provider = ScriptedProvider(insufficient_responses())
        report = self.run_fixture(provider, case_ids=["astronomy", "writing"], limit=1)
        self.assertEqual(self.history.get_case_results(report.run_id)[0]["case_id"], "writing")
        for options in ({"case_ids": ["unknown"]}, {"limit": 0}, {"max_steps": 2},
                        {"agent_mode": "invalid"}, {"summary_run_id": 999}):
            with self.subTest(options=options):
                unused = ScriptedProvider([])
                with self.assertRaises((ValueError, LookupError)):
                    self.run_fixture(unused, **options)
                self.assertEqual(unused.calls, 0)
        data = dataset_payload(self.segment_id, self.session_id)
        data["cases"][0]["gold_evidence_ranges"][0]["segment_ids"] = [999]
        self.dataset_path.write_text(json.dumps(data), encoding="utf-8")
        unused = ScriptedProvider([])
        with self.assertRaisesRegex(ValueError, "segment IDs"):
            self.run_fixture(unused, case_ids=["astronomy"])
        self.assertEqual(unused.calls, 0)

    def test_interruption_persists_case_and_finalizes_run(self) -> None:
        with patch.object(ScriptedProvider, "structured_chat", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fixture(ScriptedProvider([]))
        run = self.history.get_run(1)
        self.assertEqual(run["status"], "interrupted")
        self.assertIsNotNone(run["finished_at"])
        self.assertEqual(len(self.history.get_case_results(1)), 1)
        self.assertIn("KeyboardInterrupt", self.history.get_case_results(1)[0]["error_text"])


if __name__ == "__main__":
    unittest.main()
