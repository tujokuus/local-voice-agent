from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

from local_voice_agent.agent import MAX_AGENT_STEPS
from local_voice_agent.config import Settings
from local_voice_agent.evaluation.database import EvaluationDatabase
from local_voice_agent.evaluation.runner import run_evaluation
from local_voice_agent.llm import OllamaProvider
from local_voice_agent.storage import Database

DEFAULT_EVALUATION_DATABASE = Path("data/local_voice_agent_evaluations.db")


def add_evaluation_parser(subparsers: argparse._SubParsersAction, settings: Settings) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="Run a dataset against a stored transcript and append evaluation history"
    )
    parser.add_argument("dataset", type=Path, help="Validated evaluation dataset JSON")
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument(
        "--evaluation-database", type=Path, default=DEFAULT_EVALUATION_DATABASE,
        help=f"Separate history SQLite path (default: {DEFAULT_EVALUATION_DATABASE})",
    )
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--agent-mode", choices=("final", "manual"), default="final")
    parser.add_argument("--label", help="Name for this comparison run")
    parser.add_argument(
        "--case-id", action="append", dest="case_ids",
        help="Select a case ID; repeat to select several (dataset order is preserved)",
    )
    parser.add_argument("--limit", type=int, help="Run at most this many selected cases")
    parser.add_argument("--ollama-url", default=settings.ollama_base_url)
    parser.add_argument(
        "--timeout", type=float, default=settings.ollama_timeout_seconds,
        help="Timeout in seconds for each model HTTP request, not the entire case",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5,
        help=f"Maximum agent steps, up to {MAX_AGENT_STEPS} (default: 5)",
    )
    parser.add_argument(
        "--run-id", type=int,
        help="Pin a stored summary run for search orientation; defaults to the latest",
    )
    parser.add_argument(
        "--remap-segment-ids", action="store_true",
        help="Resolve gold audit IDs by timestamps when evaluating an imported transcript",
    )


def _percentage(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def evaluate(args: argparse.Namespace, settings: Settings) -> int:
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and greater than zero")
    source_path = (args.database or settings.database_path).expanduser().resolve()
    history_path = args.evaluation_database.expanduser().resolve()
    if source_path == history_path or (
        source_path.exists() and history_path.exists() and source_path.samefile(history_path)
    ):
        raise ValueError("--evaluation-database must differ from the application --database")
    if not source_path.is_file():
        raise FileNotFoundError(f"Application database was not found: {source_path}")

    provider = OllamaProvider(
        model_name=args.model,
        base_url=args.ollama_url,
        timeout_seconds=args.timeout,
    )
    print(
        f"Evaluating session {args.session_id} with {args.model} "
        f"(agent_mode={args.agent_mode})...",
        flush=True,
    )
    try:
        report = run_evaluation(
            dataset_path=args.dataset,
            database=Database(source_path),
            evaluation_database=EvaluationDatabase(history_path),
            session_id=args.session_id,
            provider=provider,
            agent_mode=args.agent_mode,
            label=args.label,
            case_ids=args.case_ids,
            limit=args.limit,
            max_steps=args.max_steps,
            summary_run_id=args.run_id,
            remap_segment_ids=args.remap_segment_ids,
            configuration={
                "ollama_url": args.ollama_url,
                "timeout_seconds_per_request": args.timeout,
            },
            progress=lambda message: print(message, flush=True),
        )
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(f"Evaluation could not read or store its data: {exc}") from exc

    metrics = report.metrics
    print(f"\nEvaluation run {report.run_id}: {report.status}")
    print(f"History: {history_path}")
    print(
        f"Cases: {metrics['successful_cases']} successful, {metrics['failed_cases']} failed "
        f"/ {metrics['attempted_cases']} attempted"
    )
    print(
        f"Execution success: {_percentage(metrics['execution_success_rate'])} | "
        f"Answerability accuracy: {_percentage(metrics['answerability_accuracy'])}"
    )
    print(
        f"Retrieval recall: {_percentage(metrics['retrieval_evidence_recall'])} | "
        f"Citation recall: {_percentage(metrics['citation_recall'])} | "
        f"Citation precision: {_percentage(metrics['citation_precision'])}"
    )
    print(
        f"Average case time: {metrics['average_processing_seconds']:.2f}s | "
        f"Total run time: {report.duration_seconds:.2f}s"
    )
    return 0 if report.status == "completed" else 1
