from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

from local_voice_agent.agent import MAX_AGENT_STEPS
from local_voice_agent.config import Settings
from local_voice_agent.evaluation.database import EvaluationDatabase
from local_voice_agent.evaluation.history import read_history
from local_voice_agent.evaluation.matrix import run_matrix
from local_voice_agent.evaluation.reporting import compare_runs, run_table, show_run
from local_voice_agent.evaluation.runner import run_evaluation
from local_voice_agent.llm import OllamaProvider
from local_voice_agent.storage import Database

DEFAULT_EVALUATION_DATABASE = Path("data/local_voice_agent_evaluations.db")


def _execution_options(
    parser: argparse.ArgumentParser, settings: Settings, *, matrix: bool = False,
) -> None:
    parser.add_argument("dataset", type=Path, help="Validated evaluation dataset JSON")
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument(
        "--evaluation-database", type=Path, default=DEFAULT_EVALUATION_DATABASE,
        help=f"Separate history SQLite path (default: {DEFAULT_EVALUATION_DATABASE})",
    )
    if matrix:
        parser.add_argument("--models", nargs="+", default=["qwen3.5:4b", "qwen3.5:9b"])
        parser.add_argument(
            "--agent-modes", nargs="+", choices=("final", "manual", "pydanticai"),
            default=["final", "manual"]
        )
    else:
        parser.add_argument("--model", default=settings.ollama_model)
        parser.add_argument(
            "--agent-mode", choices=("final", "manual", "pydanticai"), default="final"
        )
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


def add_evaluation_parser(subparsers: argparse._SubParsersAction, settings: Settings) -> None:
    _execution_options(subparsers.add_parser(
        "evaluate", help="Run a dataset against a stored transcript and append evaluation history"
    ), settings)
    _execution_options(subparsers.add_parser(
        "evaluate-matrix", help="Run all selected model/mode combinations with frozen input data"
    ), settings, matrix=True)
    runs = subparsers.add_parser("evaluation-runs", help="List stored evaluation runs")
    runs.add_argument("--session-id", type=int)
    runs.add_argument("--group-id")
    runs.add_argument("--limit", type=int, help="Show only the newest N matching runs")
    show = subparsers.add_parser("evaluation-show", help="Show questions, answers and references")
    show.add_argument("evaluation_run_id", type=int)
    show.add_argument("--case-id", help="Show only one selected case")
    compare = subparsers.add_parser("evaluation-compare", help="Compare runs and per-case metrics")
    compare.add_argument("evaluation_run_ids", type=int, nargs="*")
    compare.add_argument("--group-id", help="Compare a matrix group instead of explicit run IDs")
    compare.add_argument("--case-id", help="Also print this case's answers and reference points")
    compare.add_argument("--allow-mismatch", action="store_true")
    for parser in (runs, show, compare):
        parser.add_argument(
            "--evaluation-database", type=Path, default=DEFAULT_EVALUATION_DATABASE,
        )


def _percentage(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _execution_paths(args: argparse.Namespace, settings: Settings) -> tuple[Path, Path]:
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
    return source_path, history_path


def evaluate(args: argparse.Namespace, settings: Settings) -> int:
    source_path, history_path = _execution_paths(args, settings)

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


def evaluate_matrix(args: argparse.Namespace, settings: Settings) -> int:
    source_path, history_path = _execution_paths(args, settings)
    try:
        report = run_matrix(
            args.dataset, Database(source_path), EvaluationDatabase(history_path), args.session_id,
            lambda model: OllamaProvider(
                model_name=model, base_url=args.ollama_url, timeout_seconds=args.timeout,
            ),
            models=args.models, agent_modes=args.agent_modes, label=args.label,
            case_ids=args.case_ids, limit=args.limit, max_steps=args.max_steps,
            summary_run_id=args.run_id, remap_segment_ids=args.remap_segment_ids,
            configuration={
                "ollama_url": args.ollama_url, "timeout_seconds_per_request": args.timeout,
            },
            progress=lambda message: print(message, flush=True),
        )
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(f"Evaluation matrix could not read or store its data: {exc}") from exc
    runs = read_history(history_path, run_ids=[run.run_id for run in report.runs])
    print("\n" + (compare_runs(runs) if len(runs) > 1 else run_table(runs)))
    print(f"\nComparison group: {report.group_id}")
    print(f"History: {history_path}")
    return 0 if all(run.status == "completed" for run in report.runs) else 1


def evaluation_runs(args: argparse.Namespace, _: Settings) -> int:
    print(run_table(read_history(
        args.evaluation_database, group_id=args.group_id,
        session_id=args.session_id, limit=args.limit,
    )))
    return 0


def evaluation_show(args: argparse.Namespace, _: Settings) -> int:
    run = read_history(args.evaluation_database, run_ids=[args.evaluation_run_id])[0]
    print(show_run(run, args.case_id))
    return 0


def evaluation_compare(args: argparse.Namespace, _: Settings) -> int:
    if bool(args.group_id) == bool(args.evaluation_run_ids):
        raise ValueError("provide either two or more evaluation run IDs, or --group-id")
    runs = read_history(
        args.evaluation_database, run_ids=args.evaluation_run_ids or None, group_id=args.group_id,
    )
    if args.group_id:
        runs.sort(key=lambda run: run["id"])
        expected = runs[0]["configuration"].get("comparison_expected_combinations", [])
        if expected and len(runs) != len(expected):
            print(f"Partial comparison group: {len(runs)}/{len(expected)} runs have started.")
    print(compare_runs(runs, allow_mismatch=args.allow_mismatch, case_id=args.case_id))
    return 0
