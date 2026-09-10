"""Sequential direct-code evaluation, preserving every case and partial telemetry."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from local_voice_agent.agent import MAX_AGENT_STEPS, QuestionAnswer
from local_voice_agent.agent_modes import agent_class_for
from local_voice_agent.evaluation.database import EvaluationDatabase, canonical_json
from local_voice_agent.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
    validate_dataset_session,
)
from local_voice_agent.evaluation.metrics import aggregate_metrics, compute_case_metrics
from local_voice_agent.llm import LLMProvider
from local_voice_agent.models import SessionStatus, StoredTranscriptSegment
from local_voice_agent.storage import Database
from local_voice_agent.summaries import StoredFinalSummary, StoredSummaryCheckpoint


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: int
    status: str
    duration_seconds: float
    metrics: dict[str, int | float | None]


@dataclass(frozen=True, slots=True)
class PreparedEvaluation:
    dataset_bytes: bytes
    dataset: EvaluationDataset
    cases: tuple[EvaluationCase, ...]
    transcript: list[StoredTranscriptSegment]
    summary: StoredFinalSummary | None
    checkpoints: list[StoredSummaryCheckpoint]
    resolved_ids: dict[str, list[list[int]]]
    transcript_json: str
    summary_json: str


def prepare_evaluation(
    dataset_path: Path, database: Database, session_id: int, *,
    case_ids: Sequence[str] | None = None, limit: int | None = None,
    summary_run_id: int | None = None, remap_segment_ids: bool = False,
) -> PreparedEvaluation:
    """Validate once and retain the same input snapshots for every matrix member."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    dataset_bytes = dataset_path.read_bytes()
    dataset = EvaluationDataset.model_validate_json(dataset_bytes)
    selected_ids = set(case_ids or [])
    unknown_ids = selected_ids - {case.id for case in dataset.cases}
    if unknown_ids:
        raise ValueError(f"Unknown evaluation case IDs: {', '.join(sorted(unknown_ids))}")
    cases = [case for case in dataset.cases if not selected_ids or case.id in selected_ids]
    if limit is not None:
        cases = cases[:limit]
    database.initialize()
    session = database.get_session(session_id)
    if session is None:
        raise LookupError(f"Session {session_id} was not found")
    if session.status is not SessionStatus.COMPLETED:
        raise ValueError(f"Session {session_id} is not completed")
    transcript = database.get_transcript(session_id)
    resolved_ids = validate_dataset_session(
        dataset, transcript, session_id=session_id, remap_segment_ids=remap_segment_ids
    )
    summary = database.get_final_summary(session_id, summary_run_id)
    if summary_run_id is not None and summary is None:
        raise LookupError(f"Summary run {summary_run_id} was not found for session {session_id}")
    checkpoints = database.get_summary_checkpoints(session_id, summary.id) if summary else []
    transcript_json = canonical_json([row.model_dump(mode="json") for row in transcript])
    summary_json = canonical_json({
        "summary": summary.model_dump(mode="json") if summary else None,
        "checkpoints": [checkpoint.model_dump(mode="json") for checkpoint in checkpoints],
    })
    return PreparedEvaluation(
        dataset_bytes, dataset, tuple(cases), transcript, summary, checkpoints,
        resolved_ids, transcript_json, summary_json,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def _code_configuration() -> dict[str, Any]:
    package = Path(__file__).resolve().parent.parent
    sources = [
        package / "agent.py", package / "retrieval.py", package / "llm" / "ollama_provider.py",
        package / "agent_modes.py", package / "pydantic_agent.py",
        *sorted(Path(__file__).parent.glob("*.py")),
    ]
    try:
        application_version = version("local-voice-agent")
    except PackageNotFoundError:
        application_version = "uninstalled-source"
    return {
        "python_version": platform.python_version(),
        "application_version": application_version,
        "source_sha256": {
            str(path.relative_to(package)).replace("\\", "/"): _sha256(path.read_bytes())
            for path in sources
        },
    }


def run_evaluation(
    dataset_path: Path,
    database: Database,
    evaluation_database: EvaluationDatabase,
    session_id: int,
    provider: LLMProvider,
    agent_mode: str = "final",
    label: str | None = None,
    case_ids: Sequence[str] | None = None,
    limit: int | None = None,
    max_steps: int = 5,
    summary_run_id: int | None = None,
    configuration: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    remap_segment_ids: bool = False,
    _prepared: PreparedEvaluation | None = None,
) -> RunReport:
    """Preflight the entire dataset, then isolate each case with a fresh agent.

    Model/provider failures are persisted and the next case proceeds. An explicit
    interruption persists partial telemetry, finalizes history, and propagates.
    """
    agent_class = agent_class_for(agent_mode)
    minimum_steps = 1 if agent_mode == "manual" else 3
    if not minimum_steps <= max_steps <= MAX_AGENT_STEPS:
        raise ValueError(f"max_steps for {agent_mode} must be {minimum_steps}..{MAX_AGENT_STEPS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    same_path = database.path.resolve() == evaluation_database.path.resolve()
    same_file = (
        database.path.exists() and evaluation_database.path.exists()
        and database.path.samefile(evaluation_database.path)
    )
    if same_path or same_file:
        raise ValueError("the evaluation database must be separate from the application database")
    prepared = _prepared or prepare_evaluation(
        dataset_path, database, session_id, case_ids=case_ids, limit=limit,
        summary_run_id=summary_run_id, remap_segment_ids=remap_segment_ids,
    )
    dataset, cases = prepared.dataset, prepared.cases
    transcript, summary = prepared.transcript, prepared.summary
    checkpoints = prepared.checkpoints
    config = {
        **(configuration or {}),
        **_code_configuration(),
        "retrieval_strategy": "lexical",
        "max_steps": max_steps,
        "requested_summary_run_id": summary_run_id,
        "resolved_summary_run_id": summary.id if summary else None,
        "selected_case_ids": [case.id for case in cases],
        "remap_segment_ids": remap_segment_ids,
        "metric_version": 1,
        "application_database_path": str(database.path.resolve()),
    }
    if agent_mode == "pydanticai":
        config["pydantic_ai_version"] = version("pydantic-ai-slim")
        config["model_transport"] = "ollama-openai-compatible-v1"
    evaluation_database.initialize()
    started_at = _now()
    started = perf_counter()
    run_id = evaluation_database.start_run({
        "dataset_path": str(dataset_path.resolve()),
        "dataset_id": dataset.id,
        "label": label.strip() or None if label is not None else None,
        "source_session_id": session_id,
        "model_name": provider.model_name,
        "agent_mode": agent_mode,
        "started_at": started_at,
        "dataset_sha256": _sha256(prepared.dataset_bytes),
        "transcript_sha256": _sha256(prepared.transcript_json),
        "summary_sha256": _sha256(prepared.summary_json),
        "dataset_snapshot_json": canonical_json({
            "dataset": dataset.model_dump(mode="json"),
            "resolved_gold_segment_ids": prepared.resolved_ids,
        }),
        "transcript_snapshot_json": prepared.transcript_json,
        "summary_snapshot_json": prepared.summary_json,
        "configuration_json": canonical_json(config),
    })
    rows_by_id = {row.id: row for row in transcript}
    results: list[dict[str, Any]] = []
    status = "failed"
    run_error: str | None = None
    try:
        if progress:
            progress(
                f"Evaluation run {run_id}: {len(cases)} cases ({agent_mode}, {provider.model_name})"
            )
        for index, case in enumerate(cases, 1):
            if progress:
                progress(f"[{index}/{len(cases)}] {case.id}: {case.question}")
            agent = agent_class(
                provider=provider, transcript=transcript, summary=summary,
                checkpoints=checkpoints, max_steps=max_steps,
            )
            case_started_at = _now()
            case_started = perf_counter()
            error: BaseException | None = None
            answer: QuestionAnswer | None = None
            try:
                answer = agent.answer(case.question)
            except (Exception, KeyboardInterrupt) as exc:
                error = exc
                answer = agent.last_telemetry
            elapsed = perf_counter() - case_started
            retrieved_ids = list(answer.retrieved_segment_ids) if answer else []
            retrieved_rows = [rows_by_id[item] for item in retrieved_ids if item in rows_by_id]
            citations = list(answer.citation_ranges) if answer else []
            insufficient = answer.evidence_insufficient if answer else None
            metrics = compute_case_metrics(
                case, execution_success=error is None, evidence_insufficient=insufficient,
                retrieved_ranges=[(row.start_seconds, row.end_seconds) for row in retrieved_rows],
                citation_ranges=citations,
            )
            result = {
                "case_id": case.id, "question": case.question,
                "answer": answer.answer if answer else None,
                "execution_success": error is None,
                "error_text": f"{type(error).__name__}: {error}" if error is not None else None,
                "started_at": case_started_at, "finished_at": _now(),
                "processing_seconds": elapsed,
                "evidence_insufficient": insufficient,
                "retrieved_segment_ids": retrieved_ids,
                "retrieved_ranges": [{
                    "segment_id": row.id, "start_seconds": row.start_seconds,
                    "end_seconds": row.end_seconds,
                } for row in retrieved_rows],
                "cited_segment_ids": list(answer.cited_segment_ids) if answer else [],
                "citation_ranges": [{"start_seconds": start, "end_seconds": end}
                                    for start, end in citations],
                "tool_calls": list(answer.tool_calls) if answer else [],
                "step_count": answer.step_count if answer else 0,
                "case_snapshot": case.model_dump(mode="json"), "metrics": metrics,
            }
            evaluation_database.append_case(run_id, result)
            results.append(result)
            if progress:
                outcome = "completed" if error is None else f"failed ({result['error_text']})"
                progress(f"  {outcome}; {elapsed:.2f}s")
            if isinstance(error, KeyboardInterrupt):
                raise error
        status = (
            "completed_with_errors"
            if any(not row["execution_success"] for row in results) else "completed"
        )
    except KeyboardInterrupt:
        status = "interrupted"
        run_error = "KeyboardInterrupt: evaluation interrupted"
        raise
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = perf_counter() - started
        totals = aggregate_metrics(results)
        totals["selected_cases"] = len(cases)
        evaluation_database.finish_run(
            run_id, status=status, finished_at=_now(), duration_seconds=duration,
            metrics=totals, error_text=run_error,
        )
    return RunReport(run_id=run_id, status=status, duration_seconds=duration, metrics=totals)
