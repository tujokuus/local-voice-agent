"""Sequential model/mode comparisons with one frozen set of evaluation inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from local_voice_agent.agent import MAX_AGENT_STEPS
from local_voice_agent.evaluation.database import EvaluationDatabase
from local_voice_agent.evaluation.runner import RunReport, prepare_evaluation, run_evaluation
from local_voice_agent.llm import LLMProvider
from local_voice_agent.storage import Database


@dataclass(frozen=True, slots=True)
class MatrixReport:
    group_id: str
    runs: tuple[RunReport, ...]


def run_matrix(
    dataset_path: Path, database: Database, evaluation_database: EvaluationDatabase,
    session_id: int, provider_factory: Callable[[str], LLMProvider], *,
    models: Sequence[str] = ("qwen3.5:4b", "qwen3.5:9b"),
    agent_modes: Sequence[str] = ("final", "manual"), label: str | None = None,
    case_ids: Sequence[str] | None = None, limit: int | None = None,
    max_steps: int = 5, summary_run_id: int | None = None,
    remap_segment_ids: bool = False, configuration: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> MatrixReport:
    if not models or len(set(models)) != len(models) or any(not name.strip() for name in models):
        raise ValueError("models must be nonempty and unique")
    if not agent_modes or len(set(agent_modes)) != len(agent_modes) or any(
        mode not in {"final", "manual"} for mode in agent_modes
    ):
        raise ValueError("agent modes must be unique choices of final/manual")
    minimum = 3 if "final" in agent_modes else 1
    if not minimum <= max_steps <= MAX_AGENT_STEPS:
        raise ValueError(f"max_steps must be {minimum}..{MAX_AGENT_STEPS}")
    if database.path.resolve() == evaluation_database.path.resolve() or (
        database.path.exists() and evaluation_database.path.exists()
        and database.path.samefile(evaluation_database.path)
    ):
        raise ValueError("evaluation database must be separate from the application database")
    prepared = prepare_evaluation(
        dataset_path, database, session_id, case_ids=case_ids, limit=limit,
        summary_run_id=summary_run_id, remap_segment_ids=remap_segment_ids,
    )
    group_id = str(uuid4())
    combinations = [
        {"model": model, "agent_mode": mode} for model in models for mode in agent_modes
    ]
    if progress:
        progress(f"Comparison group: {group_id}")
        progress(
            f"{len(combinations)} combinations x {len(prepared.cases)} cases = "
            f"{len(combinations) * len(prepared.cases)} question attempts; sequential execution."
        )
    reports = []
    for combination in combinations:
        model, mode = combination["model"], combination["agent_mode"]
        provider = provider_factory(model)
        reports.append(run_evaluation(
            dataset_path, database, evaluation_database, session_id, provider,
            agent_mode=mode, label=f"{label or 'matrix'} | {model} | {mode}",
            case_ids=case_ids, limit=limit, max_steps=max_steps,
            summary_run_id=summary_run_id, remap_segment_ids=remap_segment_ids,
            configuration={
                **(configuration or {}), "comparison_group_id": group_id,
                "comparison_group_label": label,
                "comparison_expected_combinations": combinations,
            },
            progress=progress, _prepared=prepared,
        ))
    return MatrixReport(group_id, tuple(reports))
