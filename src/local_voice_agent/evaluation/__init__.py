"""Reproducible transcript question-answer evaluation without an LLM judge."""

from local_voice_agent.evaluation.database import EvaluationDatabase
from local_voice_agent.evaluation.dataset import (
    EvaluationCase,
    EvaluationDataset,
    GoldEvidenceRange,
    load_dataset,
    validate_dataset_session,
)
from local_voice_agent.evaluation.runner import RunReport, run_evaluation

__all__ = [
    "EvaluationCase", "EvaluationDatabase", "EvaluationDataset", "GoldEvidenceRange",
    "RunReport", "load_dataset", "run_evaluation", "validate_dataset_session",
]
