"""Deterministic evidence checks; these do not grade natural-language correctness."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite
from statistics import mean
from typing import Any

from local_voice_agent.evaluation.dataset import EvaluationCase

TimeRange = tuple[float, float]


def intervals_overlap(first: TimeRange, second: TimeRange) -> bool:
    """Touching endpoints are not overlap."""
    if not all(isfinite(value) for value in (*first, *second)):
        return False
    if first[0] >= first[1] or second[0] >= second[1]:
        return False
    return first[0] < second[1] and first[1] > second[0]


def compute_case_metrics(
    case: EvaluationCase,
    *,
    execution_success: bool,
    evidence_insufficient: bool | None,
    retrieved_ranges: Sequence[TimeRange],
    citation_ranges: Sequence[TimeRange],
) -> dict[str, Any]:
    gold = [(item.start_seconds, item.end_seconds) for item in case.gold_evidence_ranges]
    retrieval_hits = sum(
        any(intervals_overlap(item, row) for row in retrieved_ranges) for item in gold
    )
    citation_hits = sum(
        any(intervals_overlap(item, row) for row in citation_ranges) for item in gold
    )
    supported_citations = sum(
        any(intervals_overlap(citation, item) for item in gold) for citation in citation_ranges
    )
    return {
        "execution_success": execution_success,
        "answerability_correct": (
            evidence_insufficient == (not case.answerable)
            if execution_success and evidence_insufficient is not None else None
        ),
        "gold_range_count": len(gold),
        "retrieval_hit_count": retrieval_hits,
        "citation_hit_count": citation_hits,
        "supported_citation_count": supported_citations,
        "citation_count": len(citation_ranges),
        "retrieval_evidence_recall": retrieval_hits / len(gold) if gold else None,
        "citation_recall": citation_hits / len(gold) if gold else None,
        "citation_precision": (
            supported_citations / len(citation_ranges) if citation_ranges else None
        ),
    }


def aggregate_metrics(results: Iterable[dict[str, Any]]) -> dict[str, int | float | None]:
    """Macro means over applicable cases; failures count in attempted denominators.

    Missing answerability on a failed/unknown output is null per case and earns
    no credit in aggregate accuracy. Partial evidence on failures is retained.
    Unanswerable recalls and precision without citations are undefined (null).
    """
    cases = list(results)
    attempted = len(cases)
    successful = sum(bool(case["metrics"]["execution_success"]) for case in cases)
    correct = sum(case["metrics"]["answerability_correct"] is True for case in cases)
    output: dict[str, int | float | None] = {
        "attempted_cases": attempted,
        "successful_cases": successful,
        "failed_cases": attempted - successful,
        "execution_success_rate": successful / attempted if attempted else None,
        "answerability_accuracy": correct / attempted if attempted else None,
        "answerability_known_cases": sum(
            case["metrics"]["answerability_correct"] is not None for case in cases
        ),
        "answerability_correct_cases": correct,
        "average_processing_seconds": (
            mean(case["processing_seconds"] for case in cases) if cases else None
        ),
    }
    for key, denominator in (
        ("retrieval_evidence_recall", "retrieval_applicable_cases"),
        ("citation_recall", "citation_recall_applicable_cases"),
        ("citation_precision", "citation_precision_applicable_cases"),
    ):
        values = [case["metrics"][key] for case in cases if case["metrics"][key] is not None]
        output[key] = mean(values) if values else None
        output[denominator] = len(values)
    return output
