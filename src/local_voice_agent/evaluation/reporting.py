"""Human-readable reports; no answer-content grading is implied by these metrics."""

from __future__ import annotations

import json
from typing import Any

from local_voice_agent.evaluation.history import comparison_mismatches

METRICS = (
    ("execution_success_rate", "Execution success"),
    ("answerability_accuracy", "Answerability accuracy"),
    ("retrieval_evidence_recall", "Retrieval recall"),
    ("citation_recall", "Citation recall"),
    ("citation_precision", "Citation precision"),
    ("average_processing_seconds", "Average case time"),
)


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    return "\n".join(
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in [headers, ["-" * width for width in widths], *rows]
    )


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def run_table(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "No evaluation runs found."
    rows = []
    for run in runs:
        metrics, config = run["metrics"], run["configuration"]
        attempted = metrics["attempted_cases"]
        selected = len(config.get("selected_case_ids", []))
        rows.append([
            str(run["id"]), run["started_at"], run["label"] or "-", run["model_name"],
            run["agent_mode"], run["status"], f"{attempted}/{selected or '?'}",
            f"{metrics['successful_cases']}/{metrics['failed_cases']}",
            percentage(metrics["answerability_accuracy"]),
            str(config.get("comparison_group_id", "-")),
        ])
    return table(
        ["Run", "Started", "Label", "Model", "Mode", "Status", "Cases", "OK/Fail",
         "Answerability", "Group"], rows,
    )


def _ranges(value: list[dict]) -> str:
    return ", ".join(
        f"{item['start_seconds']:g}-{item['end_seconds']:g}s"
        + (f" (segments {item['segment_ids']})" if "segment_ids" in item else "")
        for item in value
    ) or "none"


def show_run(run: dict[str, Any], case_id: str | None = None) -> str:
    cases = run["cases"]
    selected = run["configuration"].get("selected_case_ids", [])
    if case_id:
        cases = [case for case in cases if case["case_id"] == case_id]
        if not cases and case_id not in selected:
            raise LookupError(f"Case {case_id} was not selected in evaluation run {run['id']}")
    lines = [run_table([run]), "", "Answer content is not automatically graded."]
    if run["status"] == "running":
        lines.append("Partial stored results; 'running' does not confirm a live process.")
    if run["error_text"]:
        lines.append(f"Run error: {run['error_text']}")
    for case in cases:
        snapshot, metrics = case["snapshot"], case["metrics"]
        state = {None: "unknown", 0: "false", 1: "true"}[case["evidence_insufficient"]]
        lines.extend([
            "", f"Case: {case['case_id']}", f"Question: {case['question']}",
            f"Status: {'completed' if case['execution_success'] else 'failed'} | "
            f"Time: {case['processing_seconds']:.2f}s | Evidence insufficient: {state}",
            f"Gold answerable: {snapshot['answerable']}",
            "Generated answer:", case["answer"] or "(no accepted answer)",
            "Reference answer:", snapshot["reference_answer"], "Required points:",
            *(f"- {point}" for point in snapshot["required_points"]),
            "Gold evidence: " + _ranges(snapshot["gold_evidence_ranges"]),
            "Retrieved evidence: " + _ranges(json.loads(case["retrieved_ranges_json"])),
            "Citations: " + _ranges(json.loads(case["citation_ranges_json"])),
            f"Answerability correct: {metrics['answerability_correct']} | "
            f"Retrieval recall: {percentage(metrics['retrieval_evidence_recall'])} | "
            f"Citation recall: {percentage(metrics['citation_recall'])} | "
            f"Citation precision: {percentage(metrics['citation_precision'])}",
        ])
        if case["error_text"]:
            lines.append(f"Error: {case['error_text']}")
    stored = {case["case_id"] for case in run["cases"]}
    missing = [item for item in selected if item not in stored and (not case_id or item == case_id)]
    if missing:
        lines.append("\nNo stored result yet: " + ", ".join(missing))
    return "\n".join(lines)


def compare_runs(
    runs: list[dict[str, Any]], *, allow_mismatch: bool = False, case_id: str | None = None,
) -> str:
    if len(runs) < 2:
        raise ValueError("comparison requires at least two evaluation runs")
    mismatches = comparison_mismatches(runs)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "Runs have different evaluation inputs/settings:\n" + "\n".join(mismatches)
            + "\nUse --allow-mismatch for an explicitly non-equivalent comparison."
        )
    lines = [run_table(runs), ""]
    if mismatches:
        lines.extend(["NON-EQUIVALENT COMPARISON", *mismatches, ""])
    if any(run["status"] not in {"completed", "completed_with_errors"} for run in runs):
        lines.append("Partial/unfinished runs included; denominators may differ.")
    if len({json.dumps(run["configuration"].get("source_sha256"), sort_keys=True)
            for run in runs}) > 1:
        lines.append("Implementation fingerprints differ between runs.")
    lines.append(f"Deltas are relative to run {runs[0]['id']}; pp = percentage points.")
    if case_id:
        lines.append("Aggregate metrics cover the full run; --case-id filters case details only.")
    rows = []
    for key, title in METRICS:
        row = [title]
        baseline = runs[0]["metrics"][key]
        for index, run in enumerate(runs):
            value = run["metrics"][key]
            is_time = key == "average_processing_seconds"
            text = "n/a" if value is None else (f"{value:.2f}s" if is_time else percentage(value))
            if index and value is not None and baseline is not None:
                delta = value - baseline
                text += f" ({delta:+.2f}s)" if is_time else f" ({delta * 100:+.1f}pp)"
            row.append(text)
        rows.append(row)
    lines.extend([table(["Metric", *(str(run["id"]) for run in runs)], rows), ""])
    lines.append("Per case: exec / answerability / retrieval / citation recall / precision / time")
    by_run = [{case["case_id"]: case for case in run["cases"]} for run in runs]
    ids = list(dict.fromkeys(
        item for run in runs for item in [
            *run["configuration"].get("selected_case_ids", []),
            *(case["case_id"] for case in run["cases"]),
        ]
    ))
    if case_id:
        if case_id not in ids:
            raise LookupError(f"Case was not selected in these runs: {case_id}")
        ids = [case_id]
    rows = []
    for item in ids:
        row = [item]
        for cases in by_run:
            case = cases.get(item)
            if case is None:
                row.append("no stored result")
                continue
            metrics = case["metrics"]
            answerability = {None: "?", False: "wrong", True: "correct"}[
                metrics["answerability_correct"]
            ]
            row.append(" / ".join([
                "ok" if case["execution_success"] else "error", answerability,
                *(percentage(metrics[key]) for key in
                  ("retrieval_evidence_recall", "citation_recall", "citation_precision")),
                f"{case['processing_seconds']:.2f}s",
            ]))
        rows.append(row)
    lines.append(table(["Case", *(str(run["id"]) for run in runs)], rows))
    if case_id:
        for run in runs:
            if case_id in run["configuration"].get("selected_case_ids", []) or any(
                case["case_id"] == case_id for case in run["cases"]
            ):
                lines.extend(["", show_run(run, case_id)])
    lines.append(
        "\nThese scores do not grade answer content; inspect reference answers and points."
    )
    return "\n".join(lines)
