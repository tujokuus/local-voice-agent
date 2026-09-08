"""Read-only history snapshots and comparisons, including unfinished runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from local_voice_agent.evaluation.metrics import aggregate_metrics

RUN_COLUMNS = """id, dataset_path, dataset_id, label, source_session_id, model_name,
    agent_mode, started_at, finished_at, duration_seconds, status, dataset_sha256,
    transcript_sha256, summary_sha256, configuration_json, metrics_json, error_text"""


def read_history(
    path: Path, *, run_ids: list[int] | None = None,
    group_id: str | None = None, session_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Never create, migrate, or modify a database while viewing it."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation history was not found: {path}")
    if run_ids and (len(set(run_ids)) != len(run_ids) or any(item < 1 for item in run_ids)):
        raise ValueError("evaluation run IDs must be positive and unique")
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN")
            clauses, parameters = [], []
            if run_ids:
                clauses.append(f"id IN ({','.join('?' for _ in run_ids)})")
                parameters.extend(run_ids)
            if session_id is not None:
                clauses.append("source_session_id=?")
                parameters.append(session_id)
            query = f"SELECT {RUN_COLUMNS} FROM evaluation_runs"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            rows = connection.execute(query + " ORDER BY id DESC", parameters).fetchall()
            runs = []
            for row in rows:
                run = dict(row)
                run["configuration"] = json.loads(run["configuration_json"])
                if group_id and run["configuration"].get("comparison_group_id") != group_id:
                    continue
                if limit is not None and len(runs) >= limit:
                    break
                cases = connection.execute(
                    "SELECT * FROM evaluation_case_results WHERE run_id=? ORDER BY id", (run["id"],)
                ).fetchall()
                run["cases"] = [dict(case) for case in cases]
                for case in run["cases"]:
                    case["metrics"] = json.loads(case["metrics_json"])
                    case["snapshot"] = json.loads(case["case_snapshot_json"])
                run["metrics"] = (
                    json.loads(run["metrics_json"]) if run["metrics_json"]
                    else aggregate_metrics(run["cases"])
                )
                runs.append(run)
        finally:
            connection.close()
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read evaluation history: {exc}") from exc
    if run_ids:
        by_id = {run["id"]: run for run in runs}
        missing = set(run_ids) - by_id.keys()
        if missing:
            raise LookupError(f"Evaluation runs not found: {', '.join(map(str, sorted(missing)))}")
        return [by_id[item] for item in run_ids]
    if group_id and not runs:
        raise LookupError(f"Comparison group was not found: {group_id}")
    return runs


def comparison_mismatches(runs: list[dict[str, Any]]) -> list[str]:
    baseline = runs[0]
    differences = []
    for run in runs[1:]:
        fields = [
            name for name in ("dataset_sha256", "transcript_sha256", "summary_sha256")
            if run[name] != baseline[name]
        ]
        for name in ("selected_case_ids", "metric_version", "max_steps",
                     "timeout_seconds_per_request"):
            if run["configuration"].get(name) != baseline["configuration"].get(name):
                fields.append(name)
        # Also catch changed stored gold answers/questions, including legacy imported runs.
        base_cases = {case["case_id"]: case["snapshot"] for case in baseline["cases"]}
        for case in run["cases"]:
            if case["case_id"] in base_cases and case["snapshot"] != base_cases[case["case_id"]]:
                fields.append(f"case:{case['case_id']}")
        if fields:
            differences.append(f"Run {run['id']} vs {baseline['id']}: {', '.join(fields)}")
    return differences
