"""Separate SQLite history with immutable cases and one run-finalization transition."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class EvaluationDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._active_runs: set[int] = set()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA recursive_triggers = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_path TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    label TEXT,
                    source_session_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    agent_mode TEXT NOT NULL CHECK (agent_mode IN ('final', 'manual')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL CHECK (duration_seconds >= 0),
                    status TEXT NOT NULL DEFAULT 'running' CHECK (
                        status IN (
                            'running','completed','completed_with_errors','interrupted','failed'
                        )
                    ),
                    dataset_sha256 TEXT NOT NULL,
                    transcript_sha256 TEXT NOT NULL,
                    summary_sha256 TEXT NOT NULL,
                    dataset_snapshot_json TEXT NOT NULL,
                    transcript_snapshot_json TEXT NOT NULL,
                    summary_snapshot_json TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    metrics_json TEXT,
                    error_text TEXT
                );
                CREATE TABLE IF NOT EXISTS evaluation_case_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL REFERENCES evaluation_runs(id),
                    case_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    execution_success INTEGER NOT NULL CHECK (execution_success IN (0, 1)),
                    error_text TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    processing_seconds REAL NOT NULL CHECK (processing_seconds >= 0),
                    evidence_insufficient INTEGER CHECK (evidence_insufficient IN (0, 1)),
                    retrieved_segment_ids_json TEXT NOT NULL,
                    retrieved_ranges_json TEXT NOT NULL,
                    cited_segment_ids_json TEXT NOT NULL,
                    citation_ranges_json TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    case_snapshot_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    UNIQUE (run_id, case_id)
                );
                CREATE TRIGGER IF NOT EXISTS evaluation_cases_no_update
                BEFORE UPDATE ON evaluation_case_results BEGIN
                    SELECT RAISE(ABORT, 'evaluation case history is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS evaluation_cases_no_replace
                BEFORE INSERT ON evaluation_case_results
                WHEN EXISTS (SELECT 1 FROM evaluation_case_results
                    WHERE id=NEW.id OR (run_id=NEW.run_id AND case_id=NEW.case_id))
                BEGIN SELECT RAISE(ABORT, 'evaluation case history cannot be replaced'); END;
                CREATE TRIGGER IF NOT EXISTS evaluation_cases_no_delete
                BEFORE DELETE ON evaluation_case_results BEGIN
                    SELECT RAISE(ABORT, 'evaluation case history is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS evaluation_cases_running_run_only
                BEFORE INSERT ON evaluation_case_results
                WHEN COALESCE(
                    (SELECT status FROM evaluation_runs WHERE id=NEW.run_id), ''
                ) != 'running'
                BEGIN SELECT RAISE(ABORT, 'cases require an active evaluation run'); END;
                CREATE TRIGGER IF NOT EXISTS evaluation_runs_no_delete
                BEFORE DELETE ON evaluation_runs BEGIN
                    SELECT RAISE(ABORT, 'evaluation run history is immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS evaluation_runs_no_replace
                BEFORE INSERT ON evaluation_runs
                WHEN EXISTS (SELECT 1 FROM evaluation_runs WHERE id=NEW.id)
                BEGIN SELECT RAISE(ABORT, 'evaluation run history cannot be replaced'); END;
                CREATE TRIGGER IF NOT EXISTS evaluation_runs_finalize_only
                BEFORE UPDATE ON evaluation_runs
                WHEN OLD.status != 'running' OR NEW.status = 'running'
                    OR NEW.id IS NOT OLD.id
                    OR NEW.dataset_path IS NOT OLD.dataset_path
                    OR NEW.dataset_id IS NOT OLD.dataset_id
                    OR NEW.label IS NOT OLD.label
                    OR NEW.source_session_id IS NOT OLD.source_session_id
                    OR NEW.model_name IS NOT OLD.model_name
                    OR NEW.agent_mode IS NOT OLD.agent_mode
                    OR NEW.started_at IS NOT OLD.started_at
                    OR NEW.dataset_sha256 IS NOT OLD.dataset_sha256
                    OR NEW.transcript_sha256 IS NOT OLD.transcript_sha256
                    OR NEW.summary_sha256 IS NOT OLD.summary_sha256
                    OR NEW.dataset_snapshot_json IS NOT OLD.dataset_snapshot_json
                    OR NEW.transcript_snapshot_json IS NOT OLD.transcript_snapshot_json
                    OR NEW.summary_snapshot_json IS NOT OLD.summary_snapshot_json
                    OR NEW.configuration_json IS NOT OLD.configuration_json
                BEGIN SELECT RAISE(ABORT, 'evaluation runs can only be finalized once'); END;
                """
            )
            connection.commit()

    def start_run(self, metadata: dict[str, Any]) -> int:
        columns = (
            "dataset_path", "dataset_id", "label", "source_session_id", "model_name",
            "agent_mode", "started_at", "dataset_sha256", "transcript_sha256", "summary_sha256",
            "dataset_snapshot_json", "transcript_snapshot_json", "summary_snapshot_json",
            "configuration_json",
        )
        with self.connect() as connection:
            cursor = connection.execute(
                f"INSERT INTO evaluation_runs ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(metadata[column] for column in columns),
            )
            run_id = int(cursor.lastrowid)
            connection.commit()
        self._active_runs.add(run_id)
        return run_id

    def append_case(self, run_id: int, result: dict[str, Any]) -> None:
        if run_id not in self._active_runs:
            raise ValueError("can only append cases to this evaluator's active run")
        columns = (
            "case_id", "question", "answer", "execution_success", "error_text", "started_at",
            "finished_at", "processing_seconds", "evidence_insufficient", "step_count",
        )
        json_columns = (
            "retrieved_segment_ids", "retrieved_ranges", "cited_segment_ids", "citation_ranges",
            "tool_calls", "case_snapshot", "metrics",
        )
        names = ["run_id", *columns, *(name + "_json" for name in json_columns)]
        values = [run_id, *(result[column] for column in columns)]
        values.extend(canonical_json(result[column]) for column in json_columns)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO evaluation_case_results ({', '.join(names)}) "
                f"VALUES ({', '.join('?' for _ in names)})",
                values,
            )
            connection.commit()

    def finish_run(
        self, run_id: int, *, status: str, finished_at: str, duration_seconds: float,
        metrics: dict[str, Any], error_text: str | None = None,
    ) -> None:
        if run_id not in self._active_runs:
            raise ValueError("can only finalize this evaluator's active run once")
        with self.connect() as connection:
            cursor = connection.execute(
                """UPDATE evaluation_runs
                   SET status=?, finished_at=?, duration_seconds=?, metrics_json=?, error_text=?
                   WHERE id=? AND status='running'""",
                (status, finished_at, duration_seconds,
                 canonical_json(metrics), error_text, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("evaluation run is missing or already finalized")
            connection.commit()
        self._active_runs.remove(run_id)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_case_results(self, run_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evaluation_case_results WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]
