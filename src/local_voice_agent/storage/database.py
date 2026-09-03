from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from local_voice_agent.models import (
    Session,
    SessionStatus,
    StoredTranscriptSegment,
    TranscriptionResult,
)
from local_voice_agent.summaries.models import (
    ContentMode,
    GeneratedCheckpoint,
    GeneratedSummaryBundle,
    StoredFinalSummary,
    StoredSummaryRun,
    StoredSummaryCheckpoint,
)


SCHEMA_VERSION = 7


class Database:
    """Small SQLite boundary for the first end-to-end slice."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_audio_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('processing', 'completed', 'failed')),
                    language TEXT NOT NULL,
                    transcription_model TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    duration_seconds REAL CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
                    detected_language TEXT,
                    detected_language_probability REAL CHECK (
                        detected_language_probability IS NULL OR
                        detected_language_probability BETWEEN 0 AND 1
                    ),
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    segment_index INTEGER NOT NULL CHECK (segment_index >= 0),
                    start_seconds REAL NOT NULL CHECK (start_seconds >= 0),
                    end_seconds REAL NOT NULL CHECK (end_seconds > start_seconds),
                    text TEXT NOT NULL CHECK (length(trim(text)) > 0),
                    UNIQUE (session_id, segment_index)
                );

                CREATE INDEX IF NOT EXISTS idx_transcript_segments_session_time
                    ON transcript_segments(session_id, start_seconds, end_seconds);

                CREATE TABLE IF NOT EXISTS summary_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    start_seconds REAL NOT NULL CHECK (start_seconds >= 0),
                    end_seconds REAL NOT NULL CHECK (end_seconds > start_seconds),
                    summary TEXT NOT NULL CHECK (length(trim(summary)) > 0),
                    notes_json TEXT NOT NULL DEFAULT '[]',
                    topics_json TEXT NOT NULL,
                    key_claims_json TEXT NOT NULL DEFAULT '[]',
                    key_concepts_json TEXT NOT NULL DEFAULT '[]',
                    uncertainties_json TEXT NOT NULL DEFAULT '[]',
                    terms_to_verify_json TEXT NOT NULL DEFAULT '[]',
                    decisions_json TEXT NOT NULL,
                    action_items_json TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    generation_seconds REAL CHECK (
                        generation_seconds IS NULL OR generation_seconds >= 0
                    ),
                    attempt_seconds_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (session_id, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS final_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL UNIQUE
                        REFERENCES sessions(id) ON DELETE CASCADE,
                    overall_summary TEXT NOT NULL CHECK (length(trim(overall_summary)) > 0),
                    important_notes_json TEXT NOT NULL DEFAULT '[]',
                    main_topics_json TEXT NOT NULL,
                    key_claims_json TEXT NOT NULL DEFAULT '[]',
                    key_concepts_json TEXT NOT NULL DEFAULT '[]',
                    uncertainties_json TEXT NOT NULL DEFAULT '[]',
                    terms_to_verify_json TEXT NOT NULL DEFAULT '[]',
                    decisions_json TEXT NOT NULL,
                    action_items_json TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    checkpoint_count INTEGER NOT NULL CHECK (checkpoint_count > 0),
                    raw_response TEXT NOT NULL,
                    processing_seconds REAL CHECK (
                        processing_seconds IS NULL OR processing_seconds >= 0
                    ),
                    final_generation_seconds REAL CHECK (
                        final_generation_seconds IS NULL OR final_generation_seconds >= 0
                    ),
                    final_attempt_seconds_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_summary_checkpoints_session_time
                    ON summary_checkpoints(session_id, start_seconds, end_seconds);

                CREATE TABLE IF NOT EXISTS summary_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    label TEXT,
                    model_name TEXT NOT NULL,
                    chunk_seconds REAL CHECK (chunk_seconds IS NULL OR chunk_seconds > 0),
                    content_mode TEXT NOT NULL DEFAULT 'auto'
                        CHECK (content_mode IN ('auto', 'informational', 'meeting')),
                    overall_summary TEXT NOT NULL CHECK (length(trim(overall_summary)) > 0),
                    important_notes_json TEXT NOT NULL DEFAULT '[]',
                    main_topics_json TEXT NOT NULL,
                    key_claims_json TEXT NOT NULL DEFAULT '[]',
                    key_concepts_json TEXT NOT NULL DEFAULT '[]',
                    uncertainties_json TEXT NOT NULL DEFAULT '[]',
                    terms_to_verify_json TEXT NOT NULL DEFAULT '[]',
                    decisions_json TEXT NOT NULL,
                    action_items_json TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    checkpoint_count INTEGER NOT NULL CHECK (checkpoint_count > 0),
                    raw_response TEXT NOT NULL,
                    processing_seconds REAL CHECK (
                        processing_seconds IS NULL OR processing_seconds >= 0
                    ),
                    final_generation_seconds REAL CHECK (
                        final_generation_seconds IS NULL OR final_generation_seconds >= 0
                    ),
                    final_attempt_seconds_json TEXT NOT NULL DEFAULT '[]',
                    legacy_final_summary_id INTEGER UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_summary_runs_session_created
                    ON summary_runs(session_id, id DESC);

                CREATE TABLE IF NOT EXISTS summary_run_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_run_id INTEGER NOT NULL
                        REFERENCES summary_runs(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    start_seconds REAL NOT NULL CHECK (start_seconds >= 0),
                    end_seconds REAL NOT NULL CHECK (end_seconds > start_seconds),
                    summary TEXT NOT NULL CHECK (length(trim(summary)) > 0),
                    notes_json TEXT NOT NULL DEFAULT '[]',
                    topics_json TEXT NOT NULL,
                    key_claims_json TEXT NOT NULL DEFAULT '[]',
                    key_concepts_json TEXT NOT NULL DEFAULT '[]',
                    uncertainties_json TEXT NOT NULL DEFAULT '[]',
                    terms_to_verify_json TEXT NOT NULL DEFAULT '[]',
                    decisions_json TEXT NOT NULL,
                    action_items_json TEXT NOT NULL,
                    open_questions_json TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    generation_seconds REAL CHECK (
                        generation_seconds IS NULL OR generation_seconds >= 0
                    ),
                    attempt_seconds_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (summary_run_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_run_checkpoints_time
                    ON summary_run_checkpoints(
                        summary_run_id, start_seconds, end_seconds
                    );
                """
            )
            self._add_json_column_if_missing(
                connection, "summary_checkpoints", "notes_json"
            )
            self._add_json_column_if_missing(
                connection, "summary_checkpoints", "key_claims_json"
            )
            self._add_json_column_if_missing(
                connection, "summary_checkpoints", "key_concepts_json"
            )
            self._add_json_column_if_missing(
                connection, "summary_checkpoints", "uncertainties_json"
            )
            self._add_json_column_if_missing(
                connection, "summary_checkpoints", "terms_to_verify_json"
            )
            self._add_json_column_if_missing(
                connection, "summary_checkpoints", "attempt_seconds_json"
            )
            self._add_json_column_if_missing(
                connection, "final_summaries", "important_notes_json"
            )
            self._add_json_column_if_missing(
                connection, "final_summaries", "key_claims_json"
            )
            self._add_json_column_if_missing(
                connection, "final_summaries", "key_concepts_json"
            )
            self._add_json_column_if_missing(
                connection, "final_summaries", "uncertainties_json"
            )
            self._add_json_column_if_missing(
                connection, "final_summaries", "terms_to_verify_json"
            )
            self._add_column_if_missing(
                connection,
                "final_summaries",
                "final_generation_seconds",
                "REAL CHECK (final_generation_seconds IS NULL OR "
                "final_generation_seconds >= 0)",
            )
            self._add_json_column_if_missing(
                connection, "final_summaries", "final_attempt_seconds_json"
            )
            self._add_column_if_missing(
                connection,
                "summary_runs",
                "content_mode",
                "TEXT NOT NULL DEFAULT 'auto' "
                "CHECK (content_mode IN ('auto', 'informational', 'meeting'))",
            )
            self._add_json_column_if_missing(
                connection, "summary_runs", "important_notes_json"
            )
            self._add_json_column_if_missing(
                connection, "summary_run_checkpoints", "notes_json"
            )
            self._migrate_legacy_summaries(connection)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()

    def create_session(self, source_audio_path: Path, language: str, model: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (
                    source_audio_path, status, language, transcription_model
                ) VALUES (?, ?, ?, ?)
                """,
                (str(source_audio_path), SessionStatus.PROCESSING.value, language, model),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def store_transcription(self, session_id: int, result: TranscriptionResult) -> None:
        """Persist all segments and mark the session completed in one transaction."""

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO transcript_segments (
                    session_id, segment_index, start_seconds, end_seconds, text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        segment.index,
                        segment.start_seconds,
                        segment.end_seconds,
                        segment.text,
                    )
                    for segment in result.segments
                ],
            )
            cursor = connection.execute(
                """
                UPDATE sessions
                SET status = ?, completed_at = CURRENT_TIMESTAMP,
                    duration_seconds = ?, detected_language = ?,
                    detected_language_probability = ?, error_message = NULL
                WHERE id = ? AND status = ?
                """,
                (
                    SessionStatus.COMPLETED.value,
                    result.duration_seconds,
                    result.detected_language,
                    result.detected_language_probability,
                    session_id,
                    SessionStatus.PROCESSING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"Processing session {session_id} was not found")
            connection.commit()

    def mark_session_failed(self, session_id: int, error_message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE id = ?
                """,
                (SessionStatus.FAILED.value, error_message[:2000], session_id),
            )
            connection.commit()

    def get_session(self, session_id: int) -> Session | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self) -> list[Session]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY id DESC"
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def get_transcript(self, session_id: int) -> list[StoredTranscriptSegment]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, segment_index, start_seconds, end_seconds, text
                FROM transcript_segments
                WHERE session_id = ?
                ORDER BY segment_index
                """,
                (session_id,),
            ).fetchall()

        return [
            StoredTranscriptSegment(
                id=row["id"],
                session_id=row["session_id"],
                index=row["segment_index"],
                start_seconds=row["start_seconds"],
                end_seconds=row["end_seconds"],
                text=row["text"],
            )
            for row in rows
        ]

    def store_summary_run(
        self,
        *,
        session_id: int,
        model_name: str,
        chunk_seconds: float,
        content_mode: ContentMode,
        label: str | None,
        bundle: GeneratedSummaryBundle,
    ) -> int:
        """Append one complete summary run after every LLM call succeeds."""

        with self.connect() as connection:
            session = connection.execute(
                "SELECT status FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise LookupError(f"Session {session_id} was not found")
            if session["status"] != SessionStatus.COMPLETED.value:
                raise ValueError(
                    f"Session {session_id} is not completed and cannot be summarized"
                )

            final = bundle.final_summary
            cursor = connection.execute(
                """
                INSERT INTO summary_runs (
                    session_id, label, model_name, chunk_seconds, content_mode,
                    overall_summary, important_notes_json,
                    main_topics_json, key_claims_json, key_concepts_json,
                    uncertainties_json, terms_to_verify_json, decisions_json,
                    action_items_json, open_questions_json, checkpoint_count,
                    raw_response, processing_seconds, final_generation_seconds,
                    final_attempt_seconds_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    self._normalize_label(label),
                    model_name,
                    chunk_seconds,
                    content_mode,
                    final.overall_summary,
                    self._json(final.important_notes),
                    self._json(final.main_topics),
                    self._json([claim.model_dump() for claim in final.key_claims]),
                    self._json([concept.model_dump() for concept in final.key_concepts]),
                    self._json(
                        [item.model_dump() for item in final.uncertainties_and_debates]
                    ),
                    self._json([term.model_dump() for term in final.terms_to_verify]),
                    self._json(final.decisions),
                    self._json([action.model_dump() for action in final.action_items]),
                    self._json(final.open_questions),
                    len(bundle.checkpoints),
                    bundle.final_raw_response,
                    bundle.processing_seconds,
                    bundle.final_generation_seconds,
                    self._json(bundle.final_attempt_seconds),
                ),
            )
            summary_run_id = int(cursor.lastrowid)

            connection.executemany(
                """
                INSERT INTO summary_run_checkpoints (
                    summary_run_id, chunk_index, start_seconds, end_seconds, summary,
                    notes_json, topics_json, key_claims_json, key_concepts_json,
                    uncertainties_json, terms_to_verify_json, decisions_json,
                    action_items_json, open_questions_json, raw_response,
                    generation_seconds, attempt_seconds_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    self._checkpoint_values(summary_run_id, item)
                    for item in bundle.checkpoints
                ],
            )
            connection.commit()
            return summary_run_id

    def get_summary_checkpoints(
        self, session_id: int, summary_run_id: int | None = None
    ) -> list[StoredSummaryCheckpoint]:
        with self.connect() as connection:
            run_id = self._resolve_summary_run_id(
                connection, session_id=session_id, summary_run_id=summary_run_id
            )
            if run_id is None:
                return []
            rows = connection.execute(
                """
                SELECT checkpoint.*, run.session_id, run.model_name
                FROM summary_run_checkpoints AS checkpoint
                JOIN summary_runs AS run ON run.id = checkpoint.summary_run_id
                WHERE checkpoint.summary_run_id = ?
                ORDER BY checkpoint.chunk_index
                """,
                (run_id,),
            ).fetchall()

        return [
            StoredSummaryCheckpoint(
                id=row["id"],
                summary_run_id=row["summary_run_id"],
                session_id=row["session_id"],
                chunk_index=row["chunk_index"],
                start_seconds=row["start_seconds"],
                end_seconds=row["end_seconds"],
                summary=row["summary"],
                notes=json.loads(row["notes_json"]),
                topics=json.loads(row["topics_json"]),
                key_claims=json.loads(row["key_claims_json"]),
                key_concepts=json.loads(row["key_concepts_json"]),
                uncertainties_and_debates=json.loads(row["uncertainties_json"]),
                terms_to_verify=json.loads(row["terms_to_verify_json"]),
                decisions=json.loads(row["decisions_json"]),
                action_items=json.loads(row["action_items_json"]),
                open_questions=json.loads(row["open_questions_json"]),
                model_name=row["model_name"],
                generation_seconds=row["generation_seconds"],
                attempt_seconds=tuple(json.loads(row["attempt_seconds_json"])),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_final_summary(
        self, session_id: int, summary_run_id: int | None = None
    ) -> StoredFinalSummary | None:
        with self.connect() as connection:
            if summary_run_id is None:
                row = connection.execute(
                    """
                    SELECT * FROM summary_runs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM summary_runs
                    WHERE session_id = ? AND id = ?
                    """,
                    (session_id, summary_run_id),
                ).fetchone()
        if row is None:
            return None

        return StoredFinalSummary(
            id=row["id"],
            session_id=row["session_id"],
            label=row["label"],
            overall_summary=row["overall_summary"],
            important_notes=json.loads(row["important_notes_json"]),
            main_topics=json.loads(row["main_topics_json"]),
            key_claims=json.loads(row["key_claims_json"]),
            key_concepts=json.loads(row["key_concepts_json"]),
            uncertainties_and_debates=json.loads(row["uncertainties_json"]),
            terms_to_verify=json.loads(row["terms_to_verify_json"]),
            decisions=json.loads(row["decisions_json"]),
            action_items=json.loads(row["action_items_json"]),
            open_questions=json.loads(row["open_questions_json"]),
            model_name=row["model_name"],
            chunk_seconds=row["chunk_seconds"],
            content_mode=row["content_mode"],
            checkpoint_count=row["checkpoint_count"],
            processing_seconds=row["processing_seconds"],
            final_generation_seconds=row["final_generation_seconds"],
            final_attempt_seconds=tuple(json.loads(row["final_attempt_seconds_json"])),
            created_at=row["created_at"],
        )

    def list_summary_runs(self, session_id: int) -> list[StoredSummaryRun]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, label, model_name, chunk_seconds, content_mode,
                       checkpoint_count, processing_seconds,
                       final_generation_seconds, created_at
                FROM summary_runs
                WHERE session_id = ?
                ORDER BY id DESC
                """,
                (session_id,),
            ).fetchall()
        return [StoredSummaryRun.model_validate(dict(row)) for row in rows]

    @classmethod
    def _checkpoint_values(
        cls, summary_run_id: int, item: GeneratedCheckpoint
    ) -> tuple[object, ...]:
        checkpoint = item.checkpoint
        return (
            summary_run_id,
            item.chunk_index,
            item.start_seconds,
            item.end_seconds,
            checkpoint.summary,
            cls._json(checkpoint.notes),
            cls._json(checkpoint.topics),
            cls._json([claim.model_dump() for claim in checkpoint.key_claims]),
            cls._json([concept.model_dump() for concept in checkpoint.key_concepts]),
            cls._json(
                [entry.model_dump() for entry in checkpoint.uncertainties_and_debates]
            ),
            cls._json([term.model_dump() for term in checkpoint.terms_to_verify]),
            cls._json(checkpoint.decisions),
            cls._json([action.model_dump() for action in checkpoint.action_items]),
            cls._json(checkpoint.open_questions),
            item.raw_response,
            item.generation_seconds,
            cls._json(item.attempt_seconds),
        )

    @staticmethod
    def _resolve_summary_run_id(
        connection: sqlite3.Connection,
        *,
        session_id: int,
        summary_run_id: int | None,
    ) -> int | None:
        if summary_run_id is not None:
            row = connection.execute(
                "SELECT id FROM summary_runs WHERE id = ? AND session_id = ?",
                (summary_run_id, session_id),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT id FROM summary_runs
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    @staticmethod
    def _normalize_label(label: str | None) -> str | None:
        if label is None:
            return None
        normalized = label.strip()
        return normalized[:200] or None

    @staticmethod
    def _migrate_legacy_summaries(connection: sqlite3.Connection) -> None:
        legacy_summaries = connection.execute(
            """
            SELECT legacy.*
            FROM final_summaries AS legacy
            LEFT JOIN summary_runs AS run
                ON run.legacy_final_summary_id = legacy.id
            WHERE run.id IS NULL
            ORDER BY legacy.id
            """
        ).fetchall()

        for final in legacy_summaries:
            cursor = connection.execute(
                """
                INSERT INTO summary_runs (
                    session_id, label, model_name, chunk_seconds, content_mode,
                    overall_summary, important_notes_json,
                    main_topics_json, key_claims_json, key_concepts_json,
                    uncertainties_json, terms_to_verify_json, decisions_json,
                    action_items_json, open_questions_json, checkpoint_count,
                    raw_response, processing_seconds, final_generation_seconds,
                    final_attempt_seconds_json, legacy_final_summary_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final["session_id"],
                    "migrated-legacy-summary",
                    final["model_name"],
                    None,
                    "auto",
                    final["overall_summary"],
                    final["important_notes_json"],
                    final["main_topics_json"],
                    final["key_claims_json"],
                    final["key_concepts_json"],
                    final["uncertainties_json"],
                    final["terms_to_verify_json"],
                    final["decisions_json"],
                    final["action_items_json"],
                    final["open_questions_json"],
                    final["checkpoint_count"],
                    final["raw_response"],
                    final["processing_seconds"],
                    final["final_generation_seconds"],
                    final["final_attempt_seconds_json"],
                    final["id"],
                    final["created_at"],
                ),
            )
            summary_run_id = int(cursor.lastrowid)
            legacy_checkpoints = connection.execute(
                """
                SELECT * FROM summary_checkpoints
                WHERE session_id = ?
                ORDER BY chunk_index
                """,
                (final["session_id"],),
            ).fetchall()
            connection.executemany(
                """
                INSERT INTO summary_run_checkpoints (
                    summary_run_id, chunk_index, start_seconds, end_seconds, summary,
                    notes_json, topics_json, key_claims_json, key_concepts_json,
                    uncertainties_json, terms_to_verify_json, decisions_json,
                    action_items_json, open_questions_json, raw_response,
                    generation_seconds, attempt_seconds_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        summary_run_id,
                        checkpoint["chunk_index"],
                        checkpoint["start_seconds"],
                        checkpoint["end_seconds"],
                        checkpoint["summary"],
                        checkpoint["notes_json"],
                        checkpoint["topics_json"],
                        checkpoint["key_claims_json"],
                        checkpoint["key_concepts_json"],
                        checkpoint["uncertainties_json"],
                        checkpoint["terms_to_verify_json"],
                        checkpoint["decisions_json"],
                        checkpoint["action_items_json"],
                        checkpoint["open_questions_json"],
                        checkpoint["raw_response"],
                        checkpoint["generation_seconds"],
                        checkpoint["attempt_seconds_json"],
                        checkpoint["created_at"],
                    )
                    for checkpoint in legacy_checkpoints
                ],
            )

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _add_json_column_if_missing(
        connection: sqlite3.Connection, table: str, column: str
    ) -> None:
        Database._add_column_if_missing(
            connection, table, column, "TEXT NOT NULL DEFAULT '[]'"
        )

    @staticmethod
    def _add_column_if_missing(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        existing_columns = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            source_audio_path=Path(row["source_audio_path"]),
            status=row["status"],
            language=row["language"],
            transcription_model=row["transcription_model"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            duration_seconds=row["duration_seconds"],
            detected_language=row["detected_language"],
            detected_language_probability=row["detected_language_probability"],
            error_message=row["error_message"],
        )
