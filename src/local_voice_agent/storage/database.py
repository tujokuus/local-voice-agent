from __future__ import annotations

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


SCHEMA_VERSION = 1


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
                """
            )
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

