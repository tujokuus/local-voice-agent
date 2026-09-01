from __future__ import annotations

from pathlib import Path

from local_voice_agent.storage import Database
from local_voice_agent.transcription import Transcriber


class SessionProcessor:
    """Coordinates the first audio-to-database use case."""

    def __init__(
        self,
        *,
        database: Database,
        transcriber: Transcriber,
        language: str,
        transcription_model: str,
    ) -> None:
        self.database = database
        self.transcriber = transcriber
        self.language = language
        self.transcription_model = transcription_model

    def process(self, audio_path: Path) -> int:
        resolved_path = audio_path.expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {resolved_path}")

        self.database.initialize()
        session_id = self.database.create_session(
            resolved_path,
            language=self.language,
            model=self.transcription_model,
        )

        try:
            result = self.transcriber.transcribe(resolved_path)
            self.database.store_transcription(session_id, result)
        except Exception as exc:
            self.database.mark_session_failed(session_id, str(exc))
            raise

        return session_id

