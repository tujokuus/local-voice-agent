from __future__ import annotations

from pathlib import Path
from typing import Protocol

from local_voice_agent.models import TranscriptionResult


class Transcriber(Protocol):
    """Provider boundary for local speech recognition."""

    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...

