from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with environment-variable overrides.

    English is the only supported MVP language, but keeping it in configuration
    avoids spreading a language literal through the application.
    """

    app_language: str = "en"
    transcription_language: str = "en"
    database_path: Path = Path("data/local_voice_agent.db")
    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_timeout_seconds: float = 600
    checkpoint_target_seconds: float = 600
    minimum_final_chunk_seconds: float = 120

    @classmethod
    def from_environment(cls) -> Settings:
        beam_size_text = _env("LVA_WHISPER_BEAM_SIZE", "5")
        try:
            beam_size = int(beam_size_text)
        except ValueError as exc:
            raise ValueError("LVA_WHISPER_BEAM_SIZE must be an integer") from exc

        if beam_size < 1:
            raise ValueError("LVA_WHISPER_BEAM_SIZE must be at least 1")

        def positive_float(name: str, default: str) -> float:
            raw_value = _env(name, default)
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number") from exc
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
            return value

        return cls(
            app_language=_env("LVA_APP_LANGUAGE", "en"),
            transcription_language=_env("LVA_TRANSCRIPTION_LANGUAGE", "en"),
            database_path=Path(_env("LVA_DATABASE_PATH", "data/local_voice_agent.db")),
            whisper_model=_env("LVA_WHISPER_MODEL", "small.en"),
            whisper_device=_env("LVA_WHISPER_DEVICE", "cpu"),
            whisper_compute_type=_env("LVA_WHISPER_COMPUTE_TYPE", "int8"),
            whisper_beam_size=beam_size,
            ollama_base_url=_env("LVA_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=_env("LVA_OLLAMA_MODEL", "qwen3.5:4b"),
            ollama_timeout_seconds=positive_float("LVA_OLLAMA_TIMEOUT_SECONDS", "600"),
            checkpoint_target_seconds=positive_float(
                "LVA_CHECKPOINT_TARGET_SECONDS", "600"
            ),
            minimum_final_chunk_seconds=positive_float(
                "LVA_MINIMUM_FINAL_CHUNK_SECONDS", "120"
            ),
        )
