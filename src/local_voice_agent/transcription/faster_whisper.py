from __future__ import annotations

from pathlib import Path
from typing import Any

from local_voice_agent.models import TranscriptSegment, TranscriptionResult


class FasterWhisperTranscriber:
    """English transcription backed by faster-whisper.

    The third-party import is intentionally lazy so database and CLI inspection
    commands remain usable even when transcription dependencies are not installed.
    """

    def __init__(
        self,
        *,
        model_name: str,
        language: str,
        device: str,
        compute_type: str,
        beam_size: int,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self._model: Any | None = None

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._get_model()
        segment_stream, info = model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
        )

        segments: list[TranscriptSegment] = []
        for source_segment in segment_stream:
            text = source_segment.text.strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    index=len(segments),
                    start_seconds=float(source_segment.start),
                    end_seconds=float(source_segment.end),
                    text=text,
                )
            )

        return TranscriptionResult(
            segments=tuple(segments),
            duration_seconds=self._optional_float(info, "duration"),
            detected_language=getattr(info, "language", None),
            detected_language_probability=self._optional_float(
                info, "language_probability"
            ),
        )

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install the project with "
                    "'python -m pip install -e .' first."
                ) from exc

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    @staticmethod
    def _optional_float(value: object, attribute: str) -> float | None:
        raw_value = getattr(value, attribute, None)
        return float(raw_value) if raw_value is not None else None

