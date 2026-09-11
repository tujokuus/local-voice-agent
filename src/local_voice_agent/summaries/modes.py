"""Lazy selection keeps the framework dependency optional for custom notes."""

from local_voice_agent.summaries.summarizer import SessionSummarizer


def summarizer_class_for(mode: str) -> type[SessionSummarizer]:
    if mode == "custom":
        return SessionSummarizer
    if mode == "pydanticai":
        try:
            from local_voice_agent.summaries.pydantic_summarizer import PydanticSessionSummarizer
        except ImportError as exc:
            raise RuntimeError(
                'Install optional dependencies: '
                '.\\.venv\\Scripts\\python.exe -m pip install -e ".[pydanticai]"'
            ) from exc
        return PydanticSessionSummarizer
    raise ValueError("summary_mode must be custom or pydanticai")
