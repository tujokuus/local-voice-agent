"""Agent selection without importing the optional framework for baseline runs."""

from local_voice_agent.agent import ForcedFinalAnswerAgent, SessionQuestionAgent

AGENT_MODES = ("final", "manual", "pydanticai")


def agent_class_for(mode: str) -> type[SessionQuestionAgent]:
    if mode == "final":
        return ForcedFinalAnswerAgent
    if mode == "manual":
        return SessionQuestionAgent
    if mode == "pydanticai":
        try:
            from local_voice_agent.pydantic_agent import PydanticQuestionAgent
        except ImportError as exc:
            raise RuntimeError(
                'PydanticAI dependencies are unavailable. Install: '
                '.\\.venv\\Scripts\\python.exe -m pip install -e ".[pydanticai]"'
            ) from exc
        return PydanticQuestionAgent
    raise ValueError(f"agent_mode must be one of {AGENT_MODES}")
