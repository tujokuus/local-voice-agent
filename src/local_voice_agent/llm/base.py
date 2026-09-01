from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StructuredResponse(Generic[StructuredModel]):
    data: StructuredModel
    raw_content: str
    total_duration_seconds: float | None = None


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, *, raw_content: str | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class LLMProvider(Protocol):
    """Provider boundary used by summarization and, later, the agent loop."""

    @property
    def model_name(self) -> str: ...

    def structured_chat(
        self,
        *,
        messages: list[ChatMessage],
        response_model: type[StructuredModel],
    ) -> StructuredResponse[StructuredModel]: ...

