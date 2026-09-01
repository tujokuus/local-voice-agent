from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ValidationError

from local_voice_agent.llm.base import (
    ChatMessage,
    StructuredModel,
    StructuredOutputError,
    StructuredResponse,
)


class OllamaProvider:
    """Minimal Ollama chat client built on the documented local HTTP API.

    Keeping this adapter small makes the provider boundary and structured-output
    validation visible without adding an SDK or agent framework.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600,
    ) -> None:
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model_name

    def structured_chat(
        self,
        *,
        messages: list[ChatMessage],
        response_model: type[StructuredModel],
    ) -> StructuredResponse[StructuredModel]:
        schema = response_model.model_json_schema()
        payload = {
            "model": self.model_name,
            "messages": [message.model_dump() for message in messages],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0},
        }
        response_payload = self._post_json("/api/chat", payload)

        try:
            raw_content = response_payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise StructuredOutputError(
                "Ollama response did not contain message.content"
            ) from exc

        if not isinstance(raw_content, str) or not raw_content.strip():
            raise StructuredOutputError("Ollama returned empty structured content")

        try:
            parsed = response_model.model_validate_json(raw_content)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"Ollama returned invalid {response_model.__name__}: {exc}",
                raw_content=raw_content,
            ) from exc

        duration_ns = response_payload.get("total_duration")
        duration_seconds = (
            float(duration_ns) / 1_000_000_000
            if isinstance(duration_ns, int | float)
            else None
        )
        return StructuredResponse(
            data=parsed,
            raw_content=raw_content,
            total_duration_seconds=duration_seconds,
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request exceeded {self.timeout_seconds:g} seconds"
            ) from exc

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Ollama returned an unexpected JSON response")
        return decoded

