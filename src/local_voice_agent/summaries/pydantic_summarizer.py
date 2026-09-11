"""PydanticAI structured generation within the shared chronological notes pipeline."""

from __future__ import annotations

import asyncio
from time import perf_counter

from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelRetry, NativeOutput
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider as FrameworkOllamaProvider
from pydantic_ai.usage import UsageLimits
from pydantic_graph import End

from local_voice_agent.llm import OllamaProvider, StructuredResponse
from local_voice_agent.summaries.summarizer import SessionSummarizer


class PydanticSessionSummarizer(SessionSummarizer):
    """Keep chunking, prompts, validation and storage identical to the custom workflow."""

    def __init__(self, *, model: Model | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = model

    def _structured_with_one_retry(self, *, messages, response_model, validator=None):
        async def generate(model):
            agent = Agent(
                model, output_type=NativeOutput(response_model), retries=1,
                system_prompt="\n".join(m.content for m in messages if m.role == "system"),
            )

            @agent.output_validator
            def validate(data):
                if validator is not None:
                    try:
                        validator(data)
                    except ValueError as exc:
                        raise ModelRetry(str(exc)) from exc
                return data

            attempts = []
            async with agent.iter(
                "\n\n".join(m.content for m in messages if m.role == "user"),
                usage_limits=UsageLimits(request_limit=2),
            ) as run:
                node = run.next_node
                while not isinstance(node, End):
                    is_request = Agent.is_model_request_node(node)
                    started = perf_counter()
                    try:
                        node = await run.next(node)
                    finally:
                        if is_request:
                            attempts.append(perf_counter() - started)
                result = run.result
            # Preserve actual accepted model text, not a re-serialized approximation.
            response = next(
                item for item in reversed(result.all_messages()) if isinstance(item, ModelResponse)
            )
            raw = "".join(part.content for part in response.parts if isinstance(part, TextPart))
            return StructuredResponse(data=result.output, raw_content=raw), tuple(attempts)

        async def with_client():
            if self._model is not None:
                return await generate(self._model)
            if not isinstance(self.provider, OllamaProvider):
                raise ValueError("PydanticAI notes require an OllamaProvider or injected test model")
            async with AsyncOpenAI(
                base_url=f"{self.provider.base_url}/v1", api_key="local-ollama",
                timeout=self.provider.timeout_seconds, max_retries=0,
            ) as client:
                return await generate(OllamaModel(
                    self.provider.model_name,
                    provider=FrameworkOllamaProvider(openai_client=client),
                    settings={"temperature": 0, "openai_reasoning_effort": "none"},
                ))

        try:
            return asyncio.run(with_client())
        except Exception as exc:
            raise RuntimeError(f"PydanticAI note generation failed: {exc}") from exc
