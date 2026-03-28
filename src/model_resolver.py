"""Helpers to resolve ADK and OpenAI clients for direct vs Gateway mode."""

from __future__ import annotations

import json
import os
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from openai import OpenAI

from src.config import AgentConfig


def _is_gateway_enabled(config: AgentConfig | None = None) -> bool:
    if config is not None:
        return bool(config.use_gateway)
    raw = os.environ.get("BRAINTRUST_USE_GATEWAY", "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gateway_url(config: AgentConfig | None = None) -> str:
    if config and config.gateway_url:
        return config.gateway_url
    return os.environ.get("BRAINTRUST_GATEWAY_URL", "https://gateway.braintrust.dev/v1")


def _gateway_api_key(config: AgentConfig | None = None) -> str | None:
    if config and config.gateway_api_key:
        return config.gateway_api_key
    return os.environ.get("BRAINTRUST_GATEWAY_API_KEY") or os.environ.get(
        "BRAINTRUST_API_KEY"
    )


def _gateway_logging_headers() -> dict[str, str]:
    """Build explicit Braintrust attribution headers for gateway requests."""
    project_id = os.environ.get("BRAINTRUST_PROJECT_ID", "").strip()
    project_name = os.environ.get("BRAINTRUST_PROJECT", "").strip()

    headers: dict[str, str] = {}
    if project_id:
        headers["x-bt-parent"] = f"project_id:{project_id}"
        headers["x-bt-project-id"] = project_id
    elif project_name:
        headers["x-bt-parent"] = f"project_name:{project_name}"
        headers["x-bt-project-name"] = project_name
    return headers


_OPENAI_BARE_MODEL_PREFIXES = (
    "gpt-",
    "o1",
    "o3",
    "o4",
    "chatgpt-",
    "text-embedding-",
    "omni-moderation-",
    "whisper-",
    "tts-",
    "dall-e-",
)


def _is_openai_model_name(model_name: str | None) -> bool:
    """Return True when model should be routed via OpenAI-compatible transport."""
    if not isinstance(model_name, str):
        return False

    normalized = model_name.strip().lower()
    if not normalized:
        return False

    if normalized.startswith("openai/"):
        return True

    # Provider-qualified models should be handled by their explicit provider.
    if "/" in normalized:
        return False

    return normalized.startswith(_OPENAI_BARE_MODEL_PREFIXES)


def _gateway_auth_headers(model_name: str | None = None) -> dict[str, str]:
    """Build auth headers for Braintrust Gateway, with optional provider key passthrough."""
    headers: dict[str, str] = {}

    gateway_key = _gateway_api_key()
    if gateway_key:
        headers["Authorization"] = f"Bearer {gateway_key}"

    if _is_openai_model_name(model_name):
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            # Provider key passthrough for gateways configured to use caller-supplied OpenAI creds.
            headers["x-bt-openai-api-key"] = openai_key
    return headers


class GatewayGemini(Gemini):
    """Gemini model wrapper that injects Braintrust gateway project headers."""

    def _tracking_headers(self) -> dict[str, str]:  # type: ignore[override]
        headers = super()._tracking_headers()
        headers.update(_gateway_logging_headers())
        headers.update(_gateway_auth_headers(getattr(self, "model", None)))
        return headers


class GatewayOpenAI(BaseLlm):
    """OpenAI-compatible ADK model that routes requests through Braintrust Gateway."""

    @staticmethod
    def _normalize_openai_schema(value: Any) -> Any:
        """Normalize ADK/Google schema objects into OpenAI-compatible JSON schema."""
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        if isinstance(value, list):
            return [GatewayOpenAI._normalize_openai_schema(v) for v in value]
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for k, v in value.items():
                nv = GatewayOpenAI._normalize_openai_schema(v)
                if k == "type" and isinstance(nv, str):
                    nv = nv.lower()
                normalized[k] = nv
            return normalized
        return value

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ):
        if stream:
            raise NotImplementedError("GatewayOpenAI streaming is not implemented.")

        client = make_wrapped_openai_client()
        messages: list[dict[str, Any]] = []

        system_instruction = llm_request.config.system_instruction
        if isinstance(system_instruction, str) and system_instruction.strip():
            messages.append({"role": "system", "content": system_instruction})

        for content in llm_request.contents:
            role = (content.role or "user").lower()
            role = "assistant" if role == "model" else role

            text_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for part in content.parts or []:
                if part.text:
                    text_chunks.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        {
                            "id": fc.id or f"call_{fc.name}",
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(fc.args or {}),
                            },
                        }
                    )
                if part.function_response:
                    fr = part.function_response
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": fr.id or f"call_{fr.name}",
                            "content": json.dumps(fr.response or {}),
                        }
                    )

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "\n".join(text_chunks) if text_chunks else None,
                        "tool_calls": tool_calls,
                    }
                )
            elif text_chunks:
                messages.append({"role": role, "content": "\n".join(text_chunks)})

        tools: list[dict[str, Any]] = []
        for tool in llm_request.config.tools or []:
            declarations = getattr(tool, "function_declarations", None) or []
            for decl in declarations:
                parameters = getattr(decl, "parameters_json_schema", None) or getattr(
                    decl, "parameters", None
                )
                normalized_parameters = self._normalize_openai_schema(parameters)
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": decl.name,
                            "description": decl.description or "",
                            "parameters": normalized_parameters
                            or {"type": "object", "properties": {}},
                        },
                    }
                )

        extra_headers: dict[str, str] = {}
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            extra_headers["x-bt-openai-api-key"] = openai_key

        model_for_gateway = (
            self.model.split("/", 1)[1]
            if isinstance(self.model, str) and self.model.startswith("openai/")
            else self.model
        )
        request_kwargs: dict[str, Any] = {
            "model": model_for_gateway,
            "messages": messages,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        if extra_headers:
            request_kwargs["extra_headers"] = extra_headers

        completion = client.chat.completions.create(**request_kwargs)
        choice = completion.choices[0].message

        parts: list[types.Part] = []
        if choice.content:
            parts.append(types.Part(text=choice.content))
        for call in choice.tool_calls or []:
            args: dict[str, Any] = {}
            try:
                args = json.loads(call.function.arguments or "{}")
            except Exception:
                args = {}
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.id,
                        name=call.function.name,
                        args=args,
                    )
                )
            )

        usage = completion.usage
        usage_metadata = None
        if usage is not None:
            usage_metadata = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=getattr(usage, "prompt_tokens", None),
                candidates_token_count=getattr(usage, "completion_tokens", None),
                total_token_count=getattr(usage, "total_tokens", None),
            )

        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=usage_metadata,
        )


def resolve_adk_model(model_name: str, config: AgentConfig | None = None) -> Any:
    """Return either a raw model string or Gateway-routed Gemini model object."""
    if not _is_gateway_enabled(config):
        return model_name

    api_key = _gateway_api_key(config)
    if not api_key:
        raise RuntimeError(
            "BRAINTRUST_USE_GATEWAY is enabled, but no gateway key was found. "
            "Set BRAINTRUST_GATEWAY_API_KEY or BRAINTRUST_API_KEY."
        )

    # ADK Gemini client reads GOOGLE_API_KEY from environment.
    os.environ["GOOGLE_API_KEY"] = api_key
    if _is_openai_model_name(model_name):
        return GatewayOpenAI(model=model_name)
    return GatewayGemini(model=model_name, base_url=_gateway_url(config))


def make_wrapped_openai_client() -> OpenAI:
    """Build OpenAI client routed through Braintrust Gateway when enabled."""
    if not _is_gateway_enabled():
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    gateway_key = _gateway_api_key()
    if not gateway_key:
        raise RuntimeError(
            "BRAINTRUST_USE_GATEWAY is enabled, but no gateway key was found. "
            "Set BRAINTRUST_GATEWAY_API_KEY or BRAINTRUST_API_KEY."
        )

    return OpenAI(
        api_key=gateway_key,
        base_url=_gateway_url(),
        default_headers=_gateway_logging_headers(),
    )
