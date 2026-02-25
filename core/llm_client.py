"""Multi-provider LLM adapter normalizing tool-calling across Anthropic, OpenAI, and Gemini."""

from __future__ import annotations
import os
import json
from abc import ABC, abstractmethod
from typing import Any

from core.schemas import LLMResponse, ToolCall


# ── Tool schema translators ───────────────────────────────────────────────────

def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Translate Anthropic tool schema to OpenAI function-calling format."""
    oai_tools = []
    for t in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return oai_tools


def _anthropic_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """Translate Anthropic tool schema to Google Gemini function declarations."""
    declarations = []
    for t in tools:
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        # Gemini uses "parameters" with JSON Schema
        declarations.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": schema,
        })
    return declarations


# ── Base class ────────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send messages and return a normalized LLMResponse."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        ...

    def build_assistant_message(self, response: "LLMResponse") -> dict:
        """Build an assistant message dict for message history (provider-specific format)."""
        return {"role": "assistant", "content": response.raw_content}

    def build_tool_result_messages(self, tool_results: list[dict]) -> list[dict]:
        """Build tool result message(s) for message history.

        tool_results: list of {"tool_use_id": str, "name": str, "content": str}
        Returns a list of message dicts (Anthropic uses one user message; OpenAI uses one per result).
        """
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]}
            for tr in tool_results
        ]}]


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    @property
    def provider(self) -> str:
        return "anthropic"

    def chat(self, messages, tools, system, max_tokens=4096) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    id=block.id,
                    input=block.input,
                ))

        stop_reason = "end_turn" if response.stop_reason in ("end_turn", "stop_sequence") else "tool_use"
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_content=response.content,
        )


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-5.2"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @property
    def provider(self) -> str:
        return "openai"

    # Models that require max_completion_tokens instead of max_tokens
    _MAX_COMPLETION_TOKENS_MODELS = ("o1", "o3", "gpt-5")

    def build_assistant_message(self, response: "LLMResponse") -> dict:
        """Store assistant turn in OpenAI-native format (tool_calls field)."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.text or None,
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    def build_tool_result_messages(self, tool_results: list[dict]) -> list[dict]:
        """Each tool result is a separate role='tool' message for OpenAI."""
        return [
            {
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": str(tr["content"]),
            }
            for tr in tool_results
        ]

    def chat(self, messages, tools, system, max_tokens=4096) -> LLMResponse:
        # Messages are already in OpenAI format (built by build_assistant_message /
        # build_tool_result_messages), just prepend the system message.
        oai_messages = [{"role": "system", "content": system}] + messages

        # Newer OpenAI models (o-series, GPT-5.x) require max_completion_tokens
        token_param = (
            "max_completion_tokens"
            if any(self.model.startswith(p) for p in self._MAX_COMPLETION_TOKENS_MODELS)
            else "max_tokens"
        )

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=oai_messages,
        )
        kwargs[token_param] = max_tokens
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        text = msg.content or ""
        tool_calls: list[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    id=tc.id,
                    input=json.loads(tc.function.arguments),
                ))

        finish_reason = response.choices[0].finish_reason
        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_content=None,  # not used; history built via build_assistant_message()
        )


# ── Google Gemini (uses google-genai SDK) ────────────────────────────────────

class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = "gemini-3.1-pro"):
        from google import genai
        from google.genai import types as genai_types
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._types = genai_types
        self.model_name = model

    @property
    def provider(self) -> str:
        return "gemini"

    def chat(self, messages, tools, system, max_tokens=4096) -> LLMResponse:
        from google.genai import types as t

        # Convert messages to Gemini Content format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            content = m["content"]

            if isinstance(content, str):
                contents.append(t.Content(role=role, parts=[t.Part.from_text(content)]))
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "tool_result":
                        parts.append(t.Part.from_function_response(
                            name=item.get("tool_use_id", "unknown"),
                            response={"result": item.get("content", "")},
                        ))
                    elif item.get("type") == "text":
                        parts.append(t.Part.from_text(item.get("text", "")))
                    elif item.get("type") == "tool_use":
                        parts.append(t.Part.from_function_call(
                            name=item.get("name", ""),
                            args=item.get("input", {}),
                        ))
                if parts:
                    contents.append(t.Content(role=role, parts=parts))

        # Build tool declarations
        gemini_tools = None
        if tools:
            declarations = _anthropic_tools_to_gemini(tools)
            gemini_tools = [t.Tool(function_declarations=declarations)]

        config = t.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        text = ""
        tool_calls: list[ToolCall] = []
        raw_content: list[dict] = []

        for part in response.candidates[0].content.parts:
            if part.text:
                text += part.text
                raw_content.append({"type": "text", "text": part.text})
            elif part.function_call:
                fc = part.function_call
                tc_id = f"gemini_{fc.name}_{len(tool_calls)}"
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCall(name=fc.name, id=tc_id, input=args))
                raw_content.append({
                    "type": "tool_use",
                    "id": tc_id,
                    "name": fc.name,
                    "input": args,
                })

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_content=raw_content,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

PROVIDER_MODELS = {
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-6"],
    "openai": ["gpt-5.2", "gpt-5.3"],
    "gemini": ["gemini-3.1-pro", "gemini-flash-3.0"],
}


def create_llm_client(provider: str, model: str) -> BaseLLMClient:
    """Factory: create the appropriate LLM client."""
    if provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider == "openai":
        return OpenAIClient(model=model)
    elif provider == "gemini":
        return GeminiClient(model=model)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: anthropic, openai, gemini")
