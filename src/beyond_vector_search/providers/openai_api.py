"""openai_api.py — the same agent on the OpenAI Responses API.

Only the history shape changes. Tool functions stay the ones in tools.py.
The whole response output is appended each turn so reasoning items stay
in the conversation. OpenAI caches repeated prefixes on its own.
"""

from __future__ import annotations

import json

from beyond_vector_search.providers.base import ModelTurn, TextReply, ToolCall, ToolOutput
from beyond_vector_search.providers.claude import MissingExtraError
from beyond_vector_search.tools import Tool


class OpenAIProvider:
    def __init__(self, model: str, max_tokens: int = 4096):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingExtraError(
                "Install the OpenAI extra: pip install 'beyond-vector-search[openai]'"
            ) from exc
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.items: list = []
        self.instructions = ""
        self.tools: list = []

    def start(self, question: str, tools: list[Tool], system: str) -> None:
        self.tools = [tool.openai_schema() for tool in tools]
        self.instructions = system
        self.items = [{"role": "user", "content": question}]

    def complete(self) -> ModelTurn:
        response = self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            tools=self.tools,
            input=self.items,
        )
        self.items.extend(list(response.output))
        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            calls.append(ToolCall(
                id=item.call_id,
                name=item.name,
                arguments=_parse_args(getattr(item, "arguments", None)),
            ))
        usage = response.usage
        cached = _cached_tokens(usage)
        total_in = getattr(usage, "input_tokens", 0) or 0
        return ModelTurn(
            text=response.output_text or "",
            tool_calls=calls,
            input_tokens=max(0, total_in - cached),
            cache_read_tokens=cached,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def add_tool_results(self, results: list[ToolOutput]) -> None:
        for item in results:
            self.items.append({
                "type": "function_call_output",
                "call_id": item.id,
                "output": item.output,
            })

    def complete_text(self, system: str, user: str) -> TextReply:
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
        )
        usage = response.usage
        cached = _cached_tokens(usage)
        total_in = getattr(usage, "input_tokens", 0) or 0
        return TextReply(
            text=response.output_text or "",
            input_tokens=max(0, total_in - cached),
            cache_read_tokens=cached,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


def _parse_args(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cached_tokens(usage: object) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "input_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("input_tokens_details")
    if details is None:
        return 0
    if isinstance(details, dict):
        return details.get("cached_tokens") or 0
    return getattr(details, "cached_tokens", 0) or 0
