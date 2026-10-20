"""claude.py — Claude messages API, with a rolling prompt cache.

The loop re-sends the whole history every turn. One cache breakpoint sits
on the system prompt. A second moves to the newest tool result, so the
next turn reads the previous conversation from cache.
"""

from __future__ import annotations

from beyond_vector_search.providers.base import ModelTurn, TextReply, ToolCall, ToolOutput
from beyond_vector_search.tools import Tool


class MissingExtraError(ImportError):
    """Raised when a provider SDK is not installed."""


class ClaudeProvider:
    def __init__(self, model: str, max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as exc:
            raise MissingExtraError(
                "Install the Anthropic extra: pip install 'beyond-vector-search[anthropic]'"
            ) from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.messages: list = []
        self.system: list = []
        self.tools: list = []

    def start(self, question: str, tools: list[Tool], system: str) -> None:
        self.tools = [tool.anthropic_schema() for tool in tools]
        self.system = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
        self.messages = [{"role": "user", "content": question}]

    def complete(self) -> ModelTurn:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.content})
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        usage = response.usage
        return ModelTurn(
            text="".join(texts),
            tool_calls=calls,
            input_tokens=usage.input_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            output_tokens=usage.output_tokens or 0,
        )

    def add_tool_results(self, results: list[ToolOutput]) -> None:
        blocks = [
            {"type": "tool_result", "tool_use_id": item.id, "content": item.output}
            for item in results
        ]
        self._move_cache_marker(blocks)
        self.messages.append({"role": "user", "content": blocks})

    def complete_text(self, system: str, user: str) -> TextReply:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return TextReply(
            text=text,
            input_tokens=usage.input_tokens or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            output_tokens=usage.output_tokens or 0,
        )

    def _move_cache_marker(self, newest: list[dict]) -> None:
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        if newest:
            newest[-1]["cache_control"] = {"type": "ephemeral"}
