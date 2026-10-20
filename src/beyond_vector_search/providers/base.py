"""base.py — the small surface every model backend implements.

The agent loop does not know about Anthropic content blocks or OpenAI
response items. Each provider keeps its own history and reports a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from beyond_vector_search.tools import Tool


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class ToolOutput:
    id: str
    output: str


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TextReply:
    """A single completion with no tools. Used by RAG, the judge, and tree summaries."""

    text: str
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    def start(self, question: str, tools: list[Tool], system: str) -> None:
        """Begin a new conversation."""

    def complete(self) -> ModelTurn:
        """Take one model step. Records the assistant turn in the provider's history."""

    def add_tool_results(self, results: list[ToolOutput]) -> None:
        """Append tool outputs so the next complete() can see them."""

    def complete_text(self, system: str, user: str) -> TextReply:
        """One-shot completion that does not touch the agent conversation."""
