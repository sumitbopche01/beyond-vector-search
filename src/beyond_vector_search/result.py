"""result.py — what ask() and the eval harness return."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    """One tool call. Children are the trace of a search sub-agent."""

    name: str
    arguments: dict
    output: str
    children: list[Step] = field(default_factory=list)


@dataclass
class Result:
    """Token and timing totals for a single question.

    input_tokens counts uncached input. cache_read_tokens is the portion
    served from the prompt cache. Add them for the full input bill.
    """

    answer: str
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    stop: str = "answered"
    trace: list[Step] = field(default_factory=list)

    def stats_line(self) -> str:
        return (
            f"{self.turns} turns  ·  {self.tool_calls} searches  ·  {self.seconds}s"
            f"  ·  {self.input_tokens} in, {self.cache_read_tokens} cached, {self.output_tokens} out"
        )
