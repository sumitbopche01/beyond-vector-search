"""display.py — separate the search trace from the answer in the terminal.

The trace report is the short version of that search: which tools ran,
which search was repeated, and whether the loop stopped on a cap.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

from beyond_vector_search.result import Result, Step


def format_call(name: str, arguments: dict) -> str:
    """One tool call on one line, without a JSON blob."""
    if not arguments:
        return name
    parts: list[str] = []
    for key, value in arguments.items():
        text = str(value).replace("\n", " ")
        if len(text) > 72:
            text = text[:69] + "..."
        parts.append(f"{key}={text}")
    return f"{name}  {', '.join(parts)}"


def colors_enabled(stream: object | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    target = stream if stream is not None else sys.stdout
    check = getattr(target, "isatty", None)
    return bool(check and check())


def paint(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def format_trace_report(result: Result) -> str:
    """One short block: tool counts, repeated searches, and why the loop stopped."""
    steps = list(_iter_steps(result.trace))
    if not steps and result.stop == "answered":
        return ""
    lines: list[str] = []
    if result.stop == "token_budget":
        lines.append("stopped: token budget, before a final answer")
    elif result.stop == "max_turns":
        lines.append("stopped: turn cap, before a final answer")
    if steps:
        counts = Counter(step.name for step in steps)
        tools = ", ".join(f"{name} × {count}" for name, count in counts.most_common())
        lines.append(f"tools: {tools}")
        repeats = [
            f"{signature} × {count}"
            for signature, count in Counter(_signature(step) for step in steps).items()
            if count >= 2
        ]
        if repeats:
            lines.append("repeated: " + "; ".join(repeats))
    return "\n".join(lines)


def render_answer(answer: str, stats: str, *, enabled: bool, trace_note: str = "") -> str:
    body = answer.strip() or "The model returned an empty answer."
    rule = paint("─" * 48, "2", enabled=enabled)
    parts = [
        "",
        paint("Answer", "1", enabled=enabled),
        rule,
        body,
    ]
    note = trace_note.strip()
    if note:
        parts.extend(["", paint("Trace", "1", enabled=enabled), paint(note, "2", enabled=enabled)])
    parts.extend(["", paint(stats, "2", enabled=enabled)])
    return "\n".join(parts)


def _iter_steps(steps: list[Step]):
    for step in steps:
        yield step
        yield from _iter_steps(step.children)


def _signature(step: Step) -> str:
    keys = ("pattern", "path", "query", "question", "start_line", "end_line")
    picked = [(key, step.arguments[key]) for key in keys if key in step.arguments and step.arguments[key] is not None]
    if not picked:
        picked = list(step.arguments.items())
    rendered = ", ".join(f"{key}={value}" for key, value in picked)
    return f"{step.name} {rendered}".strip()
