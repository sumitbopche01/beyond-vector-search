"""display.py — separate the search trace from the answer in the terminal."""

from __future__ import annotations

import os
import sys


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


def render_answer(answer: str, stats: str, *, enabled: bool) -> str:
    body = answer.strip() or "The model returned an empty answer."
    rule = paint("─" * 48, "2", enabled=enabled)
    return "\n".join([
        "",
        paint("Answer", "1", enabled=enabled),
        rule,
        body,
        "",
        paint(stats, "2", enabled=enabled),
    ])
