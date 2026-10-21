"""progress.py — one line at a time while a long command is working.

Eval installs a sink for the duration of the run. The agent loop and classic
RAG emit into it. With no sink, the calls do nothing, so tests stay quiet.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_sink: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar(
    "bvs_progress",
    default=None,
)


def note(message: str) -> None:
    """Send one progress line if a command is listening."""
    sink = _sink.get()
    if sink is not None:
        sink(message)


@contextmanager
def reporting(sink: Callable[[str], None]) -> Iterator[None]:
    """Show progress lines until the block exits. Nested workers inherit it."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
