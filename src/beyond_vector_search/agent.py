"""agent.py — the search loop.

The model plans a search, the tools run, the model reads the results,
and that repeats until it can answer or it hits a turn or token cap.
Providers differ. This loop does not.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

from beyond_vector_search.config import Settings
from beyond_vector_search.display import colors_enabled, format_call, paint
from beyond_vector_search.prompts import SEARCH_PROMPT
from beyond_vector_search.providers.base import ModelTurn, Provider, ToolCall, ToolOutput
from beyond_vector_search.result import Result, Step
from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Toolset


def execute_calls(calls: list[ToolCall], toolset: Toolset) -> list[ToolOutput]:
    """Run every tool call in this turn. Order matches the model's request order."""
    if len(calls) <= 1:
        return [_run_one(call, toolset) for call in calls]
    workers = min(8, len(calls))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(toolset.call, call.name, call.arguments, call.id)
            for call in calls
        ]
        return [
            ToolOutput(id=call.id, output=future.result())
            for call, future in zip(calls, futures)
        ]


def _run_one(call: ToolCall, toolset: Toolset) -> ToolOutput:
    return ToolOutput(id=call.id, output=toolset.call(call.name, call.arguments, call.id))


def run_loop(
    question: str,
    provider: Provider,
    toolset: Toolset,
    *,
    system: str = SEARCH_PROMPT,
    max_turns: int = 25,
    max_input_tokens: int | None = None,
    verbose: bool = False,
    trace_prefix: str = "",
) -> Result:
    provider.start(question, toolset.definitions(), system)
    result = Result(answer="")
    started = time.time()
    use_color = verbose and colors_enabled(sys.stdout)
    step_no = 0
    announced = False

    for _ in range(max_turns):
        turn = provider.complete()
        _add_usage(result, turn)
        if not turn.tool_calls:
            result.answer = turn.text or "The model returned an empty answer."
            break

        spent = result.input_tokens + result.cache_read_tokens
        if max_input_tokens is not None and spent >= max_input_tokens:
            result.answer = "Stopped: hit the per-query token budget before a final answer."
            break

        if verbose and turn.tool_calls and not announced:
            print(paint("Search", "2", enabled=use_color))
            announced = True
        if verbose:
            for call in turn.tool_calls:
                step_no += 1
                line = f"{trace_prefix}{step_no:>2}  {format_call(call.name, call.arguments)}"
                print(paint(line, "2", enabled=use_color))

        outputs = execute_calls(turn.tool_calls, toolset)
        for call, output in zip(turn.tool_calls, outputs):
            children = list(toolset.take_children(call.id))
            result.trace.append(Step(call.name, call.arguments, output.output, children))
            if verbose and children:
                step_no = _print_steps(children, trace_prefix + "    ", step_no, use_color)
        result.tool_calls += len(turn.tool_calls)
        provider.add_tool_results(outputs)
    else:
        result.answer = "Stopped: hit MAX_TURNS without a final answer."

    result.seconds = round(time.time() - started, 1)
    return result


def ask(
    question: str,
    *,
    verbose: bool = True,
    settings: Settings | None = None,
    hybrid: bool = False,
    deep: bool = False,
    provider: Provider | None = None,
) -> Result:
    """Answer one question from the document root.

    hybrid adds a semantic_search tool backed by the local vector index.
    deep hands the search to sub-agents and keeps their scrap work out of
    the main context.
    """
    settings = settings or Settings.from_env()
    if not settings.docs_root.is_dir():
        raise FileNotFoundError(
            f"docs root does not exist: {settings.docs_root}. "
            "Set DOCS_ROOT or pass --docs. The sample corpus is examples/corpus."
        )
    corpus = Corpus(settings.docs_root, max_output_chars=settings.max_output_chars)
    toolset = Toolset(corpus)
    system = SEARCH_PROMPT

    if hybrid:
        from beyond_vector_search.rag import semantic_search_tool
        toolset.extras.append(semantic_search_tool(settings))

    if deep:
        from beyond_vector_search.subagent import attach_deep_search
        attach_deep_search(toolset, settings, hybrid=hybrid)
        toolset.only = {"list_files", "deep_search"}
        system = _orchestrator_prompt()

    active = provider if provider is not None else _make(settings)
    return run_loop(
        question,
        active,
        toolset,
        system=system,
        max_turns=settings.max_turns,
        max_input_tokens=settings.max_input_tokens,
        verbose=verbose,
    )


def _make(settings: Settings) -> Provider:
    from beyond_vector_search.providers import make_provider
    return make_provider(settings)


def _orchestrator_prompt() -> str:
    from beyond_vector_search.prompts import ORCHESTRATOR_PROMPT
    return ORCHESTRATOR_PROMPT


def _add_usage(result: Result, turn: ModelTurn) -> None:
    result.turns += 1
    result.input_tokens += turn.input_tokens
    result.cache_read_tokens += turn.cache_read_tokens
    result.output_tokens += turn.output_tokens


def _print_steps(steps: list[Step], prefix: str, step_no: int, use_color: bool) -> int:
    for step in steps:
        step_no += 1
        line = f"{prefix}{step_no:>2}  {format_call(step.name, step.arguments)}"
        print(paint(line, "2", enabled=use_color))
        if step.children:
            step_no = _print_steps(step.children, prefix + "    ", step_no, use_color)
    return step_no
