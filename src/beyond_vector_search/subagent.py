"""subagent.py — a search sub-agent the orchestrator can call in parallel.

Each deep_search call has its own context window. The orchestrator only
sees the excerpts that come back, not the dead-end greps along the way.
"""

from __future__ import annotations

from beyond_vector_search.config import Settings
from beyond_vector_search.prompts import SUBAGENT_PROMPT
from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Param, Tool, Toolset


def attach_deep_search(toolset: Toolset, settings: Settings, hybrid: bool = False) -> None:
    """Add deep_search to an orchestrator toolset."""

    def deep_search(call_id: str, question: str) -> tuple[str, list]:
        from beyond_vector_search.agent import run_loop
        from beyond_vector_search.providers import make_provider

        child_settings = settings.for_search_agent()
        child = Toolset(Corpus(child_settings.docs_root, child_settings.max_output_chars))
        if hybrid:
            from beyond_vector_search.rag import semantic_search_tool
            child.extras.append(semantic_search_tool(child_settings))
        result = run_loop(
            question,
            make_provider(child_settings),
            child,
            system=SUBAGENT_PROMPT,
            max_turns=child_settings.max_turns,
            max_input_tokens=child_settings.max_input_tokens,
            verbose=False,
        )
        footer = f"\n\n[search: {result.turns} turns, {result.tool_calls} tool calls]"
        return result.answer + footer, result.trace

    toolset.register_special(
        Tool(
            "deep_search",
            "Search the documents in a fresh context and return only the relevant "
            "excerpts with (file:line-range) citations. Pass one self-contained question. "
            "Call this several times in one turn for independent sub-questions.",
            [Param("question", "string", "What to find in the documents", required=True)],
            lambda question: "",
        ),
        deep_search,
    )
