"""The agent loop, driven by a scripted provider so no API is called."""

import time
from pathlib import Path

from beyond_vector_search.agent import ask, execute_calls, run_loop
from beyond_vector_search.cli import main
from beyond_vector_search.config import Settings
from beyond_vector_search.display import format_trace_report, render_answer
from beyond_vector_search.providers.base import ModelTurn, ToolCall
from beyond_vector_search.result import Result, Step
from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Param, Tool, Toolset

ROOT = Path(__file__).resolve().parents[1]


class Scripted:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.seen: tuple | None = None
        self.results: list = []

    def start(self, question: str, tools: list, system: str) -> None:
        self.seen = (question, [tool.name for tool in tools], system)

    def complete(self) -> ModelTurn:
        if not self.turns:
            return ModelTurn(text="", tool_calls=[])
        return self.turns.pop(0)

    def add_tool_results(self, results: list) -> None:
        self.results.append(results)

    def complete_text(self, system: str, user: str):
        raise AssertionError("one-shot completion is not used in the loop tests")


def test_loop_greps_then_answers(tmp_path: Path, capsys) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "addendum.md").write_text(
        "# Addendum\n\nThe cap is 2x annual fees, except as set out in MSA §9.3.\n",
        encoding="utf-8",
    )
    provider = Scripted([
        ModelTurn(text="", tool_calls=[ToolCall("c1", "grep", {"pattern": "2x"})]),
        ModelTurn(text="The cap is 2x annual fees (addendum.md:3).", tool_calls=[]),
    ])
    result = run_loop(
        "What is the cap?",
        provider,
        Toolset(Corpus(root)),
        verbose=False,
        max_turns=5,
    )
    assert result.answer.startswith("The cap is 2x")
    assert result.tool_calls == 1
    assert result.trace[0].name == "grep"
    assert "addendum.md" in provider.results[0][0].output


def test_a_listening_eval_sees_the_model_call_and_the_search(tmp_path: Path) -> None:
    from beyond_vector_search.progress import reporting

    root = tmp_path / "docs"
    root.mkdir()
    (root / "addendum.md").write_text("The cap is 2x annual fees.\n", encoding="utf-8")
    provider = Scripted([
        ModelTurn(text="", tool_calls=[ToolCall("c1", "grep", {"pattern": "2x"})]),
        ModelTurn(text="The cap is 2x.", tool_calls=[]),
    ])
    lines: list[str] = []
    with reporting(lines.append):
        run_loop("What is the cap?", provider, Toolset(Corpus(root)), verbose=False, max_turns=4)
    assert "model" in lines[0]
    assert any("grep" in line and "pattern=2x" in line for line in lines)


def test_verbose_trace_is_a_numbered_search_list(tmp_path: Path, capsys) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "addendum.md").write_text("The cap is 2x annual fees.\n", encoding="utf-8")
    provider = Scripted([
        ModelTurn(text="", tool_calls=[ToolCall("c1", "grep", {"pattern": "2x"})]),
        ModelTurn(text="The cap is 2x annual fees.", tool_calls=[]),
    ])
    run_loop("What is the cap?", provider, Toolset(Corpus(root)), verbose=True, max_turns=4)
    out = capsys.readouterr().out
    assert "Search\n" in out
    assert "1  grep  pattern=2x" in out
    assert "Answer" not in out


def test_answer_is_printed_apart_from_the_search() -> None:
    text = render_answer("- The cap is 2x annual fees.", "5 turns  ·  4 searches  ·  18.8s", enabled=False)
    assert text.index("Answer") < text.index("The cap is 2x")
    assert text.strip().endswith("18.8s")
    assert "Trace" not in text


def test_trace_report_flags_a_repeated_search() -> None:
    result = Result(
        answer="Stopped: hit MAX_TURNS without a final answer.",
        stop="max_turns",
        tool_calls=2,
        trace=[
            Step("grep", {"pattern": "2x"}, "hit"),
            Step("grep", {"pattern": "2x"}, "hit"),
            Step("read_file", {"path": "a.md", "start_line": 1}, "body"),
        ],
    )
    note = format_trace_report(result)
    assert "turn cap" in note
    assert "grep × 2" in note
    assert "repeated: grep pattern=2x × 2" in note
    text = render_answer(result.answer, "2 turns", enabled=False, trace_note=note)
    assert text.index("Answer") < text.index("Trace") < text.index("2 turns")


def test_turn_cap_and_token_budget(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nnothing here\n", encoding="utf-8")
    toolset = Toolset(Corpus(root))
    always = ModelTurn(text="", tool_calls=[ToolCall("c", "grep", {"pattern": "missing"})])

    capped = run_loop("q", Scripted([always, always, always]), toolset, max_turns=2)
    assert "MAX_TURNS" in capped.answer
    assert capped.turns == 2
    assert capped.stop == "max_turns"

    budget = run_loop(
        "q",
        Scripted([ModelTurn("", [ToolCall("c", "grep", {"pattern": "missing"})], input_tokens=50)]),
        toolset,
        max_turns=5,
        max_input_tokens=10,
    )
    assert "token budget" in budget.answer
    assert budget.tool_calls == 0
    assert budget.stop == "token_budget"


def test_parallel_results_keep_request_order() -> None:
    class Slow:
        def call(self, name: str, args: dict, call_id: str = "") -> str:
            time.sleep(args["delay"])
            return args["marker"]

        def take_children(self, call_id: str) -> list:
            return []

    calls = [
        ToolCall("1", "t", {"marker": "first", "delay": 0.05}),
        ToolCall("2", "t", {"marker": "second", "delay": 0}),
    ]
    outputs = execute_calls(calls, Slow())
    assert [item.output for item in outputs] == ["first", "second"]


def test_subagent_trace_is_nested(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n", encoding="utf-8")
    toolset = Toolset(Corpus(root))

    def deep_search(call_id: str, question: str) -> tuple[str, list]:
        return "excerpt (a.md:1)", [Step("grep", {"pattern": question}, "hit")]

    toolset.register_special(
        Tool(
            "deep_search",
            "search",
            [Param("question", "string", "q", required=True)],
            lambda question: "",
        ),
        deep_search,
    )
    provider = Scripted([
        ModelTurn("", [ToolCall("d1", "deep_search", {"question": "cap"})]),
        ModelTurn("The excerpt says cap (a.md:1).", []),
    ])
    result = run_loop("cap?", provider, toolset, max_turns=4)
    assert result.trace[0].children[0].name == "grep"
    assert "excerpt" in provider.results[0][0].output


def test_ask_selects_orchestrator_tools_without_calling_a_model(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nhello\n", encoding="utf-8")
    settings = Settings(docs_root=root, max_turns=2)
    provider = Scripted([ModelTurn("done", [])])
    ask("hello", verbose=False, settings=settings, provider=provider, deep=True)
    assert provider.seen is not None
    names = provider.seen[1]
    assert names == ["list_files", "corpus_tree", "deep_search"]
    assert "delegating search" in provider.seen[2]

    hybrid = Scripted([ModelTurn("done", [])])
    ask("hello", verbose=False, settings=settings, provider=hybrid, hybrid=True)
    assert "semantic_search" in hybrid.seen[1]
    assert "lexical_search" in hybrid.seen[1]
    assert "grep" in hybrid.seen[1]


def test_cli_grep_on_the_sample_corpus(capsys) -> None:
    code = main(["grep", "2x annual", "--docs", str(ROOT / "examples" / "corpus")])
    captured = capsys.readouterr()
    assert code == 0
    assert "enterprise_addendum.md" in captured.out

    escaped = main(["grep", "secret", "--path", "../.env", "--docs", str(ROOT / "examples" / "corpus")])
    errors = capsys.readouterr()
    assert escaped == 1
    assert "escapes" in errors.err
