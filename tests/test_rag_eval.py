"""Chunk coordinates, the keyword scorer, the citation column, and the saved report."""

import json
from pathlib import Path

from beyond_vector_search.cli import main
from beyond_vector_search.evaluate import (
    Question,
    Row,
    evaluate,
    judge_answer,
    load_questions,
    render_report,
    score_keywords,
    write_report,
)
from beyond_vector_search.rag import chunk_document, format_hits, write_index
from beyond_vector_search.result import Result
from beyond_vector_search.sandbox import Corpus

ROOT = Path(__file__).resolve().parents[1]


def test_chunks_remember_the_heading_line() -> None:
    lines = ["intro"] * 20 + ["## Refunds", "The window is 14 days."]
    text = "\n".join(lines) + "\n"
    body_line = lines.index("The window is 14 days.") + 1
    chunks = chunk_document("policies/refund_policy_2025.md", text, size=80, overlap=20)
    covering = [chunk for chunk in chunks if "14 days" in chunk.text]
    assert covering
    assert any(chunk.start_line <= body_line <= chunk.end_line for chunk in covering)


class _Collection:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict]] = {}

    def delete(self, where: dict) -> None:
        source = where["source"]
        self.rows = {key: value for key, value in self.rows.items() if value[1]["source"] != source}

    def upsert(self, ids, documents, metadatas) -> None:
        for key, doc, meta in zip(ids, documents, metadatas):
            self.rows[key] = (doc, meta)


def test_reindex_replaces_stale_chunks_for_a_file() -> None:
    collection = _Collection()
    write_index(collection, chunk_document("a.md", "hello " * 80, size=30, overlap=5))
    assert len(collection.rows) > 1
    write_index(collection, chunk_document("a.md", "bye"))
    assert list(collection.rows.values()) == [("bye", {
        "source": "a.md",
        "chunk": 0,
        "start_line": 1,
        "end_line": 1,
    })]


def test_hit_format_includes_a_line_range() -> None:
    text = format_hits(
        ["cap is 2x annual fees"],
        [{"source": "contracts/enterprise_addendum.md", "chunk": 0, "start_line": 8, "end_line": 12}],
    )
    assert "contracts/enterprise_addendum.md:8-12" in text


def test_keyword_score_and_absence() -> None:
    lookup = Question("What is the cap?", ["2x"], "lookup")
    missing = Question("Tokyo on-call?", [], "unanswerable")
    assert score_keywords("The cap is 2x annual fees.", lookup)
    assert not score_keywords("The cap is 2x annual fees.", missing)
    assert score_keywords("That is not in the documents.", missing)


def test_judge_parses_fenced_json_and_survives_garbage() -> None:
    item = Question("cap?", ["2x"], "multi_hop")
    verdict = judge_answer(
        lambda _system, _user: '{"correct": true, "citations_valid": false, "reason": "partial"}',
        item,
        "2x",
    )
    assert verdict.correct
    assert not verdict.citations_valid

    broken = judge_answer(lambda _system, _user: "not json", item, "2x")
    assert not broken.correct
    assert "judge failed" in broken.reason


def test_report_splits_by_question_type() -> None:
    hit = Question("cap?", ["2x"], "multi_hop")
    rows = [
        Row("agentic", hit, Result("2x at 9.3", input_tokens=10, seconds=1), True),
        Row("rag", hit, Result("no cap found", input_tokens=4, seconds=0.2), False),
    ]
    text = render_report(rows)
    assert "100%" in text
    assert "0%" in text
    assert "multi_hop" in text


def test_sample_questions_load() -> None:
    questions = load_questions(ROOT / "examples" / "questions.jsonl")
    assert len(questions) == 6
    assert {item.type for item in questions} >= {"multi_hop", "synonym", "unanswerable"}


def test_evaluate_uses_the_supplied_systems() -> None:
    item = Question("cap?", ["2x"], "lookup")

    def agentic(_question: str) -> Result:
        return Result("2x annual fees")

    def rag(_question: str) -> Result:
        return Result("fees in general")

    rows = evaluate([item], {"agentic": agentic, "rag": rag})
    assert [row.keyword_ok for row in rows] == [True, False]


def test_evaluate_prints_each_question_while_it_runs() -> None:
    from beyond_vector_search.progress import reporting

    lines: list[str] = []
    item = Question("What is the cap?", ["2x"], "lookup")
    with reporting(lines.append):
        evaluate([item], {"agentic": lambda _question: Result("2x", seconds=1.2, turns=3, tool_calls=2)})
    text = "\n".join(lines)
    assert "1/1" in text
    assert "agentic" in text
    assert "What is the cap?" in text
    assert "pass" in text
    assert "1.2s" in text
    assert "2 searches" in text


def test_report_adds_a_citation_column_and_a_sidecar(tmp_path: Path) -> None:
    item = Question("cap?", ["2x"], "lookup")
    corpus = Corpus(ROOT / "examples" / "corpus")
    rows = evaluate(
        [item],
        {
            "agentic": lambda _question: Result(
                "The cap is 2x (contracts/enterprise_addendum.md:11)."
            ),
            "rag": lambda _question: Result("The cap is 2x somewhere."),
        },
        corpus=corpus,
    )
    text = render_report(rows)
    assert "cited" in text
    assert "uncited" in text
    written = write_report(tmp_path / "report.md", rows)
    assert "allowed to lose" in written.read_text(encoding="utf-8")
    saved = [
        json.loads(line)
        for line in written.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert saved[0]["grounded"] is True
    assert saved[1]["grounded"] is False
    chart = written.with_suffix(".svg").read_text(encoding="utf-8")
    assert "agentic 100%" in chart
    assert "rag 0%" in chart
    assert "Keyword accuracy by question type" in chart


def test_eval_command_writes_the_report(tmp_path: Path, monkeypatch, capsys) -> None:
    def fake_ask(question, **kwargs):
        return Result(
            "The cap is 2x (contracts/enterprise_addendum.md:11). "
            "Section 9.3 excludes confidentiality (contracts/msa_2024.md:31-37).",
            turns=2,
            tool_calls=2,
            seconds=0.1,
        )

    def fake_rag(question, **kwargs):
        return Result("The cap is 2x somewhere.", turns=1, seconds=0.1)

    monkeypatch.setattr("beyond_vector_search.agent.ask", fake_ask)
    monkeypatch.setattr("beyond_vector_search.rag.rag_answer", fake_rag)
    out = tmp_path / "report.md"
    code = main([
        "eval",
        str(ROOT / "examples" / "questions.jsonl"),
        "--docs",
        str(ROOT / "examples" / "corpus"),
        "--systems",
        "agentic,rag",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert out.is_file()
    assert out.with_suffix(".jsonl").is_file()
    assert "cited" in captured.out
    assert "6 questions" in captured.err
    assert "1/12" in captured.err
    assert "agentic" in captured.err
    assert "pass" in captured.err
