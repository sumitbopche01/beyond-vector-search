"""Chunk coordinates, the keyword scorer, and the comparison report."""

from pathlib import Path

from beyond_vector_search.evaluate import (
    Question,
    Row,
    evaluate,
    judge_answer,
    load_questions,
    render_report,
    score_keywords,
)
from beyond_vector_search.rag import chunk_document, format_hits, write_index
from beyond_vector_search.result import Result

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
