"""evaluate.py — compare systems on the same questions.

Keyword checks are crude and transparent. Citation checks open the cited
lines and see whether the required facts are actually there. The optional
judge asks the model whether the answer is right. Split the report by
question type: that split is where the decision lives.

A system is allowed to lose. Write the report and keep it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from beyond_vector_search.citations import CitationScore, score_citations
from beyond_vector_search.parsing import parse_json
from beyond_vector_search.progress import note
from beyond_vector_search.prompts import JUDGE_PROMPT
from beyond_vector_search.result import Result, Step
from beyond_vector_search.sandbox import Corpus

ABSENT_PHRASES = (
    "not in the document",
    "not in the documents",
    "do not contain",
    "does not contain",
    "don't contain",
    "couldn't find",
    "could not find",
    "cannot find",
    "can't find",
    "no information",
    "not mentioned",
    "no mention",
    "aren't in",
    "are not in",
    "isn't in",
    "is not in",
    "not present",
    "not specified",
    "no such",
)


@dataclass(frozen=True)
class Question:
    q: str
    must_include: list[str] = field(default_factory=list)
    type: str = "lookup"
    absent: bool = False

    @property
    def expects_absence(self) -> bool:
        return self.absent or self.type == "unanswerable"


@dataclass
class JudgeVerdict:
    correct: bool
    citations_valid: bool
    reason: str


@dataclass
class Row:
    system: str
    question: Question
    result: Result
    keyword_ok: bool
    verdict: JudgeVerdict | None = None
    citation: CitationScore | None = None


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if not isinstance(raw, dict) or "q" not in raw:
            raise ValueError(f"{path}:{number}: each line needs a 'q' field")
        must = raw.get("must_include", [])
        if not isinstance(must, list) or not all(isinstance(item, str) for item in must):
            raise ValueError(f"{path}:{number}: must_include must be a list of strings")
        questions.append(Question(
            q=str(raw["q"]),
            must_include=list(must),
            type=str(raw.get("type", "lookup")),
            absent=bool(raw.get("absent", False)),
        ))
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions


def score_keywords(answer: str, item: Question) -> bool:
    """True when every required fact appears, or when an unanswerable question is declined."""
    lowered = answer.lower()
    if item.expects_absence:
        return any(phrase in lowered for phrase in ABSENT_PHRASES)
    return all(term.lower() in lowered for term in item.must_include)


def judge_answer(
    complete: Callable[[str, str], str],
    item: Question,
    answer: str,
) -> JudgeVerdict:
    prompt = (
        f"Question type: {item.type}\n"
        f"Question: {item.q}\n"
        f"Required facts (empty means the answer should say it is not in the documents): "
        f"{item.must_include}\n\n"
        f"Answer:\n{answer}"
    )
    try:
        raw = complete(JUDGE_PROMPT, prompt)
        data = parse_json(raw)
    except Exception as exc:
        return JudgeVerdict(False, False, f"judge failed: {type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        return JudgeVerdict(False, False, "judge did not return a JSON object")
    return JudgeVerdict(
        correct=bool(data.get("correct")),
        citations_valid=bool(data.get("citations_valid")),
        reason=str(data.get("reason", "")).strip(),
    )


def evaluate(
    questions: list[Question],
    systems: dict[str, Callable[[str], Result]],
    judge: Callable[[Question, str], JudgeVerdict] | None = None,
    corpus: Corpus | None = None,
) -> list[Row]:
    rows: list[Row] = []
    total = len(questions) * len(systems)
    done = 0
    for item in questions:
        preview = " ".join(item.q.split())
        if len(preview) > 72:
            preview = preview[:69] + "..."
        for name, answer_fn in systems.items():
            done += 1
            note(f"{done:>2}/{total:<2}  {name:<8} {item.type:<13} {preview}")
            result = answer_fn(item.q)
            verdict = None
            if judge:
                note("        judging")
                verdict = judge(item, result.answer)
            citation = score_citations(result.answer, item, corpus) if corpus is not None else None
            keyword_ok = score_keywords(result.answer, item)
            rows.append(Row(
                system=name,
                question=item,
                result=result,
                keyword_ok=keyword_ok,
                verdict=verdict,
                citation=citation,
            ))
            note("        " + _outcome(keyword_ok, citation, verdict, result))
    return rows


def _outcome(
    keyword_ok: bool,
    citation: CitationScore | None,
    verdict: JudgeVerdict | None,
    result: Result,
) -> str:
    parts = ["pass" if keyword_ok else "fail"]
    if citation is not None:
        parts.append("cited" if citation.grounded else "uncited")
    if verdict is not None:
        parts.append("judge=pass" if verdict.correct else "judge=fail")
    parts.append(f"{result.seconds}s")
    parts.append(f"{result.turns} turns")
    parts.append(f"{result.tool_calls} searches")
    return "  ".join(parts)


def render_report(rows: list[Row]) -> str:
    if not rows:
        return "No results."
    systems = list(dict.fromkeys(row.system for row in rows))
    has_judge = any(row.verdict is not None for row in rows)
    has_citations = any(row.citation is not None for row in rows)
    header = f"{'system':<12}{'accuracy':>10}"
    if has_citations:
        header += f"{'cited':>10}"
    if has_judge:
        header += f"{'judge':>10}"
    header += f"{'avg in':>10}{'avg cached':>12}{'avg out':>10}{'avg sec':>9}"
    lines = [header]

    grouped: dict[str, list[Row]] = {name: [] for name in systems}
    for row in rows:
        grouped[row.system].append(row)

    for name, group in grouped.items():
        n = len(group)
        accuracy = sum(row.keyword_ok for row in group) / n
        line = f"{name:<12}{accuracy:>10.0%}"
        if has_citations:
            cited = [row for row in group if row.citation is not None]
            rate = sum(row.citation.grounded for row in cited) / len(cited) if cited else 0
            line += f"{rate:>10.0%}"
        if has_judge:
            judged = [row for row in group if row.verdict is not None]
            rate = sum(row.verdict.correct for row in judged) / len(judged) if judged else 0
            line += f"{rate:>10.0%}"
        avg_in = sum(row.result.input_tokens for row in group) // n
        avg_cached = sum(row.result.cache_read_tokens for row in group) // n
        avg_out = sum(row.result.output_tokens for row in group) // n
        avg_sec = sum(row.result.seconds for row in group) / n
        line += f"{avg_in:>10}{avg_cached:>12}{avg_out:>10}{avg_sec:>9.1f}"
        lines.append(line)

    lines.append("")
    lines.append("by question type (keyword accuracy):")
    types = list(dict.fromkeys(row.question.type for row in rows))
    type_header = f"{'type':<16}" + "".join(f"{name:>12}" for name in systems)
    lines.append(type_header)
    for kind in types:
        cells = f"{kind:<16}"
        for name in systems:
            subset = [row for row in grouped[name] if row.question.type == kind]
            if not subset:
                cells += f"{'-':>12}"
            else:
                rate = sum(row.keyword_ok for row in subset) / len(subset)
                cells += f"{rate:>12.0%}"
        lines.append(cells)

    if has_citations:
        lines.append("")
        lines.append("by question type (citation support):")
        lines.append(type_header)
        for kind in types:
            cells = f"{kind:<16}"
            for name in systems:
                subset = [
                    row for row in grouped[name]
                    if row.question.type == kind and row.citation is not None
                ]
                if not subset:
                    cells += f"{'-':>12}"
                else:
                    rate = sum(row.citation.grounded for row in subset) / len(subset)
                    cells += f"{rate:>12.0%}"
            lines.append(cells)

    lines.append("")
    lines.append("questions:")
    for row in rows:
        mark = "pass" if row.keyword_ok else "fail"
        judge = ""
        if row.verdict is not None:
            judge = "  judge=pass" if row.verdict.correct else "  judge=fail"
            if row.verdict.reason:
                judge += f"  ({row.verdict.reason})"
        cite = ""
        if row.citation is not None:
            cite = "  cited" if row.citation.grounded else "  uncited"
            if row.citation.note and not row.citation.grounded:
                cite += f" ({row.citation.note})"
        lines.append(f"{row.system:<12}{mark:<6}{cite}{judge}  {row.question.type:<12}{row.question.q}")
    return "\n".join(lines)


_PREAMBLE = """\
# Eval

One run. Systems are allowed to lose.

Keyword accuracy is the share of answers that contain every required fact,
or, for an unanswerable question, that say the documents do not contain it.
Cited is the share of answers whose file:line-range citations resolve to
real lines and, when facts are required, contain those facts.
"""


def write_report(path: Path, rows: list[Row], *, settings: object | None = None) -> Path:
    """Write the markdown report, a JSONL sidecar, and an SVG comparison."""
    target = Path(path)
    if target.suffix.lower() != ".md":
        target = target / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    header = _PREAMBLE
    if settings is not None:
        provider = getattr(settings, "provider", "")
        model = getattr(settings, "model", "")
        docs = getattr(settings, "docs_root", "")
        try:
            docs = Path(docs).resolve().relative_to(Path.cwd().resolve())
        except (ValueError, OSError):
            docs = getattr(settings, "docs_root", "")
        header += f"\nDate {date.today().isoformat()}. Provider `{provider}`, model `{model}`, docs `{docs}`.\n"
    from beyond_vector_search.charts import render_comparison_chart
    from beyond_vector_search.sandbox import atomic_write

    chart = target.with_suffix(".svg")
    atomic_write(chart, render_comparison_chart(rows))
    image = f"\n![Keyword accuracy and citation support by system]({chart.name})\n"
    report = render_report(rows)
    atomic_write(target, header + image + "\n```\n" + report + "\n```\n")
    sidecar = target.with_suffix(".jsonl")
    lines = [_row_json(row) for row in rows]
    atomic_write(sidecar, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in lines))
    return target


def _row_json(row: Row) -> dict:
    result = row.result
    payload = {
        "system": row.system,
        "type": row.question.type,
        "q": row.question.q,
        "keyword_ok": row.keyword_ok,
        "answer": result.answer,
        "stop": result.stop,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "input_tokens": result.input_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "output_tokens": result.output_tokens,
        "seconds": result.seconds,
        "trace": [_step_json(step) for step in result.trace],
    }
    if row.citation is not None:
        payload["grounded"] = row.citation.grounded
        payload["resolved"] = row.citation.resolved
        payload["missing"] = list(row.citation.missing)
        payload["note"] = row.citation.note
        payload["citations"] = [
            {"path": item.path, "start": item.start, "end": item.end}
            for item in row.citation.citations
        ]
    if row.verdict is not None:
        payload["judge_correct"] = row.verdict.correct
        payload["judge_citations_valid"] = row.verdict.citations_valid
        payload["judge_reason"] = row.verdict.reason
    return payload


def _step_json(step: Step) -> dict:
    return {
        "name": step.name,
        "arguments": {key: _jsonable(value) for key, value in step.arguments.items()},
        "output": step.output[:400],
        "children": [_step_json(child) for child in step.children],
    }


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
