"""evaluate.py — compare systems on the same questions.

Keyword checks are crude and transparent. The optional judge asks the model
whether the answer is actually right and whether the citations look real.
Split the report by question type: that split is where the decision lives.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from beyond_vector_search.parsing import parse_json
from beyond_vector_search.prompts import JUDGE_PROMPT
from beyond_vector_search.result import Result

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
) -> list[Row]:
    rows: list[Row] = []
    for item in questions:
        for name, answer_fn in systems.items():
            result = answer_fn(item.q)
            verdict = judge(item, result.answer) if judge else None
            rows.append(Row(
                system=name,
                question=item,
                result=result,
                keyword_ok=score_keywords(result.answer, item),
                verdict=verdict,
            ))
    return rows


def render_report(rows: list[Row]) -> str:
    if not rows:
        return "No results."
    systems = list(dict.fromkeys(row.system for row in rows))
    has_judge = any(row.verdict is not None for row in rows)
    header = f"{'system':<12}{'accuracy':>10}"
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

    lines.append("")
    lines.append("questions:")
    for row in rows:
        mark = "pass" if row.keyword_ok else "fail"
        judge = ""
        if row.verdict is not None:
            judge = "  judge=pass" if row.verdict.correct else "  judge=fail"
            if row.verdict.reason:
                judge += f"  ({row.verdict.reason})"
        lines.append(f"{row.system:<12}{mark:<6}{judge}  {row.question.type:<12}{row.question.q}")
    return "\n".join(lines)
