"""citations.py — check that an answer's citations point at real lines.

The prompt asks for file:line-range. This module finds those citations and
checks two things: the lines exist, and the required facts are in them.
An answer can mention the right words and still fail if it never cites
the lines that contain them.

For an unanswerable question, grounded means every citation is a real line
(or there are no citations). It does not mean the citation proves absence.
The keyword score is what checks that the answer declines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from beyond_vector_search.sandbox import TEXT_EXTENSIONS, Corpus

_EXT = "|".join(re.escape(ext[1:]) for ext in sorted(TEXT_EXTENSIONS))
_CITATION = re.compile(
    rf"(?<![\w./])(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:{_EXT})):(?P<start>\d+)(?:\s*[-–—]\s*(?P<end>\d+))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Citation:
    path: str
    start: int
    end: int


@dataclass(frozen=True)
class CitationScore:
    citations: tuple[Citation, ...]
    resolved: bool
    grounded: bool
    missing: tuple[str, ...]
    note: str


def find_citations(answer: str) -> list[Citation]:
    """Pull file:line and file:start-end citations out of an answer."""
    found: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _CITATION.finditer(answer or ""):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        path = _clean_citation_path(match.group("path"))
        if path is None:
            path = match.group("path").replace("\\", "/").strip()
        if not path:
            continue
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        found.append(Citation(path, start, end))
    return found


def score_citations(answer: str, item: object, corpus: Corpus) -> CitationScore:
    """Score one answer against the live files.

    `item` needs `expects_absence` and `must_include`, which Question has.
    """
    citations = find_citations(answer)
    resolved, spans, bad = _read_citations(citations, corpus)
    expects_absence = bool(getattr(item, "expects_absence", False))
    required = [] if expects_absence else [str(term) for term in getattr(item, "must_include", [])]

    if not resolved:
        shown = ", ".join(bad[:3])
        return CitationScore(tuple(citations), False, False, tuple(required), f"bad citation: {shown}")

    if expects_absence:
        return CitationScore(tuple(citations), True, True, (), "")

    if not citations:
        return CitationScore((), True, False, tuple(required), "no citations")

    haystack = "\n".join(spans).casefold()
    missing = tuple(term for term in required if term.casefold() not in haystack)
    if missing:
        shown = ", ".join(missing[:4])
        return CitationScore(
            tuple(citations), True, False, missing, f"missing in cited lines: {shown}",
        )
    return CitationScore(tuple(citations), True, True, (), "")


def _read_citations(citations: list[Citation], corpus: Corpus) -> tuple[bool, list[str], list[str]]:
    spans: list[str] = []
    bad: list[str] = []
    for citation in citations:
        target = _resolve(corpus, citation.path)
        if target is None or not target.is_file():
            bad.append(f"{citation.path}:{citation.start}")
            continue
        lines = corpus.read_lines(target)
        start, end = citation.start, citation.end
        if start < 1 or end < start or end > len(lines):
            bad.append(f"{citation.path}:{start}-{end}")
            continue
        spans.append("\n".join(lines[start - 1:end]))
    return (not bad), spans, bad


def _clean_citation_path(path: str) -> str | None:
    """Normalise a citation path. Reject `..` so a citation cannot be rewritten."""
    cleaned = path.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    parts = cleaned.split("/")
    if not cleaned or any(part in {"", ".", ".."} for part in parts):
        return None
    return cleaned


def _resolve(corpus: Corpus, path: str):
    if _clean_citation_path(path) is None:
        return None
    try:
        return corpus.safe_path(path)
    except (FileNotFoundError, ValueError, OSError):
        pass
    name = path.rsplit("/", 1)[-1]
    if "/" in path:
        return None
    matches = [
        item for item in corpus.root.rglob("*")
        if item.is_file() and item.name == name
        and not any(part.startswith(".") for part in item.relative_to(corpus.root).parts)
    ]
    if len(matches) == 1:
        return matches[0]
    return None
