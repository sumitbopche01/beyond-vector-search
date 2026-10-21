"""tree.py — heading tree for long documents.

This is the structural half of tree navigation: sections, nested headings,
and the line range of each one. Summaries are an optional LLM pass on top,
run once per document, not on every query.

corpus_tree is that map for the whole folder, saved next to the lexical
index so a later question can open it before grepping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from beyond_vector_search.parsing import parse_json
from beyond_vector_search.sandbox import Corpus, atomic_write, text_snapshot

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Node:
    level: int
    title: str
    start_line: int
    end_line: int = 0
    summary: str = ""
    children: list[Node] = field(default_factory=list)


def build_tree(lines: list[str]) -> list[Node]:
    """Nest markdown headings. A section runs until the next sibling."""
    found: list[tuple[int, int, str]] = []
    for number, line in enumerate(lines, 1):
        match = _HEADING.match(line)
        if match:
            found.append((number, len(match.group(1)), match.group(2)))

    roots: list[Node] = []
    stack: list[Node] = []
    for start, level, title in found:
        node = Node(level=level, title=title, start_line=start)
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    def assign_ends(nodes: list[Node], parent_end: int) -> None:
        for index, node in enumerate(nodes):
            if index + 1 < len(nodes):
                node.end_line = nodes[index + 1].start_line - 1
            else:
                node.end_line = parent_end
            assign_ends(node.children, node.end_line)

    if lines:
        assign_ends(roots, len(lines))
    return roots


def render_outline(lines: list[str]) -> str:
    return render_nodes(build_tree(lines))


def render_nodes(nodes: list[Node]) -> str:
    rows: list[str] = []

    def walk(level_nodes: list[Node], depth: int) -> None:
        for node in level_nodes:
            indent = "  " * depth
            suffix = f" - {node.summary}" if node.summary else ""
            heading = f"{'#' * node.level} {node.title}{suffix}"
            rows.append(f"{indent}{node.start_line}-{node.end_line}\t{heading}")
            walk(node.children, depth + 1)

    walk(nodes, 0)
    return "\n".join(rows)


def flatten(nodes: list[Node]) -> list[Node]:
    flat: list[Node] = []

    def walk(level_nodes: list[Node]) -> None:
        for node in level_nodes:
            flat.append(node)
            walk(node.children)

    walk(nodes)
    return flat


def summary_request(nodes: list[Node]) -> str:
    """Prompt body: one line per heading, so the model can hand summaries back."""
    lines = [
        "Write a one-sentence summary of what each section is for.",
        "Use the heading text only. Do not invent facts that are not implied by the title.",
        'Return a JSON array of {"start_line": number, "summary": string}.',
        "",
    ]
    for node in flatten(nodes):
        lines.append(f"{node.start_line}\t{'#' * node.level} {node.title}")
    return "\n".join(lines)


def render_corpus(corpus: Corpus, path: str = ".") -> str:
    """Heading tree for every text file under path, grouped by file."""
    base = corpus.safe_path(path)
    if base.is_file():
        files = [base] if corpus.is_text(base) else []
    else:
        files = [
            item for item in sorted(base.rglob("*"))
            if item.is_file() and corpus.is_text(item)
            and not any(part.startswith(".") for part in item.relative_to(corpus.root).parts)
        ]
    blocks: list[str] = []
    for item in files:
        rendered = render_outline(corpus.read_lines(item))
        if not rendered:
            continue
        blocks.append(f"{corpus.relative(item)}\n{rendered}")
    return "\n\n".join(blocks)


def load_corpus_tree(corpus: Corpus, cache: Path | None = None, path: str = ".") -> str:
    """Return the corpus heading tree, reusing the saved file when it is still fresh."""
    use_cache = cache is not None and path.strip() in {"", "."}
    sources, newest = text_snapshot(corpus.root)
    if use_cache and cache is not None and _tree_is_fresh(cache, sources, newest):
        cached = cache.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    text = render_corpus(corpus, path)
    if use_cache and cache is not None:
        atomic_write(cache, (text + "\n") if text else "")
        atomic_write(
            _stamp_path(cache),
            json.dumps({"sources": list(sources), "newest": newest}),
        )
    return text


def _stamp_path(cache: Path) -> Path:
    return cache.with_suffix(".stamp.json")


def _tree_is_fresh(cache: Path, sources: tuple[str, ...], newest: float) -> bool:
    stamp = _stamp_path(cache)
    if not cache.is_file() or not stamp.is_file():
        return False
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or data.get("sources") != list(sources):
        return False
    try:
        saved_newest = float(data.get("newest") or 0)
    except (TypeError, ValueError):
        return False
    return saved_newest >= newest


def apply_summaries(nodes: list[Node], payload: str) -> None:
    """Merge a model's JSON summaries onto the tree. Unknown lines are ignored."""
    data = parse_json(payload)
    if not isinstance(data, list):
        raise ValueError("summary payload must be a JSON array")
    by_line = {}
    for item in data:
        if isinstance(item, dict) and "start_line" in item and "summary" in item:
            by_line[int(item["start_line"])] = str(item["summary"]).strip()
    for node in flatten(nodes):
        if node.start_line in by_line:
            node.summary = by_line[node.start_line]
