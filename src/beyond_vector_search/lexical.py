"""lexical.py — BM25 over the same chunks the vector index uses.

Dense search misses IDs, clause numbers, and exact names. This index
scores word overlap and returns file:line-range, same as semantic_search.
No extra package: Okapi BM25 on the tokens in each chunk.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from beyond_vector_search.config import Settings
from beyond_vector_search.sandbox import atomic_write, text_snapshot
from beyond_vector_search.tools import Param, Tool

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_K1 = 1.5
_B = 0.75
# Query words that appear in almost every sentence. They stay in the document
# so a real ID next to "the" still scores; they are dropped from the query.
_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "as", "by", "at", "from", "that", "this",
    "it", "we", "do", "how", "what", "when", "where", "who", "which", "does",
    "did", "not", "no", "our", "your", "their", "there", "if",
})


def tokenize(text: str) -> list[str]:
    """Lowercase tokens. A hyphenated ID is kept whole and split into parts."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        tokens.append(token)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens


def write_lexical(
    path: Path,
    chunks: list,
    *,
    sources: tuple[str, ...] | None = None,
    newest: float | None = None,
) -> None:
    """Persist chunk text, line ranges, and the file list those chunks came from."""
    if sources is None:
        sources = tuple(dict.fromkeys(_source_of(chunk) for chunk in chunks))
    payload = {
        "version": 2,
        "sources": list(sources),
        "newest": 0.0 if newest is None else newest,
        "chunks": [_chunk_dict(chunk) for chunk in chunks],
    }
    atomic_write(path, json.dumps(payload, ensure_ascii=False))


def load_lexical(settings: Settings) -> list[dict]:
    """Return chunks, rebuilding when a file is edited, added, or deleted."""
    path = settings.lexical_path
    sources, newest = text_snapshot(settings.docs_root)
    if path.is_file() and _index_is_fresh(path, sources, newest):
        data = json.loads(path.read_text(encoding="utf-8"))
        return [row for row in data.get("chunks", []) if isinstance(row, dict)]

    from beyond_vector_search.rag import iter_corpus_chunks

    built = iter_corpus_chunks(settings.docs_root)
    write_lexical(path, built, sources=sources, newest=newest)
    return [_chunk_dict(chunk) for chunk in built]


def search_chunks(chunks: list[dict], query: str, k: int) -> list[dict]:
    """Top chunks by Okapi BM25. A chunk that shares no query term is dropped."""
    docs = [(row, tokenize(str(row.get("text", "")))) for row in chunks]
    docs = [(row, tokens) for row, tokens in docs if tokens]
    if not docs:
        return []
    query_terms = [term for term in tokenize(query) if term not in _STOP]
    if not query_terms:
        query_terms = tokenize(query)
    if not query_terms:
        return []

    df: Counter[str] = Counter()
    for _row, tokens in docs:
        df.update(set(tokens))
    n_docs = len(docs)
    avgdl = sum(len(tokens) for _row, tokens in docs) / n_docs

    ranked: list[tuple[float, dict]] = []
    for row, tokens in docs:
        score = _bm25(query_terms, tokens, df, n_docs, avgdl)
        if score > 0:
            ranked.append((score, row))
    ranked.sort(key=lambda item: (
        -item[0],
        str(item[1].get("source", "")),
        int(item[1].get("chunk") or 0),
    ))
    return [row for _score, row in ranked[:k]]


def lexical_search_tool(settings: Settings) -> Tool:
    """Word-overlap search. The index is built on first use if `bvs index` was not run."""

    def lexical_search(query: str, k: int = 8) -> str:
        try:
            limit = max(1, min(20, int(k)))
        except (TypeError, ValueError):
            limit = 8
        try:
            rows = search_chunks(load_lexical(settings), query, limit)
        except Exception as exc:
            return f"ERROR {type(exc).__name__}: {exc}"
        if not rows:
            return "No lexical matches. Try grep with a shorter stem, or semantic_search."
        from beyond_vector_search.rag import format_hits

        documents = [str(row.get("text", "")) for row in rows]
        metadatas = [
            {
                "source": row.get("source", "?"),
                "chunk": row.get("chunk", "?"),
                "start_line": row.get("start_line", "?"),
                "end_line": row.get("end_line", "?"),
            }
            for row in rows
        ]
        return format_hits(documents, metadatas)

    return Tool(
        "lexical_search",
        "Word-overlap search over all documents (BM25). Use for names, IDs, clause "
        "numbers, and error codes when you know the words. Returns file:line-range. "
        "Then read_file that source.",
        [
            Param("query", "string", "Words to match: a name, an ID, a clause number", required=True),
            Param("k", "integer", "How many chunks to return. Default 8"),
        ],
        lexical_search,
    )


def attach_hybrid_tools(toolset, settings: Settings) -> None:
    """Give an agent lexical search and the dense semantic tool."""
    from beyond_vector_search.rag import semantic_search_tool

    toolset.extras.append(lexical_search_tool(settings))
    toolset.extras.append(semantic_search_tool(settings))


def _bm25(query_terms: list[str], doc_tokens: list[str], df: Counter, n_docs: int, avgdl: float) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in set(query_terms):
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        n = df.get(term, 0)
        idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
        denom = freq + _K1 * (1 - _B + _B * dl / avgdl)
        score += idf * (freq * (_K1 + 1)) / denom
    return score


def _source_of(chunk) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("source", ""))
    return str(chunk.source)


def _index_is_fresh(path: Path, sources: tuple[str, ...], newest: float) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or data.get("version") != 2:
        return False
    saved = data.get("sources")
    if not isinstance(saved, list) or saved != list(sources):
        return False
    try:
        saved_newest = float(data.get("newest") or 0)
    except (TypeError, ValueError):
        return False
    return saved_newest >= newest and isinstance(data.get("chunks"), list)


def _chunk_dict(chunk) -> dict:
    return {
        "source": chunk.source,
        "chunk": chunk.index,
        "text": chunk.text,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }
