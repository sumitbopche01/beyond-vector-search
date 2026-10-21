"""rag.py — classic vector RAG, plus the semantic_search tool for hybrid agents.

Chunking is the baseline the article compares against: fixed character
windows, one lookup, no second chance. The hybrid tool returns line
ranges so the agent can read the real file instead of trusting the chunk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from beyond_vector_search.config import Settings
from beyond_vector_search.prompts import RAG_PROMPT
from beyond_vector_search.result import Result
from beyond_vector_search.sandbox import TEXT_EXTENSIONS
from beyond_vector_search.tools import Param, Tool


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str
    start_line: int
    end_line: int


def chunk_document(source: str, text: str, size: int = 1500, overlap: int = 200) -> list[Chunk]:
    """Fixed-size character chunks. Line numbers are recorded so a hit can be re-read."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    if not text:
        return [Chunk(source, 0, "", 1, 1)]

    line_starts = [0]
    for index, char in enumerate(text):
        if char == "\n" and index + 1 < len(text):
            line_starts.append(index + 1)

    def line_at(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(Chunk(
            source=source,
            index=index,
            text=text[start:end],
            start_line=line_at(start),
            end_line=line_at(max(start, end - 1)),
        ))
        index += 1
        if end >= len(text):
            break
        start += step
    return chunks


def iter_corpus_chunks(root: Path, size: int = 1500, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS):
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        rel = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_document(rel, text, size=size, overlap=overlap))
    return chunks


def write_index(collection, chunks: list[Chunk]) -> None:
    """Replace each source file's chunks so a re-index does not keep stale windows."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source, []).append(chunk)
    for source, group in grouped.items():
        try:
            collection.delete(where={"source": source})
        except Exception:
            # An empty collection, or a backend that rejects a miss, is fine.
            pass
        collection.upsert(
            ids=[f"{source}#{chunk.index}" for chunk in group],
            documents=[chunk.text for chunk in group],
            metadatas=[{
                "source": chunk.source,
                "chunk": chunk.index,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            } for chunk in group],
        )


def open_collection(settings: Settings, reset: bool = False):
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "Install the RAG extra: pip install 'beyond-vector-search[rag]'"
        ) from exc
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    if reset:
        try:
            client.delete_collection("docs")
        except Exception:
            pass
    return client.get_or_create_collection("docs")


def build_index(settings: Settings | None = None, collection=None) -> int:
    settings = settings or Settings.from_env()
    if not settings.docs_root.is_dir():
        raise FileNotFoundError(f"docs root does not exist: {settings.docs_root}")
    chunks = iter_corpus_chunks(settings.docs_root)
    from beyond_vector_search.lexical import write_lexical
    from beyond_vector_search.sandbox import text_snapshot

    sources, newest = text_snapshot(settings.docs_root)
    write_lexical(settings.lexical_path, chunks, sources=sources, newest=newest)
    if collection is None:
        try:
            collection = open_collection(settings, reset=True)
        except ImportError as exc:
            raise ImportError(
                f"{exc}. Lexical index written to {settings.lexical_path}."
            ) from exc
    if chunks and collection is not None:
        write_index(collection, chunks)
    return len(chunks)


def format_hits(documents: list[str], metadatas: list[dict]) -> str:
    if not documents:
        return "No semantic matches."
    rows = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        source = meta.get("source", "?")
        start = meta.get("start_line", "?")
        end = meta.get("end_line", "?")
        chunk = meta.get("chunk", "?")
        preview = " ".join((doc or "").split())[:200]
        rows.append(f"{source}:{start}-{end} (chunk {chunk}): {preview!r}")
    return "\n".join(rows)


def _query(settings: Settings, query: str, k: int) -> tuple[list[str], list[dict]]:
    collection = open_collection(settings)
    count = collection.count()
    if count == 0:
        return [], []
    hits = collection.query(query_texts=[query], n_results=min(k, count))
    documents = (hits.get("documents") or [[]])[0] or []
    metadatas = (hits.get("metadatas") or [[]])[0] or []
    return list(documents), list(metadatas)


def semantic_search_tool(settings: Settings) -> Tool:
    def semantic_search(query: str, k: int = 8) -> str:
        try:
            limit = max(1, min(20, int(k)))
        except (TypeError, ValueError):
            limit = 8
        try:
            documents, metadatas = _query(settings, query, limit)
        except ImportError as exc:
            return f"ERROR ImportError: {exc}"
        except Exception as exc:
            return f"ERROR {type(exc).__name__}: {exc}"
        if not documents:
            return "No semantic index. Run `bvs index` first, or keep searching with grep."
        return format_hits(documents, metadatas)

    return Tool(
        "semantic_search",
        "Meaning-based search over all documents. Use for conceptual questions "
        "or when grep finds nothing. Returns file:line-range. Then read_file that source.",
        [
            Param("query", "string", "What you are looking for, in ordinary words", required=True),
            Param("k", "integer", "How many chunks to return. Default 8"),
        ],
        semantic_search,
    )


def rag_answer(question: str, settings: Settings | None = None, k: int = 5) -> Result:
    """One retrieval, then one generation. No loop and no second search."""
    settings = settings or Settings.from_env()
    started = time.time()
    documents, metadatas = _query(settings, question, max(1, k))
    if not documents:
        return Result(
            answer="No semantic index. Run `bvs index` first.",
            turns=1,
            seconds=round(time.time() - started, 1),
        )

    context = "\n\n".join(
        f"[{(meta or {}).get('source')} #{(meta or {}).get('chunk')} "
        f"lines {(meta or {}).get('start_line')}-{(meta or {}).get('end_line')}]\n{doc}"
        for doc, meta in zip(documents, metadatas)
    )
    from beyond_vector_search.providers import make_provider
    reply = make_provider(settings).complete_text(
        RAG_PROMPT,
        f"Context:\n{context}\n\nQuestion: {question}",
    )
    return Result(
        answer=reply.text,
        turns=1,
        input_tokens=reply.input_tokens,
        cache_read_tokens=reply.cache_read_tokens,
        output_tokens=reply.output_tokens,
        seconds=round(time.time() - started, 1),
    )
