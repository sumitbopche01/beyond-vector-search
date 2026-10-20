"""ingest.py — turn PDFs, Word files, and HTML into text the agent can grep.

This is the only ingestion step agentic search needs: keep headings, keep
page numbers, and mirror the source folders. No chunking and no embeddings.
Re-running skips files whose output is already newer than the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

_TO_MARKDOWN = {".pdf", ".docx", ".html", ".htm", ".txt"}
_COPY = {".md", ".rst", ".csv", ".json", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".sql"}


@dataclass(frozen=True)
class IngestResult:
    source: str
    target: str
    status: str
    detail: str = ""


def ingest(src: Path, dst: Path) -> list[IngestResult]:
    source = Path(src).expanduser().resolve()
    dest = Path(dst).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (dest == source or _is_inside(dest, source)):
        raise ValueError("destination must be outside the source tree")

    if source.is_file():
        files = [source]
    else:
        files = [
            path for path in sorted(source.rglob("*"))
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(source).parts)
        ]

    results: list[IngestResult] = []
    for path in files:
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        results.append(_convert(path, dest / relative))
    return results


def pdf_to_markdown(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pip install 'beyond-vector-search[ingest]'") from exc

    reader = PdfReader(pdf)
    parts = [f"# {pdf.stem}\n"]
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+\n", "\n", text).strip()
        body = text or "[no text layer - scanned page, needs OCR]"
        parts.append(f"## Page {number}\n\n{body}\n")
    return "\n".join(parts)


def docx_to_markdown(path: Path) -> str:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise ImportError("pip install 'beyond-vector-search[ingest]'") from exc

    document = Document(str(path))
    parts = [f"# {path.stem}\n"]
    for block in _docx_blocks(document, Paragraph, Table):
        if isinstance(block, Paragraph):
            text = " ".join(block.text.split())
            if not text:
                continue
            style = block.style.name if block.style is not None else ""
            if style.startswith("Heading"):
                parts.append(f"{'#' * _heading_level(style)} {text}\n")
            else:
                parts.append(text + "\n")
        else:
            for row in block.rows:
                cells = [" ".join(cell.text.split()) for cell in row.cells]
                parts.append("| " + " | ".join(cells) + " |")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def html_to_markdown(path: Path) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    body = parser.markdown().strip()
    if not body:
        body = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not body.lstrip().startswith("#"):
        body = f"# {path.stem}\n\n{body}"
    return body.rstrip() + "\n"


def txt_to_markdown(path: Path) -> str:
    body = path.read_text(encoding="utf-8", errors="ignore").strip()
    return f"# {path.stem}\n\n{body}\n"


class _HTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0
        self._buf: list[str] = []
        self._heading = 0
        self._capture = False
        self._tag = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}:
            self._flush()
            self._capture = True
            self._heading = int(tag[1]) if tag.startswith("h") else 0
            self._buf = []
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip or not self._capture:
            return
        self._buf.append(data)

    def _flush(self) -> None:
        if not self._capture:
            return
        text = " ".join("".join(self._buf).split())
        self._capture = False
        self._buf = []
        if not text:
            return
        if self._heading:
            self.parts.append(f"{'#' * self._heading} {text}\n")
        elif getattr(self, "_tag", "") == "li":
            self.parts.append(f"- {text}\n")
        else:
            self.parts.append(text + "\n")

    def markdown(self) -> str:
        self._flush()
        return "\n".join(self.parts)


def _convert(path: Path, dest_base: Path) -> IngestResult:
    suffix = path.suffix.lower()
    if suffix in _TO_MARKDOWN:
        target = dest_base.with_suffix(".md")
        writers = {
            ".pdf": pdf_to_markdown,
            ".docx": docx_to_markdown,
            ".html": html_to_markdown,
            ".htm": html_to_markdown,
            ".txt": txt_to_markdown,
        }
        writer = writers[suffix]
    elif suffix in _COPY:
        target = dest_base
        writer = _read
    else:
        return IngestResult(str(path), "", "unsupported", suffix or "no extension")

    if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return IngestResult(str(path), str(target), "skipped")
    try:
        text = writer(path)
    except Exception as exc:
        return IngestResult(str(path), str(target), "error", f"{type(exc).__name__}: {exc}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return IngestResult(str(path), str(target), "wrote")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _docx_blocks(document, paragraph_type, table_type):
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield paragraph_type(child, document)
        elif child.tag.endswith("}tbl"):
            yield table_type(child, document)


def _heading_level(style: str) -> int:
    tail = style.split()[-1]
    if tail.isdigit():
        return min(6, max(1, int(tail)))
    return 2


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True
