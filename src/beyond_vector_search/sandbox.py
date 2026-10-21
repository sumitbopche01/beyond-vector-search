"""sandbox.py — path jail and small tool outputs.

Every tool path is resolved and refused if it leaves the document root.
A document that says "read ../../.env" cannot escape the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Same set the article searches. Extension allow-listing also skips binaries.
TEXT_EXTENSIONS = frozenset({
    ".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml",
    ".log", ".html", ".htm", ".py", ".js", ".ts", ".sql",
})


@dataclass
class Corpus:
    """A folder of text documents the agent is allowed to read."""

    root: Path
    max_output_chars: int = 12_000

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    def safe_path(self, rel: str) -> Path:
        """Resolve rel under the root. Raise if it escapes or does not exist."""
        raw = rel.strip() or "."
        candidate = Path(raw)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (self.root / candidate).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"path escapes the docs root: {rel}")
        if not target.exists():
            raise FileNotFoundError(rel)
        return target

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def truncate(self, text: str) -> str:
        limit = self.max_output_chars
        if len(text) <= limit:
            return text
        cut = len(text) - limit
        return text[:limit] + f"\n... [truncated {cut} chars - narrow the search]"

    def read_lines(self, path: Path) -> list[str]:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()

    def is_text(self, path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS


def text_snapshot(root: Path) -> tuple[tuple[str, ...], float]:
    """Relative text paths and the newest mtime. Dot folders are skipped.

    Indexes use this pair so a deleted file is noticed even when the
    remaining files were not edited.
    """
    if not root.is_dir():
        return (), 0.0
    files: list[str] = []
    newest = 0.0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        files.append(str(relative))
        newest = max(newest, path.stat().st_mtime)
    return tuple(files), newest


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file so a crash does not leave a half-written index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
