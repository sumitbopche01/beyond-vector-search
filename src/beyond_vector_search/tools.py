"""tools.py — the search tools and their JSON schemas.

list_files, grep, read_file, outline, and corpus_tree are deliberately boring.
The model decides which one to call. Outputs stay small so a long
session does not fill the context with noise.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from beyond_vector_search.sandbox import TEXT_EXTENSIONS, Corpus
from beyond_vector_search.tree import load_corpus_tree, render_outline

ParamFn = Callable[..., str]


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    description: str
    required: bool = False


@dataclass
class Tool:
    """One callable the model may invoke, plus the schema both APIs share."""

    name: str
    description: str
    params: list[Param]
    fn: ParamFn

    @property
    def param_names(self) -> set[str]:
        return {p.name for p in self.params}

    def anthropic_schema(self) -> dict:
        properties = {
            p.name: {"type": p.type, "description": p.description}
            for p in self.params
        }
        schema: dict = {"type": "object", "properties": properties}
        required = [p.name for p in self.params if p.required]
        if required:
            schema["required"] = required
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    def openai_schema(self) -> dict:
        body = self.anthropic_schema()
        return {
            "type": "function",
            "name": body["name"],
            "description": body["description"],
            "parameters": body["input_schema"],
            "strict": False,
        }


class Toolset:
    """Tools bound to one corpus. Extra tools (semantic search, deep_search) register later."""

    def __init__(self, corpus: Corpus, tree_path: Path | None = None):
        self.corpus = corpus
        self.tree_path = tree_path
        self.extras: list[Tool] = []
        self.only: set[str] | None = None
        self._special: dict[str, Callable[..., tuple[str, list]]] = {}
        self._children: dict[str, list] = {}

    def definitions(self) -> list[Tool]:
        tools = [
            Tool(
                "list_files",
                "List text files under a folder with their line counts. "
                "Use first to see what documents exist.",
                [
                    Param("path", "string", "Folder relative to the docs root. Default '.'"),
                    Param("pattern", "string", "Glob, e.g. '**/*.md'. Default '**/*'"),
                ],
                self.list_files,
            ),
            Tool(
                "grep",
                "Regex search across files (ripgrep). Returns file:line:text with context. "
                "Best for exact terms, IDs, names, numbers and section references. "
                "If nothing matches, retry with synonyms or a word stem.",
                [
                    Param("pattern", "string", "Regular expression", required=True),
                    Param("path", "string", "File or folder to search. Default '.'"),
                    Param("ignore_case", "boolean", "Case-insensitive search. Default true"),
                    Param("context", "integer", "Lines of context around each hit"),
                    Param("max_matches", "integer", "Max hits per file. Default 20"),
                ],
                self.grep,
            ),
            Tool(
                "read_file",
                "Read a line range of a file (default 200 lines). Use after grep or "
                "outline to read the exact section you need, not the whole file.",
                [
                    Param("path", "string", "File relative to the docs root", required=True),
                    Param("start_line", "integer", "First line to read, 1-based"),
                    Param("end_line", "integer", "Last line to read, inclusive"),
                ],
                self.read_file,
            ),
            Tool(
                "outline",
                "Show the headings of a Markdown document with line ranges — "
                "a table of contents to navigate long documents.",
                [Param("path", "string", "Markdown file relative to the docs root", required=True)],
                self.outline,
            ),
            Tool(
                "corpus_tree",
                "Heading tree for the corpus: each file, its headings, and their line ranges. "
                "Use this first to see where a long document set is organised. "
                "The tree is saved on disk and reused until a file changes.",
                [Param("path", "string", "Folder or file relative to the docs root. Default '.'")],
                self.corpus_tree,
            ),
            *self.extras,
        ]
        if self.only is not None:
            return [tool for tool in tools if tool.name in self.only]
        return tools

    def register_special(self, tool: Tool, fn: Callable[..., tuple[str, list]]) -> None:
        """Register a tool that also returns a nested trace, keyed by call id."""
        self.extras.append(tool)
        self._special[tool.name] = fn

    def call(self, name: str, args: dict | None, call_id: str = "") -> str:
        args = args or {}
        try:
            tools = {tool.name: tool for tool in self.definitions()}
            if name not in tools and name not in self._special:
                return f"ERROR KeyError: unknown tool {name}"
            if name in self._special:
                allowed = tools[name].param_names if name in tools else set()
                kwargs = {k: v for k, v in args.items() if k in allowed and v is not None}
                output, children = self._special[name](call_id=call_id, **kwargs)
                self._children[call_id] = children
                return output
            tool = tools[name]
            kwargs = {k: v for k, v in args.items() if k in tool.param_names and v is not None}
            return tool.fn(**kwargs)
        except Exception as exc:
            return f"ERROR {type(exc).__name__}: {exc}"

    def take_children(self, call_id: str) -> list:
        return self._children.pop(call_id, [])

    def list_files(self, path: str = ".", pattern: str = "**/*") -> str:
        base = self.corpus.safe_path(path)
        if base.is_file():
            files = [base] if self.corpus.is_text(base) else []
        else:
            files = [
                item for item in sorted(base.glob(pattern))
                if item.is_file() and self.corpus.is_text(item)
            ]
        rows = []
        for item in files:
            rows.append(f"{self.corpus.relative(item)}  ({len(self.corpus.read_lines(item))} lines)")
        return self.corpus.truncate("\n".join(rows) or "No text files found.")

    def grep(
        self,
        pattern: str,
        path: str = ".",
        ignore_case: bool = True,
        context: int = 2,
        max_matches: int = 20,
        use_rg: bool | None = None,
    ) -> str:
        base = self.corpus.safe_path(path)
        ignore_case = _as_bool(ignore_case, default=True)
        context = _clamp(_as_int(context, 2), 0, 5)
        max_matches = _clamp(_as_int(max_matches, 20), 1, 50)
        prefer_rg = shutil.which("rg") is not None if use_rg is None else use_rg
        if prefer_rg and shutil.which("rg"):
            text = self._rg(pattern, base, ignore_case, context, max_matches)
        else:
            text = self._py_grep(pattern, base, ignore_case, context, max_matches)
        if not text.strip():
            text = f"No matches for {pattern!r}. Try a synonym or a shorter stem."
        return self.corpus.truncate(text)

    def _rg(self, pattern: str, base: Path, ignore_case: bool, context: int, max_matches: int) -> str:
        cmd = [
            "rg", "--line-number", "--no-heading", "--color=never",
            "-C", str(context), "-m", str(max_matches),
        ]
        if ignore_case:
            cmd.append("-i")
        for ext in sorted(TEXT_EXTENSIONS):
            cmd += ["-g", f"*{ext}"]
        cmd += ["-e", pattern, str(base)]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if completed.returncode not in (0, 1):
            detail = completed.stderr.strip() or f"exit {completed.returncode}"
            return f"ERROR rg: {detail}"
        out = completed.stdout.replace(str(self.corpus.root) + os.sep, "")
        return out.replace(str(self.corpus.root) + "/", "")

    def _py_grep(
        self,
        pattern: str,
        base: Path,
        ignore_case: bool,
        context: int,
        max_matches: int,
    ) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern, flags)
        files = [base] if base.is_file() else [
            item for item in base.rglob("*")
            if item.is_file() and self.corpus.is_text(item)
        ]
        blocks: list[str] = []
        for item in sorted(files):
            lines = self.corpus.read_lines(item)
            hits = 0
            for index, line in enumerate(lines):
                if not rx.search(line):
                    continue
                lo = max(0, index - context)
                hi = min(len(lines), index + context + 1)
                blocks.extend(
                    f"{self.corpus.relative(item)}:{line_no}:{lines[line_no - 1]}"
                    for line_no in range(lo + 1, hi + 1)
                )
                blocks.append("--")
                hits += 1
                if hits >= max_matches:
                    break
        return "\n".join(blocks)

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        target = self.corpus.safe_path(path)
        if target.is_dir():
            return f"ERROR IsADirectoryError: {path} is a folder; pass a file."
        lines = self.corpus.read_lines(target)
        start = _as_int(start_line, 1)
        if start < 1:
            return "ERROR ValueError: start_line must be >= 1."
        if start > len(lines):
            return f"{path} has only {len(lines)} lines."
        if end_line is None:
            end = start + 199
        else:
            end = _as_int(end_line, start + 199)
        if end < start:
            return "ERROR ValueError: end_line is before start_line."
        end = min(end, start + 399, len(lines))
        body = "\n".join(f"{i}\t{lines[i - 1]}" for i in range(start, end + 1))
        header = f"{self.corpus.relative(target)} - lines {start}-{end} of {len(lines)}"
        return self.corpus.truncate(f"{header}\n{body}")

    def corpus_tree(self, path: str = ".") -> str:
        try:
            rendered = load_corpus_tree(self.corpus, self.tree_path, path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return f"ERROR {type(exc).__name__}: {exc}"
        if not rendered:
            return "No headings found - use list_files and grep."
        return self.corpus.truncate(rendered)

    def outline(self, path: str) -> str:
        target = self.corpus.safe_path(path)
        if target.is_dir():
            return f"ERROR IsADirectoryError: {path} is a folder; pass a file."
        rendered = render_outline(self.corpus.read_lines(target))
        if not rendered:
            return "No headings found - use grep or read_file."
        header = f"{self.corpus.relative(target)}"
        return self.corpus.truncate(f"{header}\n{rendered}")


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if value is None:
        return default
    return bool(value)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
