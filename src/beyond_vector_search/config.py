"""config.py — runtime settings.

Reads the environment variables from the article (DOCS_ROOT, CLAUDE_MODEL,
OPENAI_MODEL) plus BVS_PROVIDER, which selects the backend for ask().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _optional_int(raw: str | None) -> int | None:
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def load_dotenv(path: Path | None = None) -> None:
    """Read a .env file into the process. Variables already set in the shell win."""
    env_path = path or (Path.cwd() / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one run. Frozen so sub-agents can share it."""

    docs_root: Path = Path("./docs")
    provider: str = "anthropic"
    claude_model: str = "claude-sonnet-5-5"
    openai_model: str = "gpt-5"
    search_model: str | None = None
    max_turns: int = 25
    max_output_chars: int = 12_000
    max_input_tokens: int | None = None
    chroma_path: Path = Path(".chroma")
    lexical_path: Path = Path(".bvs/lexical.json")
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(self, "docs_root", Path(self.docs_root).expanduser().resolve())
        object.__setattr__(self, "chroma_path", Path(self.chroma_path).expanduser().resolve())
        object.__setattr__(self, "lexical_path", Path(self.lexical_path).expanduser().resolve())
        object.__setattr__(self, "provider", _normalize_provider(self.provider))
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            docs_root=Path(os.environ.get("DOCS_ROOT", "./docs")),
            provider=os.environ.get("BVS_PROVIDER", "anthropic"),
            claude_model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-5"),
            search_model=os.environ.get("BVS_SEARCH_MODEL") or None,
            max_turns=int(os.environ.get("BVS_MAX_TURNS", "25")),
            max_output_chars=int(os.environ.get("BVS_MAX_OUTPUT_CHARS", "12000")),
            max_input_tokens=_optional_int(os.environ.get("BVS_MAX_INPUT_TOKENS")),
            chroma_path=Path(os.environ.get("BVS_CHROMA_PATH", ".chroma")),
            lexical_path=Path(os.environ.get("BVS_LEXICAL_PATH", ".bvs/lexical.json")),
        )

    @property
    def tree_path(self) -> Path:
        """Saved heading tree, next to the lexical index."""
        return self.lexical_path.with_name("tree.md")

    def using(self, **changes: object) -> Settings:
        return replace(self, **changes)

    @property
    def model(self) -> str:
        if self.provider == "openai":
            return self.openai_model
        return self.claude_model

    def for_search_agent(self) -> Settings:
        """Sub-agents can run a cheaper model without changing the orchestrator."""
        if not self.search_model:
            return self
        if self.provider == "openai":
            return self.using(openai_model=self.search_model)
        return self.using(claude_model=self.search_model)


def _normalize_provider(name: str) -> str:
    key = name.strip().lower()
    if key in {"anthropic", "claude"}:
        return "anthropic"
    if key == "openai":
        return "openai"
    raise ValueError(f"unknown provider {name!r}; use 'anthropic' or 'openai'")
