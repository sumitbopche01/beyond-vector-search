"""Offline checks for the path jail, the four tools, and the sample corpus."""

from pathlib import Path

from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Toolset

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "corpus"


def test_rejects_paths_outside_the_corpus(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (tmp_path / ".env").write_text("API_KEY=secret", encoding="utf-8")
    toolset = Toolset(Corpus(root))

    escaped = toolset.call("read_file", {"path": "../.env"})
    assert escaped.startswith("ERROR")
    assert "escapes" in escaped
    assert "secret" not in escaped


def test_grep_and_outline_follow_the_contract_cross_reference() -> None:
    toolset = Toolset(Corpus(CORPUS))
    cap = toolset.grep("2x annual", use_rg=False)
    carve = toolset.grep(r"9\.3", path="contracts/msa_2024.md", use_rg=False)
    outline = toolset.outline("contracts/msa_2024.md")

    assert "enterprise_addendum.md" in cap
    assert "msa_2024.md" in carve
    assert "### 9.3 Carve-outs" in outline
    assert "confidentiality" not in (CORPUS / "contracts/enterprise_addendum.md").read_text().lower()


def test_sample_corpus_keeps_the_trap_questions_honest() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in CORPUS.rglob("*.md"))
    lowered = text.lower()
    assert "churn" not in lowered
    assert "tokyo" not in lowered
    assert "customer attrition" in lowered
    assert lowered.count("14") == 1


def test_read_file_window_and_truncation(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.md").write_text("\n".join(f"line {i}" for i in range(1, 30)), encoding="utf-8")
    toolset = Toolset(Corpus(root, max_output_chars=80))

    window = toolset.read_file("notes.md", start_line=3, end_line=5)
    assert "3\tline 3" in window
    assert "line 6" not in window

    dumped = toolset.read_file("notes.md")
    assert "truncated" in dumped


def test_bad_regex_is_returned_to_the_caller(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("hello", encoding="utf-8")
    message = Toolset(Corpus(root)).call("grep", {"pattern": "["})
    assert message.startswith("ERROR")
