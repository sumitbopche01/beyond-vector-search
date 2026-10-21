"""BM25 over chunk line ranges. No embeddings."""

import json
from pathlib import Path

from beyond_vector_search.config import Settings
from beyond_vector_search.lexical import lexical_search_tool, tokenize


def test_hyphenated_ids_stay_searchable() -> None:
    assert "ts-999" in tokenize("See TS-999 for the tax engine.")
    assert "999" in tokenize("See TS-999 for the tax engine.")


def test_bm25_returns_the_file_and_the_line_range(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "errors.md").write_text(
        "# Errors\n\nTS-999 means the tax engine rejected the jurisdiction code.\n",
        encoding="utf-8",
    )
    (root / "picnic.md").write_text("# Picnic\n\nBring a blanket.\n", encoding="utf-8")
    settings = Settings(docs_root=root, lexical_path=tmp_path / "lexical.json")
    text = lexical_search_tool(settings).fn(query="TS-999", k=3)
    assert text.startswith("errors.md:")
    assert "picnic.md" not in text
    saved = json.loads(settings.lexical_path.read_text(encoding="utf-8"))
    assert saved["chunks"]
    assert saved["chunks"][0]["start_line"] >= 1


def test_a_newer_file_rebuilds_the_index(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "notes.md"
    doc.write_text("# Notes\n\nThe old clause.\n", encoding="utf-8")
    settings = Settings(docs_root=root, lexical_path=tmp_path / "lexical.json")
    first = lexical_search_tool(settings)
    assert "notes.md" in first.fn(query="clause")

    import os
    os.utime(settings.lexical_path, (1, 1))
    doc.write_text("# Notes\n\nA zebra clause was added.\n", encoding="utf-8")
    second = lexical_search_tool(settings)
    text = second.fn(query="zebra")
    assert "notes.md" in text
    assert "zebra" in text


def test_deleting_a_file_drops_it_from_the_index(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "keep.md").write_text("# Keep\n\nordinary notes\n", encoding="utf-8")
    gone = root / "gone.md"
    gone.write_text("# Gone\n\nA zebra clause.\n", encoding="utf-8")
    settings = Settings(docs_root=root, lexical_path=tmp_path / "lexical.json")
    assert "gone.md" in lexical_search_tool(settings).fn(query="zebra")
    gone.unlink()
    assert "gone.md" not in lexical_search_tool(settings).fn(query="zebra")


def test_stopwords_do_not_find_attrition_for_a_churn_question(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "examples" / "corpus"
    settings = Settings(docs_root=root, lexical_path=tmp_path / "lexical.json")
    text = lexical_search_tool(settings).fn(query="How do we measure churn?")
    assert "attrition" not in text.casefold()
