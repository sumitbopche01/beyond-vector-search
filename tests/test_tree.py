"""Heading trees: line ranges nest, summaries merge, and the corpus map is saved."""

import os
from pathlib import Path

from beyond_vector_search.cli import main
from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Toolset
from beyond_vector_search.tree import apply_summaries, build_tree, render_nodes

ROOT = Path(__file__).resolve().parents[1]


def test_sections_run_until_the_next_sibling() -> None:
    lines = ["# A", "intro", "## B", "body", "# C", "end"]
    nodes = build_tree(lines)

    assert [node.title for node in nodes] == ["A", "C"]
    assert nodes[0].start_line == 1
    assert nodes[0].end_line == 4
    assert nodes[0].children[0].title == "B"
    assert nodes[0].children[0].end_line == 4
    assert nodes[1].start_line == 5
    assert nodes[1].end_line == 6


def test_summaries_accept_a_fenced_json_payload() -> None:
    nodes = build_tree(["# A", "body"])
    apply_summaries(nodes, '```json\n[{"start_line": 1, "summary": "The top of the doc"}]\n```')
    rendered = render_nodes(nodes)
    assert "The top of the doc" in rendered
    assert "1-2" in rendered


def test_corpus_tree_is_saved_and_refreshed_when_a_file_changes(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "msa.md"
    doc.write_text("# Liability\n\n### 9.3 Carve-outs\n\nconfidentiality\n", encoding="utf-8")
    cache = tmp_path / "tree.md"
    toolset = Toolset(Corpus(root), tree_path=cache)
    first = toolset.corpus_tree()
    assert "9.3 Carve-outs" in first
    assert "9.3 Carve-outs" in cache.read_text(encoding="utf-8")

    cache.write_text("STALE TREE\n", encoding="utf-8")
    os.utime(cache, (cache.stat().st_mtime + 10, cache.stat().st_mtime + 10))
    assert toolset.corpus_tree() == "STALE TREE"

    doc.write_text("# Liability\n\n## Fresh heading\n", encoding="utf-8")
    os.utime(doc, (doc.stat().st_mtime + 5, doc.stat().st_mtime + 5))
    assert "Fresh heading" in toolset.corpus_tree()


def test_deleting_a_file_drops_it_from_the_saved_tree(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    keep = root / "keep.md"
    gone = root / "gone.md"
    keep.write_text("# Keep\n", encoding="utf-8")
    gone.write_text("# Gone heading\n", encoding="utf-8")
    toolset = Toolset(Corpus(root), tree_path=tmp_path / "tree.md")
    assert "Gone heading" in toolset.corpus_tree()
    gone.unlink()
    assert "Gone heading" not in toolset.corpus_tree()


def test_tree_all_prints_the_sample_corpus(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("BVS_LEXICAL_PATH", str(tmp_path / "lexical.json"))
    code = main(["tree", "--all", "--docs", str(ROOT / "examples" / "corpus")])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "9.3" in captured.out
    assert (tmp_path / "tree.md").is_file()
