"""MCP handlers over the sample corpus. The SDK is optional."""

from pathlib import Path

import pytest

from beyond_vector_search.config import Settings
from beyond_vector_search.mcp_server import handlers

ROOT = Path(__file__).resolve().parents[1]


def test_handlers_grep_outline_and_the_corpus_tree(tmp_path: Path) -> None:
    settings = Settings(
        docs_root=ROOT / "examples" / "corpus",
        lexical_path=tmp_path / "lexical.json",
    )
    tools = handlers(settings)
    assert "enterprise_addendum.md" in tools["grep"]("2x annual")
    assert "9.3" in tools["outline"]("contracts/msa_2024.md")
    assert "Carve-outs" in tools["corpus_tree"](".")
    assert (tmp_path / "tree.md").is_file()


def test_mcp_server_registers_the_read_only_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from mcp.server.mcpserver import MCPServer

    from beyond_vector_search.mcp_server import register

    server = MCPServer("beyond-vector-search")
    register(server, Settings(
        docs_root=ROOT / "examples" / "corpus",
        lexical_path=tmp_path / "lexical.json",
    ))
    assert sorted(server._tool_manager._tools) == [
        "ask",
        "corpus_tree",
        "grep",
        "list_files",
        "outline",
        "read_file",
    ]
