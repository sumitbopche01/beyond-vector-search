"""mcp_server.py — stdio MCP server for the same read-only tools.

Claude Desktop and Cursor call list_files, grep, read_file, outline,
corpus_tree, and ask. There is no shell tool. ask() still needs an API key
in the environment; the other tools do not.
"""

from __future__ import annotations

from collections.abc import Callable

from beyond_vector_search.config import Settings
from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Toolset


def handlers(settings: Settings | None = None) -> dict[str, Callable]:
    """Plain functions behind the MCP tools. Tests call these without the SDK."""
    settings = settings or Settings.from_env()

    def toolset() -> Toolset:
        if not settings.docs_root.is_dir():
            raise FileNotFoundError(
                f"docs root does not exist: {settings.docs_root}. Set DOCS_ROOT or pass --docs."
            )
        return Toolset(
            Corpus(settings.docs_root, max_output_chars=settings.max_output_chars),
            tree_path=settings.tree_path,
        )

    def list_files(path: str = ".", pattern: str = "**/*") -> str:
        return toolset().list_files(path=path, pattern=pattern)

    def grep(pattern: str, path: str = ".", ignore_case: bool = True) -> str:
        return toolset().grep(pattern, path=path, ignore_case=ignore_case)

    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        return toolset().read_file(path, start_line=start_line, end_line=end_line)

    def outline(path: str) -> str:
        return toolset().outline(path)

    def corpus_tree(path: str = ".") -> str:
        return toolset().corpus_tree(path=path)

    def ask(question: str, hybrid: bool = False, deep: bool = False) -> str:
        from beyond_vector_search.agent import ask as ask_question

        result = ask_question(
            question,
            verbose=False,
            settings=settings,
            hybrid=hybrid,
            deep=deep,
        )
        return f"{result.answer}\n\n{result.stats_line()}"

    return {
        "list_files": list_files,
        "grep": grep,
        "read_file": read_file,
        "outline": outline,
        "corpus_tree": corpus_tree,
        "ask": ask,
    }


def register(server, settings: Settings | None = None) -> None:
    """Attach the tools to a FastMCP server."""
    fns = handlers(settings)

    @server.tool()
    def list_files(path: str = ".", pattern: str = "**/*") -> str:
        """List text files under a folder with their line counts."""
        return fns["list_files"](path=path, pattern=pattern)

    @server.tool()
    def grep(pattern: str, path: str = ".", ignore_case: bool = True) -> str:
        """Regex search across the document root. Returns file:line:text."""
        return fns["grep"](pattern, path=path, ignore_case=ignore_case)

    @server.tool()
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a line range of one file. Lines are 1-based."""
        return fns["read_file"](path, start_line=start_line, end_line=end_line)

    @server.tool()
    def outline(path: str) -> str:
        """Heading tree for one Markdown file, with line ranges."""
        return fns["outline"](path)

    @server.tool()
    def corpus_tree(path: str = ".") -> str:
        """Heading tree for the corpus. Saved on disk and reused until a file changes."""
        return fns["corpus_tree"](path=path)

    @server.tool()
    def ask(question: str, hybrid: bool = False, deep: bool = False) -> str:
        """Search the documents and answer with file:line-range citations."""
        return fns["ask"](question, hybrid=hybrid, deep=deep)


def serve(settings: Settings | None = None) -> None:
    """Block on stdio until the client disconnects."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise ImportError(
            "Install the MCP extra: pip install 'beyond-vector-search[mcp]'"
        ) from exc
    server = MCPServer("beyond-vector-search")
    register(server, settings)
    server.run(transport="stdio")
