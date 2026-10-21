# Beyond Vector Search

[![tests](https://github.com/sumitbopche01/beyond-vector-search/actions/workflows/test.yml/badge.svg)](https://github.com/sumitbopche01/beyond-vector-search/actions/workflows/test.yml)

Agentic document search in Python. The model opens a heading tree, greps, reads the lines it needs, and follows cross-references. Nothing is chunked or embedded unless you ask for the classic RAG baseline or the hybrid tools (BM25 and semantic search).

This is the code companion to [BEYOND_VECTOR_SEARCH.md](BEYOND_VECTOR_SEARCH.md).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'          # search, ingest, tests
# pip install -e '.[all]'        # also Claude, OpenAI, Chroma, and MCP
```

`ripgrep` is optional. Without it, grep falls back to Python.

```bash
brew install ripgrep
```

## Try it with no API key

The sample corpus is a small contract folder. The enterprise cap points at MSA §9.3, and §9.3 is the only place that lists the carve-outs.

```bash
bvs list --docs examples/corpus
bvs grep '2x annual' --docs examples/corpus
bvs grep '9\.3' --path contracts/msa_2024.md --docs examples/corpus
bvs outline contracts/msa_2024.md --docs examples/corpus
bvs tree --all --docs examples/corpus
```

`bvs tree --all` writes `.bvs/tree.md`: every heading in the corpus, with the line range of that section. `ask` reads that map first (`corpus_tree`) and rebuilds it when a file changes.

From this repo, `bvs` uses `examples/corpus` when `./docs` is missing and `DOCS_ROOT` is unset.

Convert a PDF, Word file, or HTML page before searching it. Headings stay headings, and each PDF page becomes a `## Page N` section. Column-aligned PDF tables and Word tables become Markdown tables. Scanned pages with no text layer are marked for OCR (`ocrmypdf`); this package does not shell out.

```bash
bvs ingest examples/raw ./docs
```

## Ask a question

```bash
pip install -e '.[anthropic]'   # or '.[openai]'
cp .env.example .env            # then put your key in that file
bvs ask "What is the enterprise liability cap, and are there exceptions?"
```

`bvs` reads `.env` from the current directory. Set `BVS_PROVIDER=openai` in that file to use OpenAI, or pass `--provider openai`.

The trace is the debugging tool. A wrong answer is five tool calls you can read, not a cosine score. Under the answer, the trace report lists which tools ran, any search that was repeated, and whether the loop stopped on the turn cap or the token budget.

`--deep` sends the searching to sub-agents and keeps their dead ends out of the main context. Independent sub-questions in one turn run together. Set `BVS_SEARCH_MODEL` to use a cheaper model for those calls.

## Classic RAG, the lexical index, and the hybrid

```bash
pip install -e '.[rag,openai]'    # or anthropic
bvs index --docs examples/corpus
bvs eval examples/questions.jsonl --docs examples/corpus \
  --systems agentic,rag,hybrid --out eval/report.md
bvs ask "How do we measure churn?" --docs examples/corpus --hybrid
```

`bvs index` chunks the corpus into `.chroma` and writes a BM25 index at `.bvs/lexical.json`. The hybrid agent gets both: `lexical_search` for names, IDs, and clause numbers, `semantic_search` for wording that does not match. Each hit is a path and a line range. `read_file` opens that range.

`bvs eval` runs every system on the same questions and prints each one as it starts: the model call, then each search, then pass or fail. That log goes to stderr, and the table stays on stdout. Keyword accuracy checks the required words. Cited checks that each `(file:line-range)` citation exists and that those lines contain the required facts. A system is allowed to lose. `--out` writes `eval/report.md`, `eval/report.jsonl` (the answers and the traces), and `eval/report.svg` (keyword accuracy and citation support, overall and by question type). `--judge` adds a model grade on top of those two checks.

```bash
bvs eval examples/questions.jsonl --docs examples/corpus --systems agentic,rag,hybrid --judge
```

The sample set is small on purpose. It is there to show the loop, including a synonym question (`churn` vs `customer attrition`) and a question whose answer is not in the documents. It is not a leaderboard.

## MCP

The same read-only tools are available over stdio, so Claude Desktop or Cursor can call them.

```bash
pip install -e '.[mcp]'
bvs mcp --docs examples/corpus
```

```json
{
  "mcpServers": {
    "beyond-vector-search": {
      "command": "bvs",
      "args": ["mcp", "--docs", "/absolute/path/to/docs"]
    }
  }
}
```

Tools: `list_files`, `grep`, `read_file`, `outline`, `corpus_tree`, `ask`. `ask` needs an API key in the environment. There is no shell tool.

## Layout

| Module | Role |
| --- | --- |
| `sandbox.py` | Resolves every path and refuses anything outside the document root |
| `tools.py` | `list_files`, `grep`, `read_file`, `outline`, `corpus_tree` |
| `tree.py` | Heading tree with line ranges, a saved corpus map, an optional one-pass summary |
| `providers/` | Claude Messages API and OpenAI Responses API |
| `agent.py` | The plan / act / observe loop |
| `subagent.py` | `deep_search`, a fresh context that returns excerpts only |
| `rag.py` | Fixed-chunk baseline, and the `semantic_search` tool |
| `lexical.py` | BM25 over those same chunks, and the `lexical_search` tool |
| `ingest.py` | PDF, DOCX, and HTML to Markdown, including tables |
| `citations.py` | Checks that cited lines exist and contain the required facts |
| `charts.py` | SVG comparison written next to an eval report |
| `evaluate.py` | Keyword score, citation score, optional judge, saved report |
| `mcp_server.py` | The same read-only tools over stdio MCP |
| `cli.py` | The `bvs` command |

`ask()` builds a `Corpus`, a `Toolset`, and a provider, then runs the loop. The toolset includes `corpus_tree`, the saved heading map. `--hybrid` adds `lexical_search` and `semantic_search`. `--deep` keeps `list_files`, `corpus_tree`, and `deep_search` on the orchestrator; each sub-agent gets the search tools in its own window. Tool errors are handed back to the model as text. There is no shell tool: an agent that reads untrusted documents stays read-only.

## Tests

```bash
pytest
```

The tests do not call a model. They cover the path jail, grep, the heading tree, ingest (including tables), chunk coordinates, BM25, the keyword and citation scorers, the saved eval report, the MCP handlers, and a scripted agent loop.

## Configuration

See `.env.example`. The usual ones:

| Variable | Default | Used for |
| --- | --- | --- |
| `DOCS_ROOT` | `./docs` | Folder the agent may read |
| `BVS_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Claude model id |
| `OPENAI_MODEL` | `gpt-5` | OpenAI model id |
| `BVS_SEARCH_MODEL` | unset | Cheaper model for sub-agents |
| `BVS_MAX_TURNS` | `25` | Hard stop for one question |
| `BVS_MAX_INPUT_TOKENS` | unset | Stop before the next search once input tokens pass this |
| `BVS_CHROMA_PATH` | `.chroma` | Where the vector index is stored |
| `BVS_LEXICAL_PATH` | `.bvs/lexical.json` | BM25 index. The heading tree is `tree.md` beside it |
