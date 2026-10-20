# Beyond Vector Search

Agentic document search in Python. The model lists files, greps, reads the lines it needs, and follows cross-references. Nothing is chunked or embedded unless you ask for the classic RAG baseline or the hybrid semantic tool.

This is the code companion to [BEYOND_VECTOR_SEARCH.md](BEYOND_VECTOR_SEARCH.md).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'          # search, ingest, tests
# pip install -e '.[all]'        # also Claude, OpenAI, and Chroma
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
bvs tree contracts/msa_2024.md --docs examples/corpus
```

From this repo, `bvs` uses `examples/corpus` when `./docs` is missing and `DOCS_ROOT` is unset.

Convert a PDF, Word file, or HTML page before searching it. Headings stay headings, and each PDF page becomes a `## Page N` section. Scanned pages with no text layer are marked for OCR (`ocrmypdf`); this package does not shell out.

```bash
bvs ingest examples/raw ./docs
```

## Ask a question

```bash
pip install -e '.[anthropic]'   # or '.[openai]'
export ANTHROPIC_API_KEY=sk-ant-...
export DOCS_ROOT=./examples/corpus
bvs ask "What is the enterprise liability cap, and are there exceptions?"
```

OpenAI uses the same tools:

```bash
export OPENAI_API_KEY=sk-...
bvs ask "What is the enterprise liability cap, and are there exceptions?" --provider openai
```

The trace is the debugging tool. A wrong answer is five tool calls you can read, not a cosine score.

`--deep` sends the searching to sub-agents and keeps their dead ends out of the main context. Independent sub-questions in one turn run together. Set `BVS_SEARCH_MODEL` to use a cheaper model for those calls.

## Classic RAG, and the hybrid

```bash
pip install -e '.[rag,anthropic]'
bvs index --docs examples/corpus
bvs eval examples/questions.jsonl --docs examples/corpus --systems agentic,rag
bvs ask "How do we measure churn?" --docs examples/corpus --hybrid
```

`bvs index` chunks the corpus and stores it in `.chroma`. The hybrid agent still reads the real file: `semantic_search` returns a path and a line range, then `read_file` opens that range.

`bvs eval` scores answers with `must_include` terms, splits the result by question type, and can add a model judge:

```bash
bvs eval examples/questions.jsonl --docs examples/corpus --systems agentic,rag,hybrid --judge
```

The sample set is small on purpose. It is there to show the loop, including a synonym question (`churn` vs `customer attrition`) and a question whose answer is not in the documents. It is not a leaderboard.

## Layout

| Module | Role |
| --- | --- |
| `sandbox.py` | Resolves every path and refuses anything outside the document root |
| `tools.py` | `list_files`, `grep`, `read_file`, `outline` |
| `tree.py` | Heading tree with line ranges, plus an optional one-pass summary |
| `providers/` | Claude Messages API and OpenAI Responses API |
| `agent.py` | The plan / act / observe loop |
| `subagent.py` | `deep_search`, a fresh context that returns excerpts only |
| `rag.py` | Fixed-chunk baseline, and the `semantic_search` tool |
| `ingest.py` | PDF, DOCX, and HTML to Markdown |
| `evaluate.py` | Keyword score, optional judge, per-type report |
| `cli.py` | The `bvs` command |

`ask()` builds a `Corpus`, a `Toolset`, and a provider, then runs the loop. `--hybrid` adds `semantic_search`. `--deep` keeps `list_files` and `deep_search` on the orchestrator; each sub-agent gets the search tools in its own window. Tool errors are handed back to the model as text. There is no shell tool: an agent that reads untrusted documents stays read-only.

## Tests

```bash
pytest
```

The tests do not call a model. They cover the path jail, grep, the heading tree, ingest, chunk coordinates, the scorer, and a scripted agent loop.

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
