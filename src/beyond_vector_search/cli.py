"""cli.py — the bvs command.

Offline commands (list, grep, outline, tree, ingest) need no API key.
ask, index, and eval do.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from beyond_vector_search import __version__
from beyond_vector_search.config import Settings, load_dotenv
from beyond_vector_search.sandbox import Corpus
from beyond_vector_search.tools import Toolset


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--docs",
        type=Path,
        help="Document root. Default: DOCS_ROOT, then ./docs, then examples/corpus",
    )

    parser = argparse.ArgumentParser(
        prog="bvs",
        description="Agentic document search. A vector index is optional.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", parents=[common], help="Answer a question by searching the documents")
    ask.add_argument("question")
    ask.add_argument("--provider", choices=["anthropic", "openai"])
    ask.add_argument("--model", help="Override the model for the selected provider")
    ask.add_argument(
        "--hybrid",
        action="store_true",
        help="Also give the agent lexical_search (BM25) and semantic_search",
    )
    ask.add_argument("--deep", action="store_true", help="Delegate search to sub-agents")
    ask.add_argument("--quiet", action="store_true", help="Hide the tool trace")
    ask.set_defaults(func=cmd_ask)

    grep = sub.add_parser("grep", parents=[common], help="Regex search, no model")
    grep.add_argument("pattern")
    grep.add_argument("--path", default=".", help="File or folder inside the docs root")
    grep.add_argument("--case-sensitive", action="store_true")
    grep.set_defaults(func=cmd_grep)

    outline = sub.add_parser("outline", parents=[common], help="Heading tree for one document")
    outline.add_argument("path")
    outline.set_defaults(func=cmd_outline)

    tree = sub.add_parser("tree", parents=[common], help="Heading tree, optionally summarized")
    tree.add_argument("path", nargs="?", help="One file. Omit with --all for the whole corpus")
    tree.add_argument("--all", action="store_true", help="Heading tree for every file, saved under .bvs/")
    tree.add_argument("--summarize", action="store_true", help="One LLM pass that summarizes each heading")
    tree.add_argument("--write", type=Path, help="Also write the tree to this file")
    tree.add_argument("--provider", choices=["anthropic", "openai"])
    tree.add_argument("--model")
    tree.set_defaults(func=cmd_tree)

    listing = sub.add_parser("list", parents=[common], help="List text files")
    listing.add_argument("path", nargs="?", default=".")
    listing.set_defaults(func=cmd_list)

    ingest = sub.add_parser("ingest", help="Convert PDF, DOCX, and HTML into Markdown")
    ingest.add_argument("src", type=Path)
    ingest.add_argument("dst", type=Path)
    ingest.set_defaults(func=cmd_ingest)

    index = sub.add_parser("index", parents=[common], help="Chunk and embed the corpus for classic RAG")
    index.set_defaults(func=cmd_index)

    evaluate = sub.add_parser("eval", parents=[common], help="Compare systems on a questions.jsonl file")
    evaluate.add_argument("questions", type=Path)
    evaluate.add_argument(
        "--systems",
        default="agentic",
        help="Comma-separated: agentic, rag, hybrid, deep",
    )
    evaluate.add_argument("--judge", action="store_true", help="Also grade answers with the model")
    evaluate.add_argument("--k", type=int, default=5, help="Top-k for the classic RAG system")
    evaluate.add_argument("--hybrid", action="store_true", help="Give deep sub-agents semantic_search")
    evaluate.add_argument("--provider", choices=["anthropic", "openai"])
    evaluate.add_argument("--model")
    evaluate.add_argument(
        "--out",
        type=Path,
        help="Write the report here (Markdown) plus a .jsonl trace sidecar",
    )
    evaluate.set_defaults(func=cmd_eval)

    return parser


def cmd_ask(args: argparse.Namespace) -> int:
    from beyond_vector_search.agent import ask

    settings = resolve_settings(args)
    result = ask(
        args.question,
        verbose=not args.quiet,
        settings=settings,
        hybrid=args.hybrid,
        deep=args.deep,
    )
    from beyond_vector_search.display import colors_enabled, format_trace_report, render_answer

    print(render_answer(
        result.answer,
        result.stats_line(),
        enabled=colors_enabled(sys.stdout),
        trace_note=format_trace_report(result),
    ))
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    text = _toolset(args).grep(
        args.pattern,
        path=args.path,
        ignore_case=not args.case_sensitive,
    )
    print(text)
    return 1 if text.startswith("ERROR") else 0


def cmd_outline(args: argparse.Namespace) -> int:
    text = _toolset(args).outline(args.path)
    print(text)
    return 1 if text.startswith("ERROR") else 0


def cmd_list(args: argparse.Namespace) -> int:
    text = _toolset(args).list_files(args.path)
    print(text)
    return 1 if text.startswith("ERROR") else 0


def cmd_tree(args: argparse.Namespace) -> int:
    from beyond_vector_search.tree import apply_summaries, build_tree, load_corpus_tree, render_nodes

    corpus = _corpus(args)
    if args.all:
        if args.summarize:
            raise ValueError("--summarize applies to one file; pass a path and omit --all")
        settings = resolve_settings(args)
        text = load_corpus_tree(corpus, settings.tree_path, ".") or "No headings found - use grep or read_file."
        print(text)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(text + "\n", encoding="utf-8")
        return 0
    if not args.path:
        raise ValueError("pass a file, or --all for the whole corpus")
    target = corpus.safe_path(args.path)
    if target.is_dir():
        raise ValueError(f"{args.path} is a folder; pass a file")
    nodes = build_tree(corpus.read_lines(target))
    if args.summarize:
        if not nodes:
            print("No headings found - use grep or read_file.")
            return 0
        from beyond_vector_search.providers import make_provider
        from beyond_vector_search.tree import summary_request

        settings = resolve_settings(args)
        reply = make_provider(settings).complete_text(
            "You label sections of a document outline. Return only the JSON array you were asked for.",
            summary_request(nodes),
        )
        apply_summaries(nodes, reply.text)
    text = render_nodes(nodes) or "No headings found - use grep or read_file."
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from beyond_vector_search.ingest import ingest

    results = ingest(args.src, args.dst)
    for item in results:
        detail = f" ({item.detail})" if item.detail else ""
        target = item.target or "-"
        print(f"{item.status:<12} {item.source} -> {target}{detail}")
    errors = [item for item in results if item.status == "error"]
    return 1 if errors else 0


def cmd_index(args: argparse.Namespace) -> int:
    from beyond_vector_search.rag import build_index

    settings = resolve_settings(args)
    count = build_index(settings)
    print(f"Indexed {count} chunks into the vector index and {settings.lexical_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from beyond_vector_search.agent import ask
    from beyond_vector_search.evaluate import evaluate, load_questions, render_report, write_report
    from beyond_vector_search.rag import rag_answer

    settings = resolve_settings(args)
    if not settings.docs_root.is_dir():
        raise FileNotFoundError(
            f"docs root does not exist: {settings.docs_root}. "
            "Set DOCS_ROOT or pass --docs. The sample corpus is examples/corpus."
        )
    questions = load_questions(args.questions)
    names = [part.strip() for part in args.systems.split(",") if part.strip()]
    systems = {}
    for name in names:
        if name == "agentic":
            systems[name] = lambda question, bound=settings: ask(
                question, verbose=False, settings=bound,
            )
        elif name == "hybrid":
            systems[name] = lambda question, bound=settings: ask(
                question, verbose=False, settings=bound, hybrid=True,
            )
        elif name == "deep":
            systems[name] = lambda question, bound=settings, hybrid=args.hybrid: ask(
                question, verbose=False, settings=bound, deep=True, hybrid=hybrid,
            )
        elif name == "rag":
            systems[name] = lambda question, bound=settings, k=args.k: rag_answer(
                question, settings=bound, k=k,
            )
        else:
            raise ValueError(f"unknown system {name!r}; use agentic, rag, hybrid, or deep")

    judge = None
    if args.judge:
        from beyond_vector_search.evaluate import judge_answer
        from beyond_vector_search.providers import make_provider

        provider = make_provider(settings)

        def judge(item, answer, bound=provider):
            return judge_answer(lambda system, user: bound.complete_text(system, user).text, item, answer)

    corpus = Corpus(settings.docs_root, max_output_chars=settings.max_output_chars)
    rows = evaluate(questions, systems, judge=judge, corpus=corpus)
    print(render_report(rows))
    if args.out:
        written = write_report(args.out, rows, settings=settings)
        print(f"\nWrote {written}, {written.with_suffix('.jsonl')}, and {written.with_suffix('.svg')}")
    return 0


def resolve_settings(args: argparse.Namespace) -> Settings:
    load_dotenv()
    settings = Settings.from_env()
    changes: dict = {}
    docs = getattr(args, "docs", None)
    if docs is not None:
        changes["docs_root"] = docs
    elif "DOCS_ROOT" not in os.environ and not settings.docs_root.is_dir():
        sample = Path.cwd() / "examples" / "corpus"
        if sample.is_dir():
            changes["docs_root"] = sample

    provider = getattr(args, "provider", None)
    if provider:
        changes["provider"] = provider
    model = getattr(args, "model", None)
    if model:
        chosen = (provider or settings.provider).strip().lower()
        if chosen == "openai":
            changes["openai_model"] = model
        else:
            changes["claude_model"] = model
    if changes:
        settings = settings.using(**changes)
    return settings


def _corpus(args: argparse.Namespace) -> Corpus:
    settings = resolve_settings(args)
    if not settings.docs_root.is_dir():
        raise FileNotFoundError(
            f"docs root does not exist: {settings.docs_root}. "
            "Set DOCS_ROOT or pass --docs. The sample corpus is examples/corpus."
        )
    return Corpus(settings.docs_root, max_output_chars=settings.max_output_chars)


def _toolset(args: argparse.Namespace) -> Toolset:
    return Toolset(_corpus(args))
