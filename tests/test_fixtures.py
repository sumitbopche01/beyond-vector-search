"""Fixture pack: search oracle and ingest oracle. No model calls.

The search corpus is what `bvs eval` is allowed to read. The ingest sketches
are the PDF cases, graded later by diffing extractor output against the
golden Markdown. Mixing those two scores hides which part failed.
"""

from pathlib import Path

from beyond_vector_search.citations import score_citations
from beyond_vector_search.evaluate import load_questions, score_keywords
from beyond_vector_search.sandbox import Corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = ROOT / "examples" / "fixtures" / "corpus"
INGEST_ROOT = ROOT / "examples" / "fixtures" / "ingest"
CORPUS = Corpus(CORPUS_ROOT)

# True line holds every must_include term. Trap line holds the words a
# careless search would quote, and not the whole required set.
LINES = {
    "cross_ref": ("contracts/schedule_b.md", 5, "contracts/order_form.md", 5),
    "contradiction": ("policies/notice_amendment.md", 5, "policies/notice_master.md", 5),
    "decoy": ("product/quotas.md", 9, "product/quotas.md", 5),
}

PASSING = {
    "cross_ref": "The binding cap is 3x annual fees, reference CAP-441 (contracts/schedule_b.md:5).",
    "contradiction": "The 60-day clause is withdrawn. The notice period is 10 business days (policies/notice_amendment.md:5).",
    "decoy": "The issued storage quota code is QT-77 (product/quotas.md:9).",
    "unanswerable": "The documents do not contain the monthly rent for a Tokyo office.",
}

FAILING = {
    "cross_ref": "The cap is 1x annual fees (contracts/order_form.md:5).",
    "contradiction": "Either party may terminate on 60 days written notice (policies/notice_master.md:5).",
    "decoy": "The storage quota code is QT-10 (product/quotas.md:5).",
    "unanswerable": "The Tokyo office rent is 900000 yen (product/quotas.md:9).",
}


def _line(path: str, number: int) -> str:
    return CORPUS.read_lines(CORPUS.safe_path(path))[number - 1]


def test_search_questions_match_the_hand_score() -> None:
    questions = {item.type: item for item in load_questions(ROOT / "examples" / "fixtures" / "questions.jsonl")}
    assert set(questions) == {"cross_ref", "contradiction", "decoy", "unanswerable"}

    for kind, (true_path, true_line, trap_path, trap_line) in LINES.items():
        item = questions[kind]
        true = _line(true_path, true_line)
        trap = _line(trap_path, trap_line)
        assert all(term.lower() in true.lower() for term in item.must_include)
        assert not all(term.lower() in trap.lower() for term in item.must_include)

        passed = score_citations(PASSING[kind], item, CORPUS)
        assert score_keywords(PASSING[kind], item)
        assert passed.grounded and passed.resolved

        failed = score_citations(FAILING[kind], item, CORPUS)
        assert not score_keywords(FAILING[kind], item)
        assert not failed.grounded

    missing = questions["unanswerable"]
    assert score_keywords(PASSING["unanswerable"], missing)
    assert not score_keywords(FAILING["unanswerable"], missing)
    declined = score_citations(PASSING["unanswerable"], missing, CORPUS)
    assert declined.grounded


def test_search_corpus_does_not_contain_the_ingest_only_facts() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in CORPUS_ROOT.rglob("*.md"))
    assert "Tokyo" not in text
    assert "4419" not in text
    assert "$4 per unit" not in text


def test_ingest_oracle_keeps_each_pdf_failure_on_its_own() -> None:
    columns = (INGEST_ROOT / "columns.golden.md").read_text(encoding="utf-8")
    assert columns.index("paying customer") < columns.index("implementation partner")
    assert "## Client (left column)" in columns
    assert "## Client (right column)" in columns

    table = (INGEST_ROOT / "table.golden.md").read_text(encoding="utf-8").splitlines()
    growth = next(line for line in table if line.startswith("| Growth"))
    starter = next(line for line in table if line.startswith("| Starter"))
    assert "$4 per unit" in growth
    assert "$9 per unit" in starter
    assert "$4 per unit" not in starter

    scan = (INGEST_ROOT / "scan.golden.md").read_text(encoding="utf-8")
    layout = (INGEST_ROOT / "scan.layout.md").read_text(encoding="utf-8")
    assert "4419" in scan
    assert "no text layer" in layout
    assert "4419" in layout
