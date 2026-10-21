"""Citation checks against the sample corpus. No model calls."""

from pathlib import Path

from beyond_vector_search.citations import find_citations, score_citations
from beyond_vector_search.evaluate import Question
from beyond_vector_search.sandbox import Corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Corpus(ROOT / "examples" / "corpus")
CAP = Question(
    "What is the enterprise liability cap, and are there exceptions?",
    ["2x", "9.3", "confidentiality"],
    "multi_hop",
)


def test_citations_parse_with_and_without_parentheses() -> None:
    text = (
        "Cap is 2x (contracts/enterprise_addendum.md:11). "
        "See contracts/msa_2024.md:31-37 for the carve-outs."
    )
    found = find_citations(text)
    assert [(item.path, item.start, item.end) for item in found] == [
        ("contracts/enterprise_addendum.md", 11, 11),
        ("contracts/msa_2024.md", 31, 37),
    ]


def test_grounded_when_the_cited_lines_contain_the_facts() -> None:
    answer = (
        "The cap is 2x annual fees (contracts/enterprise_addendum.md:11). "
        "Section 9.3 excludes confidentiality (contracts/msa_2024.md:31-37)."
    )
    score = score_citations(answer, CAP, CORPUS)
    assert score.resolved
    assert score.grounded
    assert score.missing == ()


def test_right_words_in_the_answer_still_fail_without_a_citation() -> None:
    answer = "The cap is 2x annual fees, except confidentiality under section 9.3."
    score = score_citations(answer, CAP, CORPUS)
    assert not score.grounded
    assert score.note == "no citations"


def test_a_real_file_with_the_wrong_lines_is_not_grounded() -> None:
    answer = "The cap is 2x (contracts/enterprise_addendum.md:1) and 9.3 confidentiality."
    score = score_citations(answer, CAP, CORPUS)
    assert score.resolved
    assert not score.grounded
    assert "9.3" in score.missing


def test_a_missing_file_is_a_bad_citation() -> None:
    answer = "Not in the documents (tokyo.md:1)."
    item = Question("Tokyo on-call?", [], "unanswerable")
    score = score_citations(answer, item, CORPUS)
    assert not score.resolved
    assert not score.grounded
    assert "bad citation" in score.note


def test_a_bare_filename_is_accepted_when_it_is_unique() -> None:
    answer = "TS-999 means the jurisdiction code was rejected (errors.md:5)."
    item = Question("What does TS-999 mean?", ["TS-999", "jurisdiction"], "lookup")
    score = score_citations(answer, item, CORPUS)
    assert score.grounded


def test_a_wrong_directory_does_not_bind_to_another_file() -> None:
    answer = "TS-999 (other/errors.md:5)."
    item = Question("What does TS-999 mean?", ["TS-999"], "lookup")
    score = score_citations(answer, item, CORPUS)
    assert not score.resolved
    assert "bad citation" in score.note


def test_parent_segments_are_rejected() -> None:
    answer = "The cap is 2x (../contracts/enterprise_addendum.md:11)."
    score = score_citations(answer, CAP, CORPUS)
    assert not score.resolved


def test_a_line_past_the_end_of_the_file_is_rejected() -> None:
    answer = "The cap is 2x (contracts/enterprise_addendum.md:1-9999)."
    score = score_citations(answer, CAP, CORPUS)
    assert not score.resolved


def test_duplicate_citations_are_counted_once() -> None:
    answer = (
        "2x (contracts/enterprise_addendum.md:11) and again "
        "(contracts/enterprise_addendum.md:11)."
    )
    found = find_citations(answer)
    assert len(found) == 1
