"""Conversion to markdown, including a second run that skips fresh files."""

from pathlib import Path

import pytest

from beyond_vector_search.ingest import html_to_markdown, ingest

ROOT = Path(__file__).resolve().parents[1]


def test_html_keeps_headings_and_list_items(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "notice.html").write_text(
        (ROOT / "examples" / "raw" / "notice.html").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dest = tmp_path / "docs"
    first = ingest(source, dest)
    assert first[0].status == "wrote"
    text = (dest / "notice.md").read_text(encoding="utf-8")
    assert "# Visitor wifi" in text
    assert "Monday" in text
    assert "- Do not connect lab hardware" in text

    second = ingest(source, dest)
    assert second[0].status == "skipped"


def test_destination_inside_the_source_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    with pytest.raises(ValueError):
        ingest(source, source / "out")


def test_html_helper_on_a_fragment() -> None:
    rendered = html_to_markdown(ROOT / "examples" / "raw" / "notice.html")
    assert "guest network" in rendered


def test_blank_pdf_is_marked_for_ocr(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    from beyond_vector_search.ingest import pdf_to_markdown

    pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf.open("wb") as handle:
        writer.write(handle)

    text = pdf_to_markdown(pdf)
    assert "# scan" in text
    assert "## Page 1" in text
    assert "needs OCR" in text


def test_docx_headings_survive(tmp_path: Path) -> None:
    pytest.importorskip("docx")
    from docx import Document

    from beyond_vector_search.ingest import docx_to_markdown

    path = tmp_path / "addendum.docx"
    document = Document()
    document.add_heading("Liability cap", level=2)
    document.add_paragraph("The cap is 2x annual fees, except as set out in MSA §9.3.")
    document.save(path)

    text = docx_to_markdown(path)
    assert "## Liability cap" in text
    assert "2x annual fees" in text
