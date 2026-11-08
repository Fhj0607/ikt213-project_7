"""A fixed Word template; the model supplies content, never document styling."""
import re
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt

from llm.summarizer import validateSummary


HEADINGS = ("Summary", "Key points", "Detailed notes", "Unclear passages", "Original OCR text")


def cleanText(text):
    # XML 1.0 cannot store control characters occasionally emitted by OCR.
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "", text)


def writeDocument(summary, pages, destination):
    validateSummary(summary)
    document = Document()
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2)
    section.left_margin = section.right_margin = Cm(2.5)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size in (("Title", 24), ("Heading 1", 16), ("Heading 2", 12)):
        document.styles[name].font.name = "Calibri"
        document.styles[name].font.size = Pt(size)
    document.add_heading(cleanText(summary["title"]), 0)
    document.add_paragraph(f"SnapPrint | {len(pages)} source page(s)", "Subtitle")
    document.add_paragraph("Generated notes for review. The original OCR text is included below.")
    document.add_heading(HEADINGS[0], 1)
    for paragraph in summary["summary"].split("\n\n"):
        document.add_paragraph(cleanText(paragraph))
    document.add_heading(HEADINGS[1], 1)
    for point in summary["key_points"]:
        document.add_paragraph(cleanText(point), "List Bullet")
    if not summary["key_points"]:
        document.add_paragraph("No key points provided.")
    document.add_heading(HEADINGS[2], 1)
    for topic in summary["sections"]:
        document.add_heading(cleanText(topic["heading"]), 2)
        for paragraph in topic["paragraphs"]:
            document.add_paragraph(cleanText(paragraph))
    if not summary["sections"]:
        document.add_paragraph("No detailed notes provided.")
    document.add_heading(HEADINGS[3], 1)
    for passage in summary["unclear_passages"]:
        document.add_paragraph(cleanText(passage), "List Bullet")
    if not summary["unclear_passages"]:
        document.add_paragraph("No unclear passages flagged by the model. OCR accuracy has not been verified.")
    document.add_page_break()
    document.add_heading(HEADINGS[4], 1)
    for number, page in enumerate(pages, 1):
        document.add_heading(f"Page {number}", 2)
        document.add_paragraph(cleanText(page))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.save(destination)
    return destination.resolve()
