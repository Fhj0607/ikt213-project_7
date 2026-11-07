"""Shared document generation for the desktop application and CLI."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from llm.summarizer import LocalSummarizer
from output.document import writeDocument


EXPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "documents"


def createDocument(pages, destination=None):
    pages = list(pages)
    summary = LocalSummarizer().summarize(pages)
    if destination is None:
        destination = EXPORT_DIR / f"snapprint-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.docx"
    path = writeDocument(summary, pages, destination)
    return summary, path
