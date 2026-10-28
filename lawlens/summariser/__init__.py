"""Module 1: structured, paragraph-cited summaries of Indian court judgments."""

from .ingest import load_document, extract_pdf_text
from .pipeline import summarise, Summary, Point, verify_citations
from .roles import ROLES, ROLE_TITLES

__all__ = [
    "load_document",
    "extract_pdf_text",
    "summarise",
    "Summary",
    "Point",
    "verify_citations",
    "ROLES",
    "ROLE_TITLES",
]
