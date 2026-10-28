"""Read a judgment (PDF or plain text) into numbered paragraphs."""

from __future__ import annotations

import io

from .textutils import Paragraph, clean_text, split_paragraphs


def extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if len(text.split()) < 150:
        raise ValueError(
            "Very little text could be extracted. This PDF is probably a scan; "
            "LawLens does not do OCR. Use a text-based PDF or paste the text."
        )
    return text


def load_document(filename: str, data: bytes) -> list[Paragraph]:
    if filename.lower().endswith(".pdf"):
        raw = extract_pdf_text(data)
    else:
        raw = data.decode("utf-8", errors="ignore")
    paras = split_paragraphs(clean_text(raw))
    if not paras:
        raise ValueError("No paragraphs could be found in this document.")
    return paras


def load_text(text: str) -> list[Paragraph]:
    paras = split_paragraphs(clean_text(text))
    if not paras:
        raise ValueError("No paragraphs could be found in this text.")
    return paras
