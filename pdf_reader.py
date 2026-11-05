"""
PDF text extraction using PyMuPDF.

- Pages are collected in a list and joined once
- Context manager guarantees the document is closed
- Clear errors for empty, corrupt, encrypted and scanned (image-only) PDFs
- Page limit and early stop once enough text is collected
- Conservative text cleanup that keeps headings, lists and paragraphs intact
"""

import re
import logging

import pymupdf

logger = logging.getLogger("pdf_reader")

MAX_PAGES = 200            # safety cap for very large PDFs
MAX_CHARS = 60_000         # stop early once we have this much text
MIN_TEXT_CHARS = 20        # below this we assume the PDF is scanned / image-only

_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
}
_LIGATURE_RE = re.compile("|".join(_LIGATURES))


class PDFExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF."""


def _read_bytes(uploaded_file) -> bytes:
    """Support Streamlit UploadedFile, Flask FileStorage, file objects and raw bytes."""
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)
    if hasattr(uploaded_file, "getvalue"):      # Streamlit
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):          # Flask / regular file
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return data
    raise PDFExtractionError("Unsupported file object.")


def _clean_text(text: str) -> str:
    # Ligature glyphs (fi, fl, ...) -> plain letters; drop soft hyphens / null chars
    text = _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group()], text)
    text = text.replace("\u00ad", "").replace("\x00", "")
    text = text.replace("\xa0", " ")

    # Rejoin words split across lines: "learn-\ning" -> "learning".
    # Only when both sides are lowercase, so "well-\nKnown" style cases are untouched.
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)

    # Join a line break only when the sentence clearly continues:
    # previous line doesn't end in . ! ? : ; and the next line starts lowercase.
    # Headings, bullets, numbered items and new sentences keep their line breaks.
    text = re.sub(r"(?<=[^\s.!?:;])[ \t]*\n(?=[a-z])", " ", text)

    # Collapse runs of spaces/tabs and trailing spaces before newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    # Collapse 3+ newlines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    """Cut at `limit` characters without splitting the last word."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return cut[:space] if space > limit * 0.9 else cut


def extract_text(uploaded_file, max_pages: int = MAX_PAGES,
                 max_chars: int = MAX_CHARS, clean: bool = True,
                 sort: bool = True) -> str:
    """
    Extract text from an uploaded PDF and return it as a string.

    sort=True returns text in natural reading order (better for multi-column
    papers); set sort=False for slightly faster extraction.
    """
    data = _read_bytes(uploaded_file)
    if not data:
        raise PDFExtractionError("The uploaded file is empty.")

    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:

            if document.needs_pass:
                raise PDFExtractionError("This PDF is password-protected.")

            total_pages = document.page_count
            if total_pages == 0:
                raise PDFExtractionError("This PDF has no pages.")

            if total_pages > max_pages:
                logger.warning("PDF has %d pages; only reading the first %d.",
                               total_pages, max_pages)

            parts = []
            char_count = 0

            for page_number in range(min(total_pages, max_pages)):
                page_text = str(document[page_number].get_text("text", sort=sort))
                if page_text.strip():
                    parts.append(page_text)
                    char_count += len(page_text)
                if char_count >= max_chars:
                    logger.info("Reached %d characters at page %d; stopping early.",
                                char_count, page_number + 1)
                    break

    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(f"PDF extraction failed: {exc}") from exc

    content = "\n\n".join(parts).strip()

    if len(content) < MIN_TEXT_CHARS:
        raise PDFExtractionError(
            "No readable text found. This PDF may be scanned or image-based "
            "and would need OCR."
        )

    if clean:
        content = _clean_text(content)

    return _truncate(content, max_chars)