"""Text cleaning, paragraph splitting, sentence splitting and chunking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from statistics import median


@dataclass
class Paragraph:
    idx: int        # 0-based position in the document
    label: str      # citation label: the judgment's own paragraph number if present, else "P<n>"
    text: str
    kind: str = "para"   # "para" or "header" (preamble before paragraph 1)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


# --------------------------------------------------------------------------- cleaning

_NOISE_LINES = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I),
    re.compile(r"^\s*\d{1,4}\s*$"),                       # lone page numbers
    re.compile(r"^\s*https?://\S+\s*$", re.I),
    re.compile(r"indian\s*kanoon|indiankanoon\.org", re.I),
    re.compile(r"^\s*(reportable|non-reportable)\s*$", re.I),
]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Normalise unicode, undo hyphenation across line breaks, drop page furniture."""
    text = unicodedata.normalize("NFKC", text).replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    kept = []
    for line in text.split("\n"):
        if any(p.search(line) for p in _NOISE_LINES):
            continue
        kept.append(re.sub(r"[ \t]+", " ", line).rstrip())
    return "\n".join(kept)


# --------------------------------------------------------------------------- paragraphs

_NUM = re.compile(r"^\s*(\d{1,3})\s*[.)](?:\s+(\S.*))?$")


def _find_numbered_starts(lines: list[str]) -> list[tuple[int, int]]:
    """Lines that open a numbered paragraph, accepting only a plausible 1,2,3... sequence."""
    starts, last = [], 0
    for i, ln in enumerate(lines):
        m = _NUM.match(ln)
        if not m:
            continue
        n = int(m.group(1))
        if last < n <= last + 3:
            starts.append((i, n))
            last = n
    return starts


def _merge_lines(lines: list[str]) -> list[str]:
    """For text with hard line wraps and no blank lines: a short line ending in a full stop closes a paragraph."""
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        return []
    typical = median(len(ln) for ln in lines)
    blocks, cur = [], []
    for ln in lines:
        cur.append(ln)
        if re.search(r"[.?!:;\"”]$", ln) and len(ln) < 0.7 * typical:
            blocks.append(" ".join(cur))
            cur = []
    if cur:
        blocks.append(" ".join(cur))
    return blocks


def split_paragraphs(text: str) -> list[Paragraph]:
    lines = text.split("\n")
    starts = _find_numbered_starts(lines)

    # Numbered judgment (the usual Indian Supreme Court / High Court layout)
    if len(starts) >= 5:
        paras: list[Paragraph] = []
        pre = normalize_ws(" ".join(lines[: starts[0][0]]))
        if pre:
            paras.append(Paragraph(0, "Header", pre, kind="header"))
        for k, (i, n) in enumerate(starts):
            end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
            first = _NUM.match(lines[i]).group(2) or ""
            body = normalize_ws(" ".join([first] + lines[i + 1 : end]))
            if body:
                paras.append(Paragraph(len(paras), str(n), body))
        return paras

    # Unnumbered: blank-line blocks, falling back to line merging for hard-wrapped text
    blocks = [normalize_ws(b) for b in re.split(r"\n\s*\n", text) if b.strip()]
    avg = sum(len(b.split()) for b in blocks) / max(1, len(blocks))
    if len(blocks) < 5 or avg > 300:
        blocks = [normalize_ws(b) for b in _merge_lines(lines)]
    blocks = [b for b in blocks if len(b.split()) >= 4]
    return [Paragraph(i, f"P{i + 1}", b) for i, b in enumerate(blocks)]


# --------------------------------------------------------------------------- sentences & chunks

_ABBR = (
    r"Mr|Mrs|Ms|Dr|Shri|Smt|Sri|No|Nos|Ors|Anr|Hon|St|Vol|Para|Paras|Cr|Crl|Cl|Sr|Jr|Prof|Ltd|Co|Rs|"
    r"Sec|Secs|Art|Arts|Dt|viz|vs|v|etc|Mt|Messrs|Addl|Dy|Regd|Govt|Dept|Inc|Maj|Col|Capt|Lt|Gen|Adv|Sh|Kum"
)
_ABBR_RE = re.compile(rf"\b({_ABBR})\.", re.I)
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s)")
_DOT = "․"  # placeholder so abbreviations are not treated as sentence ends


def split_sentences(text: str) -> list[str]:
    t = _ABBR_RE.sub(lambda m: m.group(1) + _DOT, text)
    t = _INITIAL_RE.sub(lambda m: m.group(1) + _DOT, t)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“(\[])", t)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 120) -> list[str]:
    """Group whole sentences into chunks of at most max_words (hard-splitting any longer sentence).

    Needed because sentence-embedding models truncate at ~256 tokens: a 400-word paragraph
    would otherwise be embedded from its first few lines only.
    """
    chunks, cur, n = [], [], 0
    for s in split_sentences(text):
        w = s.split()
        if len(w) > max_words:
            if cur:
                chunks.append(" ".join(cur))
                cur, n = [], 0
            chunks.extend(" ".join(w[i : i + max_words]) for i in range(0, len(w), max_words))
            continue
        if n + len(w) > max_words and cur:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += len(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]
