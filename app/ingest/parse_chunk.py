"""Structure-aware ingestion for IRS publications.

Pipeline: extract each page's text + tables (pdfplumber), clean it, split the
prose at heading boundaries, keep tables and worked "Example N." blocks whole,
then window only *within* a block so a chunk never straddles two topics.

Every knob (chunk size, overlap, minimum size, what to keep atomic, what to
drop) lives on `ChunkConfig`, populated from `app.config.Settings`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BARE_PAGENUM = re.compile(r"^\s*\d{1,4}\s*$")
_WS = re.compile(r"\s+")

# The draft-watermark block IRS stamps on every page of these PDFs. Length-
# bounded so a page missing the closing phrase can't consume real content.
_PDF_FURNITURE = re.compile(
    r"Page\s+\d+\s+of\s+\d+\s+Fileid:.{0,400}?MUST be removed before printing\.",
    re.I | re.S,
)
_INDEX_MARKERS = (
    "to help us develop a more useful index",
    "please let us know if you have ideas for index entries",
)
_INDEX_SECTIONS = {"index", "contents", "table of contents"}

_SENT_END = ".,;:!?"
# Lines that open an atomic block (kept whole): worked examples, worksheets, tables.
_ATOMIC_HEAD = re.compile(r"^(example\s+\d+[.:]|worksheet\b|table\s+\d+[.:]?)", re.I)

TokenLen = Callable[[str], int]

# bge-small truncates input past 512 tokens, so an "atomic" table/example larger
# than this is still windowed (the embedding would otherwise ignore its tail).
_EMBED_MAX_TOKENS = 480


def _words(s: str) -> int:
    return len(s.split())


@dataclass(frozen=True)
class ChunkConfig:
    pdf_extractor: str = "pdfplumber"
    target_tokens: int = 450
    overlap_tokens: int = 64
    min_tokens: int = 32
    heading_split: bool = True
    tables_atomic: bool = True
    keep_examples_atomic: bool = True
    drop_boilerplate: bool = True
    numeric_density: float = 0.28

    @classmethod
    def from_settings(cls, s) -> ChunkConfig:
        return cls(
            pdf_extractor=s.pdf_extractor,
            target_tokens=s.chunk_target_tokens,
            overlap_tokens=s.chunk_overlap_tokens,
            min_tokens=s.chunk_min_tokens,
            heading_split=s.chunk_heading_split,
            tables_atomic=s.chunk_tables_atomic,
            keep_examples_atomic=s.chunk_keep_examples_atomic,
            drop_boilerplate=s.index_drop_boilerplate,
            numeric_density=s.index_numeric_density,
        )


@dataclass
class Block:
    text: str
    section: str
    page: int
    block_type: str  # "prose" | "table" | "example"


# --------------------------------------------------------------------------- #
# text cleaning
# --------------------------------------------------------------------------- #
def clean_lines(raw: str) -> list[str]:
    raw = _HYPHEN_BREAK.sub(r"\1\2", raw)
    raw = _PDF_FURNITURE.sub(" ", raw)
    out: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or _BARE_PAGENUM.match(s):
            continue
        out.append(s)
    return out


def clean_text(raw: str) -> str:
    return _WS.sub(" ", " ".join(clean_lines(raw))).strip()


# --------------------------------------------------------------------------- #
# structure detection
# --------------------------------------------------------------------------- #
def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not (2 <= len(s) <= 80):
        return False
    if _ATOMIC_HEAD.match(s):
        return True
    if sum(c.isalpha() for c in s) < 3:
        return False
    if s[-1] in _SENT_END:
        return False
    words = s.split()
    if s.isupper():
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return len(words) <= 12 and caps >= max(1, len(words) - 1)


def detect_section(lines: list[str], fallback: str) -> str:
    for ln in lines:
        if looks_like_heading(ln):
            return ln.strip().rstrip(".:")
    return fallback


def is_low_value(text: str, section: str, cfg: ChunkConfig) -> bool:
    """Index, table-of-contents, or page-number-list pages carry no answers."""
    low = text.lower()
    if any(m in low for m in _INDEX_MARKERS):
        return True
    if section.strip().lower() in _INDEX_SECTIONS:
        return True
    toks = text.split()
    if len(toks) >= 40:
        numish = sum(1 for t in toks if t.strip(".,") .isdigit())
        if numish / len(toks) > cfg.numeric_density:
            return True
    return False


def serialize_table(rows: list[list[str | None]], *, section: str, pub: str, page: int) -> str:
    lines: list[str] = []
    for r in rows:
        cells = [_WS.sub(" ", (c or "")).strip() for c in r]
        if any(cells):
            lines.append(" | ".join(cells))
    if not lines:
        return ""
    return f"[Table - {section} ({pub} p.{page})]\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# segmentation + windowing
# --------------------------------------------------------------------------- #
def segment_text(lines: list[str], *, page: int, default_section: str,
                 cfg: ChunkConfig) -> list[Block]:
    blocks: list[Block] = []
    section = default_section
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = _WS.sub(" ", " ".join(buf)).strip()
        buf = []
        if not text:
            return
        btype = "example" if _ATOMIC_HEAD.match(text) else "prose"
        blocks.append(Block(text=text, section=section, page=page, block_type=btype))

    for ln in lines:
        if cfg.heading_split and looks_like_heading(ln):
            flush()
            section = ln.strip().rstrip(".:")
            buf.append(ln.strip())
        else:
            buf.append(ln)
    flush()
    return blocks


def _window(text: str, cfg: ChunkConfig, token_len: TokenLen) -> list[str]:
    if token_len(text) <= cfg.target_tokens:
        return [text]
    words = text.split()
    if not words:
        return []
    # measure this block's words-per-token once, then slide by words
    ratio = max(1.0, len(words) / max(1, token_len(text)))
    tgt = max(1, int(cfg.target_tokens * ratio))
    step = max(1, tgt - int(cfg.overlap_tokens * ratio))
    out: list[str] = []
    for start in range(0, len(words), step):
        win = words[start:start + tgt]
        if not win:
            break
        out.append(" ".join(win))
        if start + tgt >= len(words):
            break
    return out


def chunk_blocks(
    blocks: list[Block], meta: dict, cfg: ChunkConfig, token_len: TokenLen = _words
) -> list[dict]:
    chunks: list[dict] = []
    per_page: dict[int, int] = {}
    for b in blocks:
        atomic = (b.block_type == "table" and cfg.tables_atomic) or (
            b.block_type == "example" and cfg.keep_examples_atomic
        )
        if atomic and token_len(b.text) <= _EMBED_MAX_TOKENS:
            pieces = [b.text]
        elif atomic:
            # too large to embed whole: window it, re-attaching the table caption
            head = b.text.split("\n", 1)[0] if b.block_type == "table" else ""
            pieces = [w if not head or w.startswith(head) else f"{head}\n{w}"
                      for w in _window(b.text, cfg, token_len)]
        else:
            pieces = _window(b.text, cfg, token_len)
        for piece in pieces:
            if b.block_type == "prose" and token_len(piece) < cfg.min_tokens:
                continue
            idx = per_page.get(b.page, 0)
            per_page[b.page] = idx + 1
            chunks.append({
                "chunk_id": f"{meta['name']}-{b.page}-{idx}",
                "text": piece,
                "pub": meta["pub"],
                "title": meta["title"],
                "section": b.section,
                "page": b.page,
                "source_url": meta["source_url"],
                "tax_year": meta["tax_year"],
                "block_type": b.block_type,
            })
    return chunks


# --------------------------------------------------------------------------- #
# PDF I/O
# --------------------------------------------------------------------------- #
# IRS publications set body text in two or three columns. pdfplumber's plain
# extract_text() groups words into full-width lines, so a left-column line and
# the right-column line at the same height get spliced together ("THEN file a
# return Standard Deductiongives the rules"). We locate the column gutters
# (vertical bands almost no word box crosses) and read each column top-to-bottom.
_SCAN_MARGIN = 0.12         # ignore the outer 12% of width when hunting gutters
_GUTTER_WIN_PT = 18         # a gutter must be the local straddle-minimum over +-this
_MERGE_PT = 26             # gutters closer than this are one gutter
_COL_MIN_SHARE = 0.08      # a column band holding fewer words than this is spurious


def _column_gutters(words: list[dict], width: float) -> list[float]:
    n = len(words)
    spans = [(float(wd["x0"]), float(wd["x1"])) for wd in words]
    xs = list(range(int(width * _SCAN_MARGIN), int(width * (1 - _SCAN_MARGIN)), 2))
    straddle = [sum(1 for a, b in spans if a < x < b) for x in xs]
    thresh = max(2, int(n * 0.012))
    win = max(1, _GUTTER_WIN_PT // 2)

    raw: list[float] = []
    for i, x in enumerate(xs):
        if straddle[i] > thresh:
            continue
        if straddle[i] == min(straddle[max(0, i - win):i + win + 1]):
            raw.append(float(x))

    merged: list[list[float]] = []
    for x in raw:
        if merged and x - merged[-1][-1] <= _MERGE_PT:
            merged[-1].append(x)
        else:
            merged.append([x])
    gutters = [sum(g) / len(g) for g in merged]

    # drop a gutter that would carve off a band with too few words
    bounds = [0.0, *gutters, width]
    keep: list[float] = []
    for gi, g in enumerate(gutters):
        left_band = (bounds[gi], g)
        right_band = (g, bounds[gi + 2])
        lc = sum(1 for a, b in spans if left_band[0] <= (a + b) / 2 < left_band[1])
        rc = sum(1 for a, b in spans if right_band[0] <= (a + b) / 2 < right_band[1])
        if lc >= n * _COL_MIN_SHARE and rc >= n * _COL_MIN_SHARE:
            keep.append(g)
    return keep


def _page_text_columns(page) -> str:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if len(words) < 60:
        return page.extract_text() or ""

    w, h = float(page.width), float(page.height)
    gutters = _column_gutters(words, w)
    if not gutters:
        return page.extract_text() or ""

    edges = [0.0, *gutters, w]
    parts = []
    for x0, x1 in zip(edges, edges[1:]):
        parts.append(page.crop((x0, 0.0, x1, h)).extract_text() or "")
    return "\n".join(p for p in parts if p.strip())


def extract_pdf_pages(path: Path, extractor: str = "pdfplumber") -> list[tuple[int, str, list]]:
    if extractor == "pypdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i, p.extract_text() or "", []) for i, p in enumerate(reader.pages, start=1)]

    import pdfplumber

    out: list[tuple[int, str, list]] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            out.append((i, _page_text_columns(page), page.extract_tables() or []))
    return out


def parse_pdf(path: Path, meta: dict, cfg: ChunkConfig | None = None) -> list[Block]:
    cfg = cfg or ChunkConfig()
    blocks: list[Block] = []
    for page_no, raw, tables in extract_pdf_pages(path, cfg.pdf_extractor):
        lines = clean_lines(raw)
        if not lines:
            continue
        default_section = detect_section(lines, meta.get("title", "General"))
        joined = _WS.sub(" ", " ".join(lines)).strip()
        if cfg.drop_boilerplate and is_low_value(joined, default_section, cfg):
            continue
        blocks.extend(segment_text(lines, page=page_no, default_section=default_section, cfg=cfg))
        if cfg.tables_atomic:
            for t in tables:
                s = serialize_table(t, section=default_section, pub=meta["pub"], page=page_no)
                if s and _words(s) >= 3:
                    blocks.append(Block(text=s, section=default_section, page=page_no,
                                        block_type="table"))
    return blocks
