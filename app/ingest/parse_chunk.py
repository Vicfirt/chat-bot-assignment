from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BARE_PAGENUM = re.compile(r"^\s*\d{1,4}\s*$")
_WS = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    raw = _HYPHEN_BREAK.sub(r"\1\2", raw)
    kept = [ln for ln in raw.splitlines() if not _BARE_PAGENUM.match(ln)]
    return _WS.sub(" ", " ".join(kept)).strip()


def detect_section(text: str, fallback: str) -> str:
    for ln in text.splitlines():
        s = ln.strip()
        if 3 <= len(s) <= 80 and (s.isupper() or s.istitle()) and not s.endswith("."):
            return s
    return fallback


def parse_pdf(path: Path, meta: dict) -> list[dict]:
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        out.append({"page": i, "text": clean_text(raw),
                    "section": detect_section(raw, meta.get("title", "General")), **meta})
    return out


def chunk_document(
    pages: list[dict], *, target_tokens: int = 600, overlap_tokens: int = 80
) -> list[dict]:
    chunks: list[dict] = []
    for pg in pages:
        words = pg["text"].split()
        if not words:
            continue
        step = max(1, target_tokens - overlap_tokens)
        idx = 0
        for start in range(0, len(words), step):
            window = words[start:start + target_tokens]
            if not window:
                break
            chunks.append({
                "chunk_id": f"{pg['name']}-{pg['page']}-{idx}",
                "text": " ".join(window),
                "pub": pg["pub"],
                "title": pg["title"],
                "section": pg.get("section", pg["title"]),
                "page": pg["page"],
                "source_url": pg["source_url"],
                "tax_year": pg["tax_year"],
            })
            idx += 1
            if start + target_tokens >= len(words):
                break
    return chunks
