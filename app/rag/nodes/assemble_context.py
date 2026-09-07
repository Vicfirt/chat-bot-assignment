from __future__ import annotations

from app.config import get_settings
from app.rag.state import RagState


def assemble_context(state: RagState) -> dict:
    budget = get_settings().context_token_budget
    hits = sorted(state.get("graded_hits", []), key=lambda h: h["score"], reverse=True)

    blocks: list[str] = []
    citations: list[dict] = []
    used = 0
    for h in hits:
        words = len(h["text"].split())
        if used + words > budget and blocks:
            break
        used += words
        blocks.append(f"[{h['pub']} — {h['section']} (p.{h['page']})]\n{h['text']}")
        citations.append({
            "pub": h["pub"], "section": h["section"], "page": h["page"],
            "source_url": h["source_url"], "quote": h["text"][:160],
        })
    return {"rag_context": "\n\n".join(blocks), "citations": citations}
