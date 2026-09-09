"""Tool #1 (retrieval): exposes the modular RAG subgraph as a single call."""
from __future__ import annotations

from app.rag.subgraph import run_rag


def retriever_tool(question: str, chat_history: list[dict],
                   tax_profile: dict | None = None) -> dict:
    return run_rag(question, chat_history, tax_profile)
