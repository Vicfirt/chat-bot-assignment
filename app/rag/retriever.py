from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import chromadb

from app.config import get_settings


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    pub: str
    section: str
    page: int
    source_url: str
    tax_year: int
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...


class ChromaRetriever:
    def __init__(self, chroma_dir: str | None = None, collection: str | None = None,
                 embedding_model: str | None = None) -> None:
        s = get_settings()
        self._client = chromadb.PersistentClient(path=chroma_dir or s.chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection or s.chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        self._model_name = embedding_model or s.embedding_model
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._load_model().encode(texts, normalize_embeddings=True).tolist()

    def token_count(self, text: str) -> int:
        """Length of `text` in the embedding model's own tokens."""
        return len(self._load_model().tokenizer.encode(text, add_special_tokens=False))

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        keys = ("pub", "section", "page", "source_url", "tax_year")
        self._collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=self._embed([c["text"] for c in chunks]),
            metadatas=[{**{k: c[k] for k in keys}, "block_type": c.get("block_type", "prose")}
                       for c in chunks],
        )

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, k: int) -> list[Chunk]:
        res = self._collection.query(query_embeddings=self._embed([query]), n_results=k)
        out: list[Chunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(Chunk(
                chunk_id=cid, text=doc, pub=meta["pub"], section=meta["section"],
                page=int(meta["page"]), source_url=meta["source_url"],
                tax_year=int(meta["tax_year"]), score=1.0 - float(dist),
            ))
        return out


_retriever: ChromaRetriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = ChromaRetriever()
    return _retriever
