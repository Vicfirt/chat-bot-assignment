"""Retrieval: dense (Chroma) + BM25 keyword, fused with RRF, then cross-encoder
reranked, with an optional table boost for amount/rate questions. Every knob is
on `app.config.Settings`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Protocol

import chromadb

from app.config import get_settings

_log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9$%]+")
_AMOUNT_RE = re.compile(
    r"\b(how much|amount|rate|bracket|table|standard deduction|deduction is|"
    r"dollars?|percent)\b|[$%]",
    re.I,
)


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
    block_type: str = "prose"


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _chunk_from_meta(cid: str, doc: str, meta: dict, score: float) -> Chunk:
    return Chunk(
        chunk_id=cid, text=doc, pub=meta.get("pub", "?"),
        section=meta.get("section", ""), page=int(meta.get("page", 0)),
        source_url=meta.get("source_url", ""), tax_year=int(meta.get("tax_year", 0)),
        score=float(score), block_type=meta.get("block_type", "prose"),
    )


class ChromaRetriever:
    """Dense vector backend."""

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

    def all_documents(self) -> list[dict]:
        got = self._collection.get(include=["documents", "metadatas"])
        return [{"chunk_id": i, "text": d, "meta": m}
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])]

    def dense_search(self, query: str, k: int, where: dict | None = None) -> list[Chunk]:
        res = self._collection.query(
            query_embeddings=self._embed([query]), n_results=k, where=where or None
        )
        if not res.get("ids") or not res["ids"][0]:
            return []
        return [
            _chunk_from_meta(cid, doc, meta, 1.0 - float(dist))
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def search(self, query: str, k: int) -> list[Chunk]:  # dense-only; hybrid path is HybridRetriever
        return self.dense_search(query, k)


class HybridRetriever:
    """Dense + BM25 + RRF + cross-encoder rerank + table boost."""

    def __init__(self, dense: ChromaRetriever | None = None) -> None:
        self._dense = dense or ChromaRetriever()
        self._bm25 = None
        self._bm25_docs: list[dict] = []
        self._reranker = None

    # -- BM25 --------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._bm25_docs = self._dense.all_documents()
        corpus = [_tokenize(d["text"]) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(corpus) if corpus else False

    def _bm25_search(self, query: str, k: int) -> list[Chunk]:
        self._ensure_bm25()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: list[Chunk] = []
        for i in order:
            if scores[i] <= 0:
                continue
            d = self._bm25_docs[i]
            out.append(_chunk_from_meta(d["chunk_id"], d["text"], d["meta"], float(scores[i])))
        return out

    # -- reranker --------------------------------------------------------
    def _ensure_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(get_settings().rerank_model)
        return self._reranker

    def _boost_tables(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        s = get_settings()
        if not (s.boost_tables_for_amount_queries and _AMOUNT_RE.search(question)):
            return chunks
        tables = [c for c in chunks if c.block_type == "table"]
        rest = [c for c in chunks if c.block_type != "table"]
        return tables + rest

    def rerank(self, question: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        s = get_settings()
        if not chunks:
            return []
        if s.rerank_enabled:
            try:
                pool = chunks[: s.rerank_top_n]
                raw = self._ensure_reranker().predict([(question, c.text) for c in pool])
                rescored = sorted(
                    (replace(c, score=float(sc)) for c, sc in zip(pool, raw)),
                    key=lambda c: c.score, reverse=True,
                )
                kept = [c for c in rescored if c.score >= s.grade_min_score]
                return self._boost_tables(question, kept)[:k]
            except Exception as e:  # noqa: BLE001 - degrade to fused order, don't 500
                _log.warning("reranker unavailable (%s); using fused order", e)
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return self._boost_tables(question, ranked)[:k]

    # -- RRF fusion ----------------------------------------------------
    def _rrf(self, ranked_lists: list[list[Chunk]], k: int) -> list[Chunk]:
        rrf_k = get_settings().rrf_k
        agg: dict[str, list] = {}
        for lst in ranked_lists:
            for rank, c in enumerate(lst):
                cur = agg.setdefault(c.chunk_id, [0.0, c])
                cur[0] += 1.0 / (rrf_k + rank + 1)
        fused = sorted(agg.values(), key=lambda t: t[0], reverse=True)[:k]
        return [replace(c, score=score) for score, c in fused]

    def search_candidates(self, query: str, k: int) -> list[Chunk]:
        """Dense + BM25 + RRF fusion, no rerank. This is the candidate pool the
        `rerank` subgraph node consumes."""
        s = get_settings()
        where = {"tax_year": s.tax_year} if s.filter_tax_year else None
        lists: list[list[Chunk]] = []
        if s.retrieval_mode in ("hybrid", "dense"):
            lists.append(self._dense.dense_search(query, s.dense_top_k, where=where))
        if s.retrieval_mode in ("hybrid", "bm25"):
            bm = self._bm25_search(query, s.bm25_top_k)
            if s.filter_tax_year:
                bm = [c for c in bm if c.tax_year == s.tax_year]
            lists.append(bm)
        if not lists:
            return []
        return self._rrf(lists, max(k, s.rerank_top_n)) if len(lists) > 1 else lists[0]

    def search(self, query: str, k: int) -> list[Chunk]:
        return self.rerank(query, self.search_candidates(query, k), k)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
