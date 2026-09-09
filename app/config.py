from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_mode: Literal["ollama", "dummy"] = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://ollama:11434"
    # When llm_mode="ollama" but the model/server is unreachable, use the dummy LLM
    # instead of failing every request. Set false to surface the error hard.
    llm_fallback_dummy: bool = True

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "irs_pubs"
    raw_pdf_dir: str = "data/raw_pdfs"

    tax_year: int = 2025
    search_k: int = 5
    min_docs: int = 3
    context_token_budget: int = 900
    max_retries: int = 1
    graph_recursion_limit: int = 25          # LangGraph superstep cap (safety net)

    # --- caching (in-process) ---
    cache_enabled: bool = True
    cache_embedding_size: int = 512           # query-embedding LRU entries
    cache_rag_size: int = 256                 # RAG-subgraph result LRU entries

    # --- retrieval (all tunable) ---
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    dense_top_k: int = 20                     # candidates from the vector store
    bm25_top_k: int = 20                      # candidates from the keyword index
    rrf_k: int = 60                           # reciprocal-rank-fusion constant
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 20                    # candidates fed to the cross-encoder
    filter_tax_year: bool = True             # restrict dense search to settings.tax_year
    boost_tables_for_amount_queries: bool = True
    grade_min_score: float = 0.0             # drop candidates below this final score

    # --- ingestion / chunking (all tunable) ---
    pdf_extractor: Literal["pdfplumber", "pypdf"] = "pdfplumber"
    chunk_target_tokens: int = 450             # embedder tokens; bge-small caps at 512
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 32                 # drop fragments smaller than this
    chunk_heading_split: bool = True           # never let a chunk cross a heading
    chunk_tables_atomic: bool = True           # keep each table as one chunk
    chunk_keep_examples_atomic: bool = True    # keep "Example N." / worksheet blocks whole
    index_drop_boilerplate: bool = True        # skip index / TOC / watermark pages
    index_numeric_density: float = 0.28        # page is "index-like" above this digit ratio

    # --- logging ---
    log_level: str = "INFO"
    log_json: bool = True                     # structured JSON lines to stdout
    log_pii: bool = False                     # if false, redact SSNs before logging

    api_url: str = "http://api:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
