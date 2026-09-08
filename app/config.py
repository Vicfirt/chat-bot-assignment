from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_mode: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://ollama:11434"
    # When llm_mode="ollama" but the model/server is unreachable, use the dummy LLM
    # instead of failing every request. Set false to surface the error hard.
    llm_fallback_dummy: bool = True

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "irs_pubs"
    raw_pdf_dir: str = "data/raw_pdfs"

    tax_year: int = 2024
    search_k: int = 5
    min_docs: int = 3
    context_token_budget: int = 2000
    max_retries: int = 1

    # --- ingestion / chunking (all tunable) ---
    pdf_extractor: str = "pdfplumber"          # "pdfplumber" (layout+tables) or "pypdf"
    chunk_target_tokens: int = 450             # embedder tokens; bge-small caps at 512
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 32                 # drop fragments smaller than this
    chunk_heading_split: bool = True           # never let a chunk cross a heading
    chunk_tables_atomic: bool = True           # keep each table as one chunk
    chunk_keep_examples_atomic: bool = True    # keep "Example N." / worksheet blocks whole
    index_drop_boilerplate: bool = True        # skip index / TOC / watermark pages
    index_numeric_density: float = 0.28        # page is "index-like" above this digit ratio

    langfuse_enabled: bool = False
    langfuse_host: str = "http://langfuse-web:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    api_url: str = "http://api:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
