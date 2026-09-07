from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_mode: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://ollama:11434"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "irs_pubs"
    raw_pdf_dir: str = "data/raw_pdfs"

    tax_year: int = 2024
    search_k: int = 5
    min_docs: int = 3
    context_token_budget: int = 2000
    max_retries: int = 1

    langfuse_enabled: bool = False
    langfuse_host: str = "http://langfuse-web:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    api_url: str = "http://api:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
