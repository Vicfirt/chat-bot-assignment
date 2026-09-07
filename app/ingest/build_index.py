from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.ingest.download import download_all, load_sources
from app.ingest.parse_chunk import chunk_document, parse_pdf
from app.rag.retriever import ChromaRetriever


def build_index(sources_path: str | Path | None = None) -> int:
    sources_path = Path(sources_path or "app/ingest/sources.yaml")
    s = get_settings()
    paths = download_all(sources_path)
    sources = {x["name"]: x for x in load_sources(sources_path)}

    all_chunks: list[dict] = []
    for pdf_path in paths:
        name = pdf_path.stem
        src = sources[name]
        meta = {"name": name, "pub": src["pub"], "title": src["title"],
                "source_url": src["url"], "tax_year": s.tax_year}
        pages = parse_pdf(pdf_path, meta)
        all_chunks.extend(chunk_document(pages))

    retriever = ChromaRetriever()
    retriever.add_chunks(all_chunks)
    return retriever.count()


if __name__ == "__main__":
    print(f"indexed {build_index()} chunks")
