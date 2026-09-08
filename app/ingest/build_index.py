from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.ingest.download import download_all, load_sources
from app.ingest.parse_chunk import ChunkConfig, chunk_blocks, parse_pdf
from app.rag.retriever import ChromaRetriever


def build_index(sources_path: str | Path | None = None) -> int:
    sources_path = Path(sources_path or "app/ingest/sources.yaml")
    s = get_settings()
    cfg = ChunkConfig.from_settings(s)
    paths = download_all(sources_path)
    sources = {x["name"]: x for x in load_sources(sources_path)}

    retriever = ChromaRetriever()
    token_len = retriever.token_count  # measure chunks in real embedder tokens

    all_chunks: list[dict] = []
    for pdf_path in paths:
        name = pdf_path.stem
        src = sources[name]
        meta = {"name": name, "pub": src["pub"], "title": src["title"],
                "source_url": src["url"], "tax_year": s.tax_year}
        blocks = parse_pdf(pdf_path, meta, cfg)
        all_chunks.extend(chunk_blocks(blocks, meta, cfg, token_len))

    retriever.add_chunks(all_chunks)
    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1
    print(f"chunks by type: {by_type}")
    return retriever.count()


if __name__ == "__main__":
    print(f"indexed {build_index()} chunks")
