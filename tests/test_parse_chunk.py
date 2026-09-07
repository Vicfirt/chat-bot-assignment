from app.ingest.parse_chunk import chunk_document, clean_text


def test_clean_text_dehyphenates():
    assert clean_text("stand-\nard deduction") == "standard deduction"


def test_clean_text_drops_bare_page_numbers():
    assert clean_text("Some line\n42\nAnother line") == "Some line Another line"


def test_chunk_document_produces_metadata_and_ids():
    pages = [
        {"page": 1, "text": "STANDARD DEDUCTION " + "word " * 900,
         "name": "pub501", "pub": "Pub. 501", "title": "T",
         "source_url": "http://x", "tax_year": 2024},
    ]
    chunks = chunk_document(pages, target_tokens=300, overlap_tokens=50)
    assert len(chunks) >= 3
    first = chunks[0]
    assert first["chunk_id"] == "pub501-1-0"
    assert first["pub"] == "Pub. 501"
    assert first["page"] == 1
    assert first["tax_year"] == 2024
    assert first["section"]  # non-empty
    assert chunks[0]["text"].split()[-10:] == chunks[1]["text"].split()[:10]
