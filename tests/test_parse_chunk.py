from app.ingest.parse_chunk import (
    Block,
    ChunkConfig,
    _words,
    chunk_blocks,
    clean_text,
    is_low_value,
    looks_like_heading,
    segment_text,
    serialize_table,
)

CFG = ChunkConfig()
META = {"name": "pub501", "pub": "Pub. 501", "title": "Dependents", "source_url": "http://x",
        "tax_year": 2024}


def test_clean_text_dehyphenates():
    assert clean_text("stand-\nard deduction") == "standard deduction"


def test_clean_text_drops_bare_page_numbers():
    assert clean_text("Some line\n42\nAnother line") == "Some line Another line"


def test_clean_text_strips_irs_draft_watermark():
    raw = (
        "Standard deduction for single filers is $14,600. "
        "Page 29 of 142 Fileid: … ication-501/2024/a/xml/cycle02/source 13:47 - 20-Jan-2026 "
        "The type and rule above prints on all proofs including departmental reproduction "
        "proofs. MUST be removed before printing. More text here."
    )
    out = clean_text(raw)
    assert "$14,600" in out and "More text here." in out
    assert "Fileid:" not in out and "MUST be removed before printing" not in out


def test_looks_like_heading():
    assert looks_like_heading("Standard Deduction")
    assert looks_like_heading("WHO MUST FILE")
    assert looks_like_heading("Example 3.")
    assert looks_like_heading("Table 8. Standard Deduction Worksheet")
    assert not looks_like_heading("The 2024 standard deduction for a single filer is $14,600.")
    assert not looks_like_heading("a")


def test_is_low_value_flags_index_toc_and_number_lists():
    assert is_low_value(
        "To help us develop a more useful index, please let us know if you have ideas "
        "for index entries.", "Index", CFG)
    assert is_low_value("anything", "Table of Contents", CFG)
    assert is_low_value(" ".join(["Deduction", "12"] * 40), "Index", CFG)
    assert not is_low_value(
        "The 2024 standard deduction for a single filer is $14,600.", "Standard Deduction", CFG)


def test_serialize_table():
    out = serialize_table(
        [["Filing status", "Standard deduction"], ["Single", "$14,600"],
         ["Married filing jointly", "$29,200"], [None, None]],
        section="Standard Deduction", pub="Pub. 501", page=29)
    assert out.splitlines()[0] == "[Table - Standard Deduction (Pub. 501 p.29)]"
    assert "Single | $14,600" in out
    assert "Married filing jointly | $29,200" in out
    assert out.count("\n") == 3  # header + 3 non-empty rows


def test_segment_text_splits_at_headings_and_marks_examples():
    lines = [
        "Standard Deduction",
        "For 2024 the standard deduction is 14,600 for single filers.",
        "It is higher if you are 65 or older.",
        "Example 1.",
        "You are single and 70. Your standard deduction is 16,550.",
        "Filing Status",
        "Your filing status determines your deduction and rates.",
    ]
    blocks = segment_text(lines, page=29, default_section="General", cfg=CFG)
    sections = [b.section for b in blocks]
    assert sections == ["Standard Deduction", "Example 1", "Filing Status"]
    assert [b.block_type for b in blocks] == ["prose", "example", "prose"]
    # no block contains a later heading's text
    assert "Filing Status" not in blocks[0].text
    assert "Example" not in blocks[0].text


def test_chunk_blocks_windows_prose_but_keeps_small_tables_and_examples_atomic():
    big_prose = "word " * 400
    blocks = [
        Block(text="Standard Deduction " + big_prose, section="Standard Deduction", page=1,
              block_type="prose"),
        Block(text="[Table - X (Pub. 501 p.29)]\nSingle | 14,600\nMFJ | 29,200", section="X",
              page=29, block_type="table"),
        Block(text="Example 2. You are 70 and single; your deduction is 16,550.",
              section="Example 2", page=5, block_type="example"),
    ]
    cfg = ChunkConfig(target_tokens=50, overlap_tokens=10)
    chunks = chunk_blocks(blocks, META, cfg)  # default word-count token_len
    kinds = {c["block_type"]: 0 for c in chunks}
    for c in chunks:
        kinds[c["block_type"]] += 1
    assert kinds["prose"] > 3          # prose was windowed
    assert kinds["table"] == 1         # small table kept whole
    assert kinds["example"] == 1       # example kept whole
    assert [c for c in chunks if c["block_type"] == "prose"][0]["chunk_id"] == "pub501-1-0"
    assert all(set(c) >= {"chunk_id", "text", "pub", "section", "page", "block_type"} for c in chunks)


def test_oversized_atomic_table_is_windowed_with_caption_kept():
    from app.ingest.parse_chunk import _EMBED_MAX_TOKENS

    head = "[Table - Rates (Pub. 17 p.6)]"
    big = head + "\n" + "bracket | rate\n" * 4000  # far over the embed cap
    blocks = [Block(text=big, section="Rates", page=6, block_type="table")]
    chunks = chunk_blocks(blocks, META, ChunkConfig(target_tokens=60, overlap_tokens=10))
    assert len(chunks) > 1                                  # was split
    assert all(c["block_type"] == "table" for c in chunks)
    assert all(c["text"].startswith(head) for c in chunks)  # caption on every piece
    assert all(_words(c["text"]) <= 200 for c in chunks)    # roughly bounded


def test_chunk_blocks_drops_tiny_prose_fragments():
    blocks = [Block(text="too short", section="S", page=1, block_type="prose")]
    cfg = ChunkConfig(min_tokens=10)
    assert chunk_blocks(blocks, META, cfg) == []


def test_chunk_config_from_settings_maps_fields():
    from app.config import get_settings

    get_settings.cache_clear()
    cfg = ChunkConfig.from_settings(get_settings())
    assert cfg.target_tokens == 450
    assert cfg.overlap_tokens == 64
    assert cfg.tables_atomic is True
    get_settings.cache_clear()
