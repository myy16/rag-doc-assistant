"""
Unit tests for app.core.chunker — chunk_text()

Run from backend/ directory:
    pytest tests/test_chunker.py -v
"""
import sys
import os

# Allow imports from backend/ without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.chunker import chunk_text, _split, _apply_overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_METADATA = {
    "file_id": "test-file-id-123",
    "source_file": "test.pdf",
    "file_type": "pdf",
}


# ---------------------------------------------------------------------------
# Empty / trivial input
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty_list():
    assert chunk_text("", BASE_METADATA) == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n\t  ", BASE_METADATA) == []


def test_short_text_produces_single_chunk():
    text = "Bu kısa bir cümledir."
    result = chunk_text(text, BASE_METADATA, chunk_size=500, chunk_overlap=0)
    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["chunk_index"] == 0
    assert result[0]["total_chunks"] == 1


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------

def test_chunk_contains_all_metadata_fields():
    text = "Test metni. " * 10
    result = chunk_text(text, BASE_METADATA, chunk_size=500, chunk_overlap=0)
    assert len(result) >= 1
    chunk = result[0]
    assert chunk["file_id"] == BASE_METADATA["file_id"]
    assert chunk["source_file"] == BASE_METADATA["source_file"]
    assert chunk["file_type"] == BASE_METADATA["file_type"]
    assert "chunk_id" in chunk
    assert "chunk_index" in chunk
    assert "total_chunks" in chunk
    assert "text" in chunk
    assert "char_count" in chunk


def test_chunk_id_is_unique():
    text = "A" * 600
    result = chunk_text(text, BASE_METADATA, chunk_size=200, chunk_overlap=0)
    ids = [c["chunk_id"] for c in result]
    assert len(ids) == len(set(ids)), "chunk_id'ler benzersiz olmalı"


def test_total_chunks_matches_result_length():
    text = "Cümle. " * 100
    result = chunk_text(text, BASE_METADATA, chunk_size=100, chunk_overlap=0)
    for chunk in result:
        assert chunk["total_chunks"] == len(result)


def test_chunk_index_is_sequential():
    text = "Kelime " * 200
    result = chunk_text(text, BASE_METADATA, chunk_size=100, chunk_overlap=0)
    for i, chunk in enumerate(result):
        assert chunk["chunk_index"] == i


def test_char_count_matches_text_length():
    text = "Türkçe karakter testi: ğ, ş, ı, ö, ü, ç. " * 20
    result = chunk_text(text, BASE_METADATA, chunk_size=200, chunk_overlap=0)
    for chunk in result:
        assert chunk["char_count"] == len(chunk["text"])


# ---------------------------------------------------------------------------
# Chunk size enforcement
# ---------------------------------------------------------------------------

def test_chunks_respect_chunk_size():
    text = "Bu bir test cümlesidir ve biraz uzundur. " * 50
    chunk_size = 200
    result = chunk_text(text, BASE_METADATA, chunk_size=chunk_size, chunk_overlap=0)
    for chunk in result:
        assert len(chunk["text"]) <= chunk_size + 10, (
            f"Chunk boyutu {chunk_size}'i aşmamalı, bulundu: {len(chunk['text'])}"
        )


def test_multiple_chunks_produced_for_long_text():
    text = "A" * 1500
    result = chunk_text(text, BASE_METADATA, chunk_size=500, chunk_overlap=0)
    assert len(result) >= 3


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def test_overlap_prepends_tail_of_previous_chunk():
    text = "A" * 500 + "\n\n" + "B" * 500
    result = chunk_text(text, BASE_METADATA, chunk_size=500, chunk_overlap=50)
    assert len(result) >= 2
    # Second chunk should start with the last 50 chars of first chunk
    tail = result[0]["text"][-50:]
    assert result[1]["text"].startswith(tail)


def test_zero_overlap_no_prefix():
    text = "İlk paragraf içeriği.\n\nİkinci paragraf içeriği."
    result = chunk_text(text, BASE_METADATA, chunk_size=30, chunk_overlap=0)
    if len(result) >= 2:
        # Without overlap, second chunk should NOT start with tail of first
        tail = result[0]["text"][-10:]
        # tail might appear naturally in text, but shouldn't be prepended
        # Just assert no double content at boundary
        assert result[1]["text"].count(tail) <= 1


# ---------------------------------------------------------------------------
# Turkish character handling
# ---------------------------------------------------------------------------

def test_turkish_characters_preserved():
    text = "Türkçe karakterler: ğ, ş, ı, ö, ü, ç, Ğ, Ş, İ, Ö, Ü, Ç"
    result = chunk_text(text, BASE_METADATA, chunk_size=500, chunk_overlap=0)
    assert len(result) == 1
    assert "ğ" in result[0]["text"]
    assert "ş" in result[0]["text"]
    assert "ı" in result[0]["text"]
    assert "İ" in result[0]["text"]


# ---------------------------------------------------------------------------
# Separator priority
# ---------------------------------------------------------------------------

def test_prefers_paragraph_split_over_word_split():
    """Double newline should be used before falling back to spaces."""
    paragraph_a = "A " * 60   # ~120 chars
    paragraph_b = "B " * 60   # ~120 chars
    text = paragraph_a.strip() + "\n\n" + paragraph_b.strip()
    result = chunk_text(text, BASE_METADATA, chunk_size=150, chunk_overlap=0)
    # Each paragraph should land in a separate chunk
    assert any("A" in c["text"] and "B" not in c["text"] for c in result)
    assert any("B" in c["text"] for c in result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_split_short_text_no_split():
    assert _split("kısa metin", 500) == ["kısa metin"]


def test_split_exact_chunk_size():
    text = "x" * 500
    result = _split(text, 500)
    assert result == [text]


def test_split_hard_cut_fallback():
    """No separators present — must hard-cut at chunk_size."""
    text = "a" * 1200
    result = _split(text, 500)
    assert all(len(c) <= 500 for c in result)
    assert "".join(result) == text


def test_apply_overlap_single_chunk():
    assert _apply_overlap(["tek chunk"], 50) == ["tek chunk"]


def test_apply_overlap_zero():
    chunks = ["birinci", "ikinci", "üçüncü"]
    assert _apply_overlap(chunks, 0) == chunks