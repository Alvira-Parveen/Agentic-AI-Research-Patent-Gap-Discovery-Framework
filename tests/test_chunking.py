import pytest
from src.schemas.documents import DocumentType, DocumentChunk
from src.preprocessing.cleaner import normalize_whitespace, clean_patent_claim_text
from src.preprocessing.chunker import split_text_into_chunks, create_section_aware_chunks
from src.preprocessing.deduplicator import deduplicate_chunks


def test_normalize_whitespace():
    raw = "This   is   a \n\n\n test    text with   excess   spaces."
    cleaned = normalize_whitespace(raw)
    assert "   " not in cleaned
    assert "test text with excess spaces." in cleaned


def test_clean_patent_claim_text():
    claim = "1. An AI system comprising: a camera; and a processor.\n\n2. The system of claim 1, further comprising a sensor."
    cleaned = clean_patent_claim_text(claim)
    assert "1. An AI system" in cleaned
    assert "2. The system" in cleaned


def test_split_text_into_chunks():
    text = "First sentence here. Second sentence follows. Third sentence appears. Fourth sentence completes."
    chunks = split_text_into_chunks(text, chunk_size=50, chunk_overlap=15)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_create_section_aware_chunks():
    sections = {
        "abstract": "This is the abstract describing the invention.",
        "claims": "1. What is claimed is a machine learning model for prior art analysis."
    }
    chunks = create_section_aware_chunks(
        doc_id="PAT-001",
        doc_type=DocumentType.PATENT,
        doc_title="Sample Patent",
        source="sample.pdf",
        sections=sections
    )
    assert len(chunks) == 2
    assert chunks[0].section == "abstract"
    assert chunks[1].section == "claims"
    assert chunks[0].document_id == "PAT-001"


def test_deduplicate_chunks():
    c1 = DocumentChunk(
        chunk_id="C1", document_id="D1", document_type=DocumentType.PATENT,
        document_title="T1", section="abstract", text="This is duplicate content for testing deduplication."
    )
    c2 = DocumentChunk(
        chunk_id="C2", document_id="D1", document_type=DocumentType.PATENT,
        document_title="T1", section="abstract", text="This is duplicate content for testing deduplication."
    )
    c3 = DocumentChunk(
        chunk_id="C3", document_id="D1", document_type=DocumentType.PATENT,
        document_title="T1", section="abstract", text="This is completely different and unique text."
    )
    deduped, removed = deduplicate_chunks([c1, c2, c3])
    assert len(deduped) == 2
    assert removed == 1
