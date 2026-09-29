import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.retrieval.vector_store import FAISSVectorStore
from src.schemas.documents import DocumentChunk, DocumentType


def test_faiss_vector_store_search_and_save():
    dim = 64
    store = FAISSVectorStore(dimension=dim)

    # Create 3 synthetic chunks and random normalized vectors
    chunks = [
        DocumentChunk(chunk_id=f"C{i}", document_id=f"D{i}", document_type=DocumentType.PATENT,
                      document_title=f"Patent {i}", section="claim", text=f"Sample text {i}")
        for i in range(3)
    ]

    raw_vecs = np.random.randn(3, dim).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norm_vecs = raw_vecs / norms

    store.add_documents(chunks, norm_vecs)
    assert store.index.ntotal == 3

    # Query with vector identical to chunk 0
    query_vec = norm_vecs[0]
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "C0"
    assert pytest.approx(top_score, abs=1e-3) == 1.0  # Cosine similarity with itself is 1.0

    # Test Save & Load
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        store.save(tmp_path)

        loaded_store = FAISSVectorStore()
        success = loaded_store.load(tmp_path)
        assert success is True
        assert loaded_store.index.ntotal == 3
        assert len(loaded_store.chunks) == 3
