import pytest
import numpy as np
from src.embeddings.embedding_service import EmbeddingService


def test_embedding_dimension_and_normalization():
    service = EmbeddingService.get_instance()
    texts = ["Retrieval augmented generation for patent search", "Machine learning drone imaging"]
    embs = service.embed_texts(texts, normalize=True)

    assert embs.shape[0] == 2
    assert embs.shape[1] == service.dimension
    assert embs.dtype == np.float32

    # Check L2 normalization: norm should be approximately 1.0
    for i in range(2):
        norm = np.linalg.norm(embs[i])
        assert pytest.approx(norm, abs=1e-4) == 1.0


def test_embed_query():
    service = EmbeddingService.get_instance()
    q_emb = service.embed_query("Patent intelligence", normalize=True)
    assert q_emb.ndim == 1
    assert q_emb.shape[0] == service.dimension
    assert pytest.approx(np.linalg.norm(q_emb), abs=1e-4) == 1.0
