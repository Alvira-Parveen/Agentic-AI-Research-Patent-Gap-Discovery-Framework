from pathlib import Path
from typing import List, Optional
from src.retrieval.vector_store import FAISSVectorStore
from src.embeddings.embedding_service import EmbeddingService
from src.schemas.retrieval import RetrievedChunk
from src.retrieval.ranking import rank_and_deduplicate, rank_hybrid_rrf
from src.retrieval.bm25 import BM25OkapiIndex
from src.config.constants import VECTOR_STORE_DIR
from src.config.settings import settings
from src.utils.logging import logger


class PaperRetriever:
    def __init__(self, index_dir: Optional[Path] = None, embedding_service: Optional[EmbeddingService] = None):
        self.index_dir = index_dir or (VECTOR_STORE_DIR / "papers")
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.vector_store = FAISSVectorStore()
        self.bm25_index: Optional[BM25OkapiIndex] = None
        self._is_loaded = False
        self._load()

    def _load(self):
        if self.index_dir.exists() and (self.index_dir / "index.faiss").exists():
            self._is_loaded = self.vector_store.load(self.index_dir)
            if self._is_loaded and self.vector_store.chunks:
                try:
                    self.bm25_index = BM25OkapiIndex(self.vector_store.chunks)
                except Exception as e:
                    logger.warning(f"Failed to initialize BM25 index for research papers: {e}")
                    self.bm25_index = None

    def retrieve(self, query: str, top_k: int = None, mode: str = "hybrid") -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k_papers
        if not self._is_loaded:
            self._load()
            if not self._is_loaded:
                logger.warning("Paper vector store not loaded. Run scripts/build_index.py first.")
                return []

        mode = mode.lower()
        if mode in ["bm25", "sparse"]:
            if not self.bm25_index:
                return []
            bm25_results = self.bm25_index.search(query, top_k=max(40, top_k * 8))
            return rank_and_deduplicate(bm25_results, max_chunks_per_doc=1, top_k=top_k)

        query_emb = self.embedding_service.embed_query(query, normalize=True)
        dense_results = self.vector_store.search(query_emb, top_k=max(40, top_k * 8))

        if mode == "hybrid" and self.bm25_index:
            bm25_results = self.bm25_index.search(query, top_k=max(40, top_k * 8))
            ranked = rank_hybrid_rrf(dense_results, bm25_results, rrf_k=60, top_k=top_k)
        else:
            ranked = rank_and_deduplicate(dense_results, max_chunks_per_doc=1, top_k=top_k)

        return ranked

