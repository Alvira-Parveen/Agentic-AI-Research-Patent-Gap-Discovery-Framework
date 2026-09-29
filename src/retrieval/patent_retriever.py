from pathlib import Path
from typing import List, Optional
from src.retrieval.vector_store import FAISSVectorStore
from src.embeddings.embedding_service import EmbeddingService
from src.schemas.retrieval import RetrievedChunk
from src.retrieval.ranking import rank_and_deduplicate
from src.config.constants import VECTOR_STORE_DIR
from src.config.settings import settings
from src.utils.logging import logger


class PatentRetriever:
    def __init__(self, index_dir: Optional[Path] = None, embedding_service: Optional[EmbeddingService] = None):
        self.index_dir = index_dir or (VECTOR_STORE_DIR / "patents")
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.vector_store = FAISSVectorStore()
        self._is_loaded = False
        self._load()

    def _load(self):
        if self.index_dir.exists() and (self.index_dir / "index.faiss").exists():
            self._is_loaded = self.vector_store.load(self.index_dir)

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k_patents
        if not self._is_loaded:
            self._load()
            if not self._is_loaded:
                logger.warning("Patent vector store not loaded. Run scripts/build_index.py first.")
                return []

        query_emb = self.embedding_service.embed_query(query, normalize=True)
        # Fetch up to 4x top_k to allow diverse ranking across documents
        raw_results = self.vector_store.search(query_emb, top_k=top_k * 4)
        ranked = rank_and_deduplicate(raw_results, max_chunks_per_doc=2, top_k=top_k)
        return ranked
