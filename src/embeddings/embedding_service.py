import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.config.settings import settings
from src.utils.logging import logger


class EmbeddingService:
    _instance = None

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.device
        logger.info(f"Initializing EmbeddingService with model={self.model_name} on device={self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded successfully. Embedding dimension: {self.dimension}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Embeds a list of strings into a float32 numpy array.
        If normalize=True, embeddings are unit-length (L2-normalized)
        such that inner product equals cosine similarity.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embeds a single query string into a 1D float32 numpy array."""
        embeddings = self.embed_texts([query], batch_size=1, normalize=normalize)
        return embeddings[0]
