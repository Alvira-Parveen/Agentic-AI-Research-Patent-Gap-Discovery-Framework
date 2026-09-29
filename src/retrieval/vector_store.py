import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from src.schemas.documents import DocumentChunk
from src.utils.logging import logger


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Inner Product on L2-normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[DocumentChunk] = []

    def add_documents(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds chunks and their pre-computed L2-normalized embeddings to the FAISS index."""
        if len(chunks) == 0:
            return
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Number of chunks ({len(chunks)}) does not match embeddings ({embeddings.shape[0]})")
        
        # Ensure float32 contiguous array
        emb_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(emb_f32)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total entries: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Performs cosine similarity search using normalized query embedding.
        Returns list of (DocumentChunk, similarity_score) sorted descending.
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty FAISS index.")
            return []

        # Ensure query is 2D float32 array
        if query_embedding.ndim == 1:
            q_f32 = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        else:
            q_f32 = np.ascontiguousarray(query_embedding, dtype=np.float32)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_f32, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            # Cosine similarity in range [-1, 1], clip to [0.0, 1.0]
            norm_score = max(0.0, min(1.0, float(score)))
            results.append((self.chunks[idx], round(norm_score, 4)))

        return results

    def save(self, directory: Path) -> None:
        """Saves the FAISS index and chunk metadata mapping to disk."""
        directory.mkdir(parents=True, exist_ok=True)
        index_file = directory / "index.faiss"
        chunks_file = directory / "chunks.pkl"

        faiss.write_index(self.index, str(index_file))
        with open(chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} items) to {directory}")

    def load(self, directory: Path) -> bool:
        """Loads the FAISS index and chunk metadata mapping from disk."""
        index_file = directory / "index.faiss"
        chunks_file = directory / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            logger.warning(f"Vector store not found in {directory}")
            return False

        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index with {self.index.ntotal} chunks from {directory}")
        return True
