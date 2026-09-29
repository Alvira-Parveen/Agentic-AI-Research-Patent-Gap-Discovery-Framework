import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import logger
from src.schemas.documents import DocumentType
from src.ingestion.metadata import load_processed_dataset
from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore
from src.config.constants import VECTOR_STORE_DIR


def build_indexes():
    logger.info("==================================================")
    logger.info("Starting FAISS Vector Index Construction Pipeline")
    logger.info("==================================================")

    embedder = EmbeddingService.get_instance()

    # Build Patent Index
    pat_meta, pat_chunks = load_processed_dataset(DocumentType.PATENT)
    logger.info(f"Loaded {len(pat_meta)} patent metadata records and {len(pat_chunks)} patent chunks.")

    if pat_chunks:
        logger.info("Generating embeddings for patent chunks...")
        pat_texts = [c.text for c in pat_chunks]
        pat_embeddings = embedder.embed_texts(pat_texts, batch_size=32, normalize=True)
        logger.info(f"Generated embeddings shape: {pat_embeddings.shape}")

        pat_store = FAISSVectorStore(dimension=embedder.dimension)
        pat_store.add_documents(pat_chunks, pat_embeddings)
        pat_index_dir = VECTOR_STORE_DIR / "patents"
        pat_store.save(pat_index_dir)
        logger.info(f"Patent FAISS index saved to {pat_index_dir}")
    else:
        logger.warning("No patent chunks found! Did you run scripts/ingest_data.py?")

    # Build Research Papers Index
    pap_meta, pap_chunks = load_processed_dataset(DocumentType.PAPER)
    logger.info(f"Loaded {len(pap_meta)} paper metadata records and {len(pap_chunks)} paper chunks.")

    if pap_chunks:
        logger.info("Generating embeddings for paper chunks...")
        pap_texts = [c.text for c in pap_chunks]
        pap_embeddings = embedder.embed_texts(pap_texts, batch_size=32, normalize=True)
        logger.info(f"Generated embeddings shape: {pap_embeddings.shape}")

        pap_store = FAISSVectorStore(dimension=embedder.dimension)
        pap_store.add_documents(pap_chunks, pap_embeddings)
        pap_index_dir = VECTOR_STORE_DIR / "papers"
        pap_store.save(pap_index_dir)
        logger.info(f"Research paper FAISS index saved to {pap_index_dir}")
    else:
        logger.warning("No paper chunks found! Did you run scripts/ingest_data.py?")

    print("\n" + "=" * 50)
    print("VECTOR INDEX BUILD COMPLETE")
    print("=" * 50)
    print(f"Patent Chunks Indexed: {len(pat_chunks)}")
    print(f"Paper Chunks Indexed:  {len(pap_chunks)}")
    print(f"Embedding Dimension:   {embedder.dimension}")
    print(f"Index Locations:")
    print(f"  Patents: {VECTOR_STORE_DIR / 'patents'}")
    print(f"  Papers:  {VECTOR_STORE_DIR / 'papers'}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    build_indexes()
