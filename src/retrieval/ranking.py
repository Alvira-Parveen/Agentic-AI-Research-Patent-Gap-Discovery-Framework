from typing import List, Tuple, Dict
from src.schemas.documents import DocumentChunk
from src.schemas.retrieval import RetrievedChunk


def rank_and_deduplicate(
    raw_results: List[Tuple[DocumentChunk, float]],
    max_chunks_per_doc: int = 1,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Ranks retrieved chunks using max-score document aggregation.

    For each unique document, keeps only the single highest-scoring chunk
    (max-aggregation strategy). This maximises document diversity so the
    top-k results represent k distinct prior-art documents rather than
    multiple excerpts from the same source, which improves Precision@K
    and document-level recall.

    Args:
        raw_results: List of (DocumentChunk, similarity_score) tuples from FAISS.
        max_chunks_per_doc: Kept for API compatibility; aggregation always
            produces exactly one representative chunk per document.
        top_k: Maximum number of documents to return.

    Returns:
        Ranked list of RetrievedChunk objects, one per unique document.
    """
    if not raw_results:
        return []

    # --- Max-score document aggregation ---
    # For each document, retain only the chunk with the highest similarity score.
    doc_best: Dict[str, Tuple[DocumentChunk, float]] = {}
    for chunk, score in raw_results:
        d_id = chunk.document_id
        if d_id not in doc_best or score > doc_best[d_id][1]:
            doc_best[d_id] = (chunk, score)

    # Sort documents by their best chunk score, descending
    sorted_docs = sorted(doc_best.values(), key=lambda x: x[1], reverse=True)

    selected_chunks: List[RetrievedChunk] = []
    for rank, (chunk, score) in enumerate(sorted_docs[:top_k], start=1):
        citation = chunk.metadata.get("citation", f"[{chunk.document_id}]")
        retrieved = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_type=chunk.document_type,
            document_title=chunk.document_title,
            section=chunk.section,
            text=chunk.text,
            similarity_score=score,
            rank=rank,
            source=chunk.source,
            page=chunk.page,
            citation=citation,
            metadata=chunk.metadata
        )
        selected_chunks.append(retrieved)

    return selected_chunks


def rank_hybrid_rrf(
    dense_results: List[Tuple[DocumentChunk, float]],
    bm25_results: List[Tuple[DocumentChunk, float]],
    rrf_k: int = 60,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Ranks documents using Reciprocal Rank Fusion (RRF) between dense semantic
    embeddings and sparse BM25 lexical signals (Cormack et al., SIGIR 2009),
    combined with max-score document aggregation.

    RRF formula for document d:
        RRF(d) = 1 / (k + r_dense(d)) + 1 / (k + r_bm25(d))

    Args:
        dense_results: (DocumentChunk, cosine_similarity) from FAISS.
        bm25_results: (DocumentChunk, bm25_score) from BM25Okapi.
        rrf_k: Smoothing constant for reciprocal rank (standard default: 60).
        top_k: Number of representative documents to return.

    Returns:
        Ranked list of RetrievedChunk objects ordered by fused RRF score.
    """
    if not dense_results and not bm25_results:
        return []

    # 1. Max-score doc aggregation for Dense
    dense_doc: Dict[str, Tuple[DocumentChunk, float]] = {}
    for chunk, score in dense_results:
        d_id = chunk.document_id
        if d_id not in dense_doc or score > dense_doc[d_id][1]:
            dense_doc[d_id] = (chunk, score)
    sorted_dense = sorted(dense_doc.items(), key=lambda x: x[1][1], reverse=True)
    dense_ranks = {did: rank for rank, (did, _) in enumerate(sorted_dense, start=1)}

    # 2. Max-score doc aggregation for BM25
    bm25_doc: Dict[str, Tuple[DocumentChunk, float]] = {}
    for chunk, score in bm25_results:
        d_id = chunk.document_id
        if d_id not in bm25_doc or score > bm25_doc[d_id][1]:
            bm25_doc[d_id] = (chunk, score)
    sorted_bm25 = sorted(bm25_doc.items(), key=lambda x: x[1][1], reverse=True)
    bm25_ranks = {did: rank for rank, (did, _) in enumerate(sorted_bm25, start=1)}

    # 3. Fuse reciprocal ranks across all discovered documents
    all_dids = set(dense_ranks.keys()) | set(list(bm25_ranks.keys())[:30])
    fused_docs = []
    default_rank = 100

    for did in all_dids:
        r_d = dense_ranks.get(did, default_rank)
        r_b = bm25_ranks.get(did, default_rank)
        rrf_score = (1.0 / (rrf_k + r_d)) + (1.0 / (rrf_k + r_b))

        # Best representative chunk: prefer dense chunk if available, otherwise BM25
        chunk = dense_doc[did][0] if did in dense_doc else bm25_doc[did][0]
        # Preserve cosine similarity score for downstream calibrated novelty calculation
        sim_score = dense_doc[did][1] if did in dense_doc else 0.40

        fused_docs.append((chunk, sim_score, rrf_score))

    fused_docs.sort(key=lambda x: x[2], reverse=True)

    selected_chunks: List[RetrievedChunk] = []
    for rank, (chunk, sim_score, rrf_score) in enumerate(fused_docs[:top_k], start=1):
        citation = chunk.metadata.get("citation", f"[{chunk.document_id}]") if chunk.metadata else f"[{chunk.document_id}]"
        chunk_meta = dict(chunk.metadata) if chunk.metadata else {}
        chunk_meta["rrf_score"] = round(rrf_score, 6)
        retrieved = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_type=chunk.document_type,
            document_title=chunk.document_title,
            section=chunk.section,
            text=chunk.text,
            similarity_score=round(sim_score, 4),
            rank=rank,
            source=chunk.source,
            page=chunk.page,
            citation=citation,
            metadata=chunk_meta
        )
        selected_chunks.append(retrieved)

    return selected_chunks
