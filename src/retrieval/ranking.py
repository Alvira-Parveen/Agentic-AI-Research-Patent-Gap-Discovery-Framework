from typing import List, Tuple, Dict
from src.schemas.documents import DocumentChunk
from src.schemas.retrieval import RetrievedChunk


def rank_and_deduplicate(
    raw_results: List[Tuple[DocumentChunk, float]],
    max_chunks_per_doc: int = 2,
    top_k: int = 5
) -> List[RetrievedChunk]:
    """
    Ranks retrieved chunks and limits the number of chunks from the same document
    to ensure diversity across prior-art citations.
    """
    # Sort descending by similarity score
    sorted_results = sorted(raw_results, key=lambda x: x[1], reverse=True)

    doc_chunk_count: Dict[str, int] = {}
    selected_chunks: List[RetrievedChunk] = []
    rank = 1

    for chunk, score in sorted_results:
        d_id = chunk.document_id
        count = doc_chunk_count.get(d_id, 0)
        if count >= max_chunks_per_doc:
            continue

        doc_chunk_count[d_id] = count + 1
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
        rank += 1

        if len(selected_chunks) >= top_k:
            break

    return selected_chunks
