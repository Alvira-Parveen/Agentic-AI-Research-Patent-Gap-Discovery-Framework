from typing import List
from src.schemas.retrieval import RetrievedChunk


def build_rag_context(chunks: List[RetrievedChunk], max_tokens_estimate: int = 1500) -> str:
    """
    Constructs a formatted context string from retrieved chunks
    with standardized citations and metadata headers.
    """
    if not chunks:
        return "No prior-art documents were retrieved for this query."

    context_blocks = []
    total_chars = 0
    char_limit = max_tokens_estimate * 4

    for idx, c in enumerate(chunks, 1):
        citation = c.citation or f"[{c.document_id}]"
        header = f"--- EVIDENCE ITEM {idx}: {citation} ---"
        meta_line = f"Title: {c.document_title} | Type: {c.document_type.value.upper()} | Similarity: {c.similarity_score:.4f}"
        body = f"{header}\n{meta_line}\nExcerpt:\n{c.text}\n"

        if total_chars + len(body) > char_limit:
            # Truncate if exceeding limit
            break

        context_blocks.append(body)
        total_chars += len(body)

    return "\n".join(context_blocks)
