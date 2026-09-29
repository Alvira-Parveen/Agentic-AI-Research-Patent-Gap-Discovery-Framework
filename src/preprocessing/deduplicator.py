import hashlib
from typing import List, Set, Tuple
from src.schemas.documents import DocumentChunk

def compute_text_hash(text: str) -> str:
    cleaned = ''.join(text.lower().split())
    return hashlib.md5(cleaned.encode('utf-8')).hexdigest()

def deduplicate_chunks(chunks: List[DocumentChunk], threshold: float = 0.95) -> Tuple[List[DocumentChunk], int]:
    """
    Remove exact and near-identical chunks to avoid vector redundancy.
    Returns (deduplicated_chunks, duplicates_removed_count).
    """
    seen_hashes: Set[str] = set()
    unique_chunks: List[DocumentChunk] = []
    duplicates_removed = 0

    for chunk in chunks:
        # Ignore empty chunks
        if len(chunk.text.strip()) < 30:
            duplicates_removed += 1
            continue

        thash = compute_text_hash(chunk.text)
        if thash in seen_hashes:
            duplicates_removed += 1
            continue

        seen_hashes.add(thash)
        unique_chunks.append(chunk)

    return unique_chunks, duplicates_removed
