import re
from typing import List, Dict, Any, Optional
from src.schemas.documents import DocumentChunk, DocumentType
from src.config.settings import settings
from src.utils.ids import generate_chunk_id, format_citation


def split_text_into_chunks(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[str]:
    """Split text into overlapping character/word chunks while respecting sentence boundaries."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Split by sentences or paragraph markers
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        s_len = len(sentence)
        if current_length + s_len > chunk_size and current_chunk:
            combined = ' '.join(current_chunk).strip()
            if combined:
                chunks.append(combined)
            # Apply overlap: keep trailing sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_length = overlap_len

        current_chunk.append(sentence)
        current_length += s_len

    if current_chunk:
        combined = ' '.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def create_section_aware_chunks(
    doc_id: str,
    doc_type: DocumentType,
    doc_title: str,
    source: str,
    sections: Dict[str, str],
    page_mapping: Optional[Dict[str, int]] = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[DocumentChunk]:
    """
    Creates section-aware chunks prioritizing abstract, claims, methodology, etc.
    """
    chunks: List[DocumentChunk] = []
    chunk_idx = 0

    for section_name, section_text in sections.items():
        if not section_text or len(section_text.strip()) < 20:
            continue

        page_num = page_mapping.get(section_name) if page_mapping else None
        raw_chunks = split_text_into_chunks(section_text, chunk_size, chunk_overlap)

        for text_piece in raw_chunks:
            chunk_id = generate_chunk_id(doc_id, chunk_idx)
            citation = format_citation(doc_id, section_name, page_num)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                document_type=doc_type,
                document_title=doc_title,
                section=section_name,
                text=text_piece,
                page=page_num,
                chunk_index=chunk_idx,
                source=source,
                metadata={
                    "citation": citation,
                    "section": section_name,
                    "doc_type": doc_type.value,
                    "doc_title": doc_title
                }
            )
            chunks.append(chunk)
            chunk_idx += 1

    return chunks
