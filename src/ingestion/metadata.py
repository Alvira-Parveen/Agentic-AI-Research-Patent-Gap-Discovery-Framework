import json
from pathlib import Path
from typing import List, Tuple
from src.schemas.documents import DocumentMetadata, DocumentChunk, DocumentType
from src.config.constants import PROCESSED_PATENTS_DIR, PROCESSED_PAPERS_DIR


def save_processed_dataset(
    doc_type: DocumentType,
    metadata_list: List[DocumentMetadata],
    chunks_list: List[DocumentChunk]
) -> None:
    target_dir = PROCESSED_PATENTS_DIR if doc_type == DocumentType.PATENT else PROCESSED_PAPERS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    meta_file = target_dir / "metadata.json"
    chunks_file = target_dir / "chunks.json"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in metadata_list], f, indent=2, ensure_ascii=False)

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in chunks_list], f, indent=2, ensure_ascii=False)


def load_processed_dataset(doc_type: DocumentType) -> Tuple[List[DocumentMetadata], List[DocumentChunk]]:
    target_dir = PROCESSED_PATENTS_DIR if doc_type == DocumentType.PATENT else PROCESSED_PAPERS_DIR
    meta_file = target_dir / "metadata.json"
    chunks_file = target_dir / "chunks.json"

    if not meta_file.exists() or not chunks_file.exists():
        return [], []

    with open(meta_file, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)
        metadata_list = [DocumentMetadata(**m) for m in meta_raw]

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_raw = json.load(f)
        chunks_list = [DocumentChunk(**c) for c in chunks_raw]

    return metadata_list, chunks_list
