import re
from src.config.constants import PATENT_PREFIX, PAPER_PREFIX

def generate_document_id(doc_type: str, index: int) -> str:
    prefix = PATENT_PREFIX if "patent" in doc_type.lower() else PAPER_PREFIX
    return f"{prefix}-{index:03d}"

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}-C{chunk_index:03d}"

def format_citation(doc_id: str, section: str, page: int = None) -> str:
    sec_clean = section.replace("_", " ").title() if section else "General"
    if page:
        return f"[{doc_id}, {sec_clean}, p.{page}]"
    return f"[{doc_id}, {sec_clean}]"

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', '_', name)
