import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import logger
from src.schemas.documents import DocumentType
from src.ingestion.patent_loader import PatentLoader
from src.ingestion.paper_loader import PaperLoader
from src.preprocessing.chunker import create_section_aware_chunks
from src.preprocessing.deduplicator import deduplicate_chunks
from src.ingestion.metadata import save_processed_dataset
from src.config.constants import RAW_PATENTS_DIR, RAW_PAPERS_DIR


def run_ingestion():
    logger.info("==================================================")
    logger.info("Starting PBL-3 Data Ingestion Pipeline")
    logger.info("==================================================")

    patent_files = sorted(list(RAW_PATENTS_DIR.glob("*.pdf")), key=lambda p: [int(s) if s.isdigit() else s for s in re.findall(r'\d+|\D+', p.name)]) if RAW_PATENTS_DIR.exists() else []
    paper_files = sorted(list(RAW_PAPERS_DIR.glob("*.pdf")), key=lambda p: [int(s) if s.isdigit() else s for s in re.findall(r'\d+|\D+', p.name)]) if RAW_PAPERS_DIR.exists() else []

    logger.info(f"Documents found: Patents={len(patent_files)}, Papers={len(paper_files)}")

    patent_loader = PatentLoader()
    paper_loader = PaperLoader()

    # Ingest Patents
    patent_meta_list = []
    patent_chunks_list = []
    patent_duplicates = 0
    patent_errors = 0

    logger.info("Ingesting Patent Corpus...")
    for idx, p_file in enumerate(patent_files, 1):
        try:
            meta, sections = patent_loader.parse_patent(str(p_file), idx)
            patent_meta_list.append(meta)
            chunks = create_section_aware_chunks(
                doc_id=meta.document_id,
                doc_type=DocumentType.PATENT,
                doc_title=meta.title,
                source=meta.source,
                sections=sections
            )
            deduped, removed = deduplicate_chunks(chunks)
            patent_chunks_list.extend(deduped)
            patent_duplicates += removed
            logger.info(f"  [{idx}/{len(patent_files)}] {meta.document_id}: {meta.title[:60]}... ({len(deduped)} chunks)")
        except Exception as e:
            logger.error(f"  Error processing patent {p_file.name}: {e}")
            patent_errors += 1

    save_processed_dataset(DocumentType.PATENT, patent_meta_list, patent_chunks_list)

    # Ingest Papers
    paper_meta_list = []
    paper_chunks_list = []
    paper_duplicates = 0
    paper_errors = 0

    logger.info("Ingesting Research Papers Corpus...")
    for idx, p_file in enumerate(paper_files, 1):
        try:
            meta, sections = paper_loader.parse_paper(str(p_file), idx)
            paper_meta_list.append(meta)
            chunks = create_section_aware_chunks(
                doc_id=meta.document_id,
                doc_type=DocumentType.PAPER,
                doc_title=meta.title,
                source=meta.source,
                sections=sections
            )
            deduped, removed = deduplicate_chunks(chunks)
            paper_chunks_list.extend(deduped)
            paper_duplicates += removed
            logger.info(f"  [{idx}/{len(paper_files)}] {meta.document_id}: {meta.title[:60]}... ({len(deduped)} chunks)")
        except Exception as e:
            logger.error(f"  Error processing paper {p_file.name}: {e}")
            paper_errors += 1

    save_processed_dataset(DocumentType.PAPER, paper_meta_list, paper_chunks_list)

    # Print summary statistics
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Documents found:")
    print(f"  Patents: {len(patent_files)}")
    print(f"  Papers:  {len(paper_files)}")
    print(f"\nChunks created:")
    print(f"  Patent chunks: {len(patent_chunks_list)}")
    print(f"  Paper chunks:  {len(paper_chunks_list)}")
    print(f"\nDuplicates removed:")
    print(f"  Patents: {patent_duplicates}")
    print(f"  Papers:  {paper_duplicates}")
    print(f"\nErrors encountered:")
    print(f"  Patents: {patent_errors}")
    print(f"  Papers:  {paper_errors}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    import re
    run_ingestion()
