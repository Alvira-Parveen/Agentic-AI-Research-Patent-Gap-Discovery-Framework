import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas.documents import DocumentType
from src.ingestion.metadata import load_processed_dataset
from src.config.constants import RAW_PATENTS_DIR, RAW_PAPERS_DIR, VECTOR_STORE_DIR
import faiss


def inspect():
    print("\n" + "=" * 55)
    print("PBL-3 CORPUS & INDEX INSPECTION REPORT")
    print("=" * 55)

    raw_pat = list(RAW_PATENTS_DIR.glob("*.pdf")) if RAW_PATENTS_DIR.exists() else []
    raw_pap = list(RAW_PAPERS_DIR.glob("*.pdf")) if RAW_PAPERS_DIR.exists() else []

    print(f"Raw PDFs Available:")
    print(f"  Patents: {len(raw_pat)} files in {RAW_PATENTS_DIR}")
    print(f"  Papers:  {len(raw_pap)} files in {RAW_PAPERS_DIR}")

    pat_meta, pat_chunks = load_processed_dataset(DocumentType.PATENT)
    pap_meta, pap_chunks = load_processed_dataset(DocumentType.PAPER)

    print(f"\nProcessed Ingested Datasets:")
    print(f"  Patents Ingested: {len(pat_meta)} documents | {len(pat_chunks)} chunks")
    print(f"  Papers Ingested:  {len(pap_meta)} documents | {len(pap_chunks)} chunks")

    # Inspect FAISS indexes
    pat_idx_file = VECTOR_STORE_DIR / "patents" / "index.faiss"
    pap_idx_file = VECTOR_STORE_DIR / "papers" / "index.faiss"

    print(f"\nFAISS Vector Stores:")
    if pat_idx_file.exists():
        idx_p = faiss.read_index(str(pat_idx_file))
        print(f"  Patent Index: ACTIVE ({idx_p.ntotal} vectors, dimension={idx_p.d})")
    else:
        print(f"  Patent Index: NOT FOUND at {pat_idx_file}")

    if pap_idx_file.exists():
        idx_r = faiss.read_index(str(pap_idx_file))
        print(f"  Paper Index:  ACTIVE ({idx_r.ntotal} vectors, dimension={idx_r.d})")
    else:
        print(f"  Paper Index:  NOT FOUND at {pap_idx_file}")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    inspect()
