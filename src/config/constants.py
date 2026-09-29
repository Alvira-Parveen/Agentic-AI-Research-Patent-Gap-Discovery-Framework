from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATENTS_DIR = DATA_DIR / "raw" / "patents"
RAW_PAPERS_DIR = DATA_DIR / "raw" / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_PATENTS_DIR = PROCESSED_DIR / "patents"
PROCESSED_PAPERS_DIR = PROCESSED_DIR / "papers"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
EVALUATION_DIR = DATA_DIR / "evaluation"

# Models & Algorithms
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_PROVIDER = "mock"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# ID Prefixes
PATENT_PREFIX = "PAT"
PAPER_PREFIX = "PAP"

# Default Chunking
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80

# Default Retrieval Counts
DEFAULT_TOP_K_PATENTS = 3
DEFAULT_TOP_K_PAPERS = 3

# Novelty Assessment Configurable Formula Weights (Prototype Heuristic)
DEFAULT_WEIGHT_SEMANTIC = 0.40
DEFAULT_WEIGHT_OVERLAP = 0.35
DEFAULT_WEIGHT_CLAIM = 0.25

# Configurable Prototype Similarity Thresholds (Initial Research Parameters, Not Empirically Validated)
DEFAULT_FEATURE_MATCH_THRESHOLD = 0.55
DEFAULT_FEATURE_PARTIAL_THRESHOLD = 0.38
DEFAULT_FEATURE_UNCERTAIN_THRESHOLD = 0.28

# OCR Quality Indicators
OCR_QUALITY_HIGH = "HIGH"
OCR_QUALITY_MEDIUM = "MEDIUM"
OCR_QUALITY_LOW = "LOW"
OCR_QUALITY_UNKNOWN = "UNKNOWN"

# Feature Match Types
MATCH_TYPE_MATCH = "MATCH"
MATCH_TYPE_PARTIAL = "PARTIAL_MATCH"
MATCH_TYPE_UNCERTAIN = "UNCERTAIN"
MATCH_TYPE_NO_MATCH = "NO_MATCH"

# Standard Disclaimers
GLOBAL_RESEARCH_DISCLAIMER = (
    "Research prototype only. Results represent AI-assisted technical overlap analysis against the retrieved local corpus. "
    "They are not a legal patentability opinion, exhaustive prior-art search, or guarantee of novelty."
)
LEGAL_DISCLAIMER = (
    "PRELIMINARY AI RESEARCH ASSESSMENT ONLY: This evaluation is an automated preliminary technical analysis "
    "based on the local seed corpus and does NOT constitute a legal patentability opinion, prior-art search guarantee, "
    "or official patent examination."
)
OCR_DISCLAIMER = "Evidence is OCR-derived and may contain transcription errors."


