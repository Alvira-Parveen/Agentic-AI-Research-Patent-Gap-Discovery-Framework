# AI-Based Patent & Research Gap Discovery using RAG + Multi-Agent AI

**PBL-3 Academic Research Prototype (CSP391 - Department of CSE, Sharda University)**  
*Developed by Alvira Parveen & Rohit*

An explainable, reproducible artificial intelligence system that accepts an AI/ML invention or research idea, semantically retrieves supporting evidence from patent and research paper corpora, performs patent claim and academic analysis, calculates a transparent multi-factor novelty assessment, and identifies white-space research/patent gaps.

---

## 📌 Three Experimental Modes Compared
1. **Mode A — Single LLM Baseline**: Evaluates ideas using parametric LLM memory without external retrieval.
2. **Mode B — LLM + RAG**: Dense vector retrieval over patent and paper corpora using a single RAG generation prompt.
3. **Mode C — Multi-Agent + RAG (Proposed)**: LangGraph-coordinated specialized agents (`Retrieval Agent` → `Patent Agent` & `Research Agent` → `Feature Comparison` → `Novelty Agent` → `Gap Discovery Agent` → `Report Agent`).

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (defaults to offline MockProvider out of the box)
cp .env.example .env
```

### 2. Data Ingestion Pipeline
Processes the 20 patents and 20 research papers, runs Tesseract OCR on scanned documents, extracts sections (abstract, claims, description), normalizes whitespace, and deduplicates chunks.
```bash
python scripts/ingest_data.py
```

### 3. Build FAISS Vector Indexes
Embeds patent and paper chunks using Sentence Transformers (`all-MiniLM-L6-v2`) and builds L2-normalized cosine similarity FAISS vector stores.
```bash
python scripts/build_index.py
```

### 4. Run the Streamlit User Interface
Interactive dashboard supporting Idea Analysis, 3-Way Comparative Evaluation, Semantic Vector Search, and Benchmark Visualization.
```bash
streamlit run app/streamlit_app.py
```

### 5. Launch FastAPI Backend
High-performance REST API with interactive Swagger docs at `http://localhost:8000/docs`:
```bash
uvicorn src.main:app --reload --port 8000
```

### 6. Run Comparative Evaluation Benchmark
Runs all 3 modes over curated test cases and outputs Precision@K, Recall@K, F1@K, MRR, citation grounding, and latency.
```bash
python scripts/run_evaluation.py
```

### 7. Run Verification Test Suite
```bash
pytest tests/ -v
```

---

## 📁 Repository Structure
```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app/
│   └── streamlit_app.py           # Interactive Streamlit Web Dashboard
├── data/
│   ├── raw/                       # 20 Patents & 20 Research Papers
│   ├── processed/                 # Chunked & deduplicated JSON datasets
│   └── evaluation/                # 20 Curated test cases & annotations
├── vector_store/                  # FAISS indexes & metadata mapping
├── src/
│   ├── config/                    # Settings & constants
│   ├── schemas/                   # Pydantic schemas for all data models
│   ├── ingestion/                 # Dual-mode PDF loader (PyMuPDF + Tesseract OCR)
│   ├── preprocessing/             # Section-aware chunking & deduplication
│   ├── embeddings/                # Sentence Transformers embedding service
│   ├── retrieval/                 # FAISS vector store & ranking modules
│   ├── rag/                       # Context builder, generator, centralized prompts
│   ├── analysis/                  # Feature extraction, comparison, novelty & gaps
│   ├── agents/                    # Retrieval, Patent, Research, Novelty, Gap, Report agents
│   ├── workflows/                 # Single LLM, Standard RAG, and LangGraph Multi-Agent
│   ├── evaluation/                # Precision@K, Recall@K, F1@K, MRR, explainability
│   ├── llm/                       # Provider abstraction (OpenAI, MockProvider)
│   ├── utils/                     # Logging, stable IDs, timing profilers
│   └── main.py                    # FastAPI application
├── scripts/
│   ├── ingest_data.py             # Data ingestion CLI
│   ├── build_index.py             # Vector indexing CLI
│   ├── run_evaluation.py          # Benchmark execution CLI
│   └── inspect_dataset.py         # Corpus statistics CLI
├── tests/                         # Full automated test suite
└── docs/
    ├── architecture.md            # Detailed system design
    ├── methodology.md             # Transparent novelty & gap discovery formulas
    ├── dataset.md                 # Corpus and benchmark specifications
    ├── evaluation.md              # Experimental benchmark protocol
    ├── api.md                     # FastAPI endpoint documentation
    ├── literature_mapping.md      # Mapping of 20 papers & 20 patents to modules
    └── viva_explanation.md        # Defense and supervisor Q&A guide
```

---

## ⚖️ System Scope & Scientific Boundaries

> **Global Research Disclaimer**:  
> *Research prototype only. Results represent AI-assisted technical overlap analysis against the retrieved local corpus. They are not a legal patentability opinion, exhaustive prior-art search, or guarantee of novelty.*

### What the system DOES
- Retrieves related patent and research evidence from a local 40-document seed corpus using dense vector embeddings (FAISS).
- Performs semantic feature comparison between user idea features and candidate prior-art evidence.
- Analyzes patent claims and specifications to extract technical problem-solution representations.
- Analyzes research papers to identify algorithmic methodologies, empirical results, and reported limitations.
- Performs preliminary prior-art overlap analysis via a transparent, multi-factor engineering heuristic.
- Identifies potential corpus-level research gaps where technical features are disclosed in retrieved patent evidence but underrepresented in retrieved research papers.
- Identifies potential patent white-space signals where concepts are explored in retrieved research papers but underrepresented in retrieved patent claims.
- Generates an explainable report with verified provenance citations grounded in retrieved chunks.
- Compares three architectures (Single LLM, Standard RAG, Multi-Agent RAG) across computational latency and software integration diagnostics.

### What the system DOES NOT DO
- Determine legal patentability under 35 U.S.C. §§ 101, 102, 103 or international patent statutes.
- Guarantee novelty or inventiveness.
- Perform an exhaustive global prior-art search (queries are strictly evaluated against the local 40-document seed corpus).
- Prove absence of prior art (absence in the local corpus indicates technical underrepresentation and does not establish global absence).
- Establish patent infringement or freedom-to-operate.
- Establish legal obviousness or inventive step.
- Provide a legal opinion or substitute for licensed patent attorney counsel.
- Prove Multi-Agent RAG superiority (comparative performance remains to be established using independently annotated evaluation data).
