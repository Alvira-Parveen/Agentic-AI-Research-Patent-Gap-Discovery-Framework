# AI-Based Patent & Research Gap Discovery using RAG + Multi-Agent AI

**PBL-3 Academic Research Prototype (CSP391 - Department of CSE, Sharda University)**  
*Developed by Alvira Parveen & Rohit*

An explainable, reproducible artificial intelligence system that accepts an AI/ML invention or research idea, semantically retrieves supporting evidence from patent and research paper corpora, performs patent claim and academic analysis, calculates a transparent multi-factor novelty assessment, and identifies white-space research/patent gaps.

---

## 🎯 Foundational Baselines (Base Paper & Base Patent)

| Category | Reference | Core Contribution | How Our PBL-3 Project Extends It |
|---|---|---|---|
| **Base Research Paper** | **Paper 4 — EVOPAT**<br>*(A Multi-LLM-Based Patents Summarization & Analysis Framework)* | Uses LLMs for patent summarization, claim structuring, and document breakdown. | EVOPAT is single-corpus (patents only) and summarization-focused. Our framework introduces **dual-corpus cross-referencing (Patents + Research Papers)** and **autonomous gap discovery**. |
| **Base Patent** | **Patent 12 — US 12,602,412 B2**<br>*(Method & System for Optimizing RAG Pipelines in Generative AI)* | Defines multi-model RAG architecture, chunking, vector DBs, and evaluation. | Patent 12 is a generalized RAG pipeline. Our system implements **specialized LangGraph multi-agent orchestration** and **deterministic novelty/white-space gap algorithms**. |

---

## 🚦 Implementation Status: 100% Complete (Step-by-Step Breakdown)

| Step | Component / Module | Implementation Details | Status |
|---|---|---|---|
| **Step 1** | **Data Ingestion & Extraction** | Dual-mode PDF parser (PyMuPDF + Tesseract OCR fallback), section-aware extraction, metadata extraction across 20 patents and 20 research papers. | ✅ **100% Done** |
| **Step 2** | **Preprocessing & Chunking** | Section-aware chunking (Abstract, Claims, Description, Methodology), whitespace normalization, MD5 deduplication (5,593 total chunks). | ✅ **100% Done** |
| **Step 3** | **Embeddings & Vector Indexing** | Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim), L2-normalized FAISS cosine similarity indexes for patents and papers. | ✅ **100% Done** |
| **Step 4** | **Retrieval Engine** | Isolated patent & paper retrievers, top-k ranking, query expansion, metadata filtering. | ✅ **100% Done** |
| **Step 5** | **Agentic Architecture (LangGraph)** | 7 specialized agents: `RetrievalAgent`, `PatentAgent`, `ResearchAgent`, `FeatureComparison`, `NoveltyAgent`, `GapDiscoveryAgent`, `ReportAgent`. | ✅ **100% Done** |
| **Step 6** | **3-Way Comparative Evaluation** | Mode A (Single LLM Baseline) vs Mode B (LLM + RAG) vs Mode C (Multi-Agent RAG). | ✅ **100% Done** |
| **Step 7** | **Novelty & Gap Analysis Engines** | Transparent engineering heuristic for novelty overlap scoring + corpus-level cross-literature gap identification. | ✅ **100% Done** |
| **Step 8** | **REST API (FastAPI)** | Full backend with `/analyze`, `/compare`, `/search`, `/benchmark`, `/corpus/stats` endpoints and OpenAPI/Swagger documentation. | ✅ **100% Done** |
| **Step 9** | **User Interface (Streamlit)** | Multi-tab interactive UI with **Citation & Source Passage Traceability Explorer** and **Multi-Agent Evidence Disclosures**. | ✅ **100% Done** |
| **Step 10**| **Benchmark & Automated Testing** | 20 test cases benchmarked across all 3 modes + 34 passing `pytest` test suites. | ✅ **100% Done** |
| **Step 11**| **End-to-End Workflow & Traceability** | Automated 6-point verification audit (`verify_end_to_end_workflow.py` 6/6 passed) + passage-anchored synthesis. | ✅ **100% Done** |

---

## 🖥️ 7-Step Interactive Prototype Demonstration

The system is ready for live evaluator demonstration in offline mode. For a complete script with defensive responses, refer to the [Demonstration Script & Viva Guide](docs/DEMONSTRATION_SCRIPT.md).

```bash
# Launch interactive demonstration
source .venv/bin/activate
streamlit run streamlit_app.py
```

1. **System Environment & Status**: Verify the sidebar displays `🟢 Full Mode (Local)` and `Offline Mock (Passage-Anchored RAG)`. Notice the explicit disclosure: `(Live API pending credit quota)`.
2. **Submit Research Concept**: On Tab 1 ("🚀 Idea Analysis"), enter an invention or research idea (e.g., *"An AI system that uses drone-based multispectral images to detect crop diseases and recommend variable-rate chemical treatment"*).
3. **Run Multi-Agent RAG**: Select `Mode C: Multi-Agent + RAG` and click **"Analyze Idea"**.
4. **Inspect Executive Summary & Novelty Heuristic**: Review the preliminary prior-art overlap assessment (`0.4557`, `Moderate`). Note the explicit disclaimer that this is a technical overlap heuristic, not a legal patentability opinion.
5. **Explore Citation & Passage Traceability**: Expand references under **"🔗 Citation & Source Passage Traceability Explorer"** to inspect verified Chunk IDs (e.g. `PAT-004-C138`), cosine similarities, and verbatim excerpts from the source document.
6. **Inspect Upstream Agent Disclosures**: Review the *Patent Analysis Agent Output* and *Research Analysis Agent Output* tabs to see how features and excerpts were structured before report generation.
7. **Examine Research White Space & Boundaries**: Inspect underrepresented features and unexplored combinations, observing the explicit corpus limitation statement: findings reflect white space in the **40-document seed corpus**, not global prior art absence.

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
├── Patents/                       # 20 Foundational Patent PDF Documents
├── Research Papers/               # 20 Foundational Research Paper PDF Documents
├── data/
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

---

## 🔍 Critical Methodological Clarifications

### 1. Prototype Novelty Heuristic vs. Legal Patentability
The novelty score (e.g., `0.4557`, `Moderate`) is an automated engineering heuristic combining cosine semantic similarity (40%), technical feature coverage (35%), and patent claim overlap (25%) against the 40-document seed corpus. It is **NOT** a statutory patentability determination, a probability of allowance by patent examiners, or a guarantee of commercial novelty.

### 2. Citation Provenance vs. Semantic Support
The reported **99.13% citation provenance** metric (228 / 230 citations) measures that cited document IDs existed within the query's retrieved candidate pool in the benchmark. Provenance verifies retrieval presence, **not** semantic claim entailment. To guarantee semantic fidelity, users must inspect the verified chunk excerpts displayed in the **Citation & Source Passage Traceability Explorer**.

### 3. Live LLM Provider Status
During end-to-end audit testing with `OpenAIProvider` (`gpt-4o-mini`), the external API returned HTTP 429 (`insufficient_quota` / `credit_balance_exhausted`), as documented in `data/evaluation/real_llm_test_record.json`. The current demonstration runs with a deterministic, passage-anchored synthesizer that ingests exact chunk IDs, excerpts, and extracted features, eliminating cross-domain hallucinations in controlled testing. Live LLM validation remains pending external account credit renewal.

