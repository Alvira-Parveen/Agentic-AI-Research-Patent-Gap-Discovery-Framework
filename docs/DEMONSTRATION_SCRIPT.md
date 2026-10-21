# PBL-3 Prototype Demonstration Script & Viva Defense Guide

**Project**: AI-Based Patent & Research Gap Discovery using RAG + Multi-Agent AI  
**Course / Code**: CSP391 — Department of Computer Science & Engineering, Sharda University  
**Authors**: Alvira Parveen & Rohit  

---

## 🎯 Demonstration Purpose

This document provides a structured, reproducible 7-step demonstration script for live evaluator presentations, viva defenses, and technical reviews. It outlines the step-by-step actions in the Streamlit application, key screen highlights, and defensive technical responses to anticipated supervisor questions.

---

## 🚀 Step-by-Step Demonstration Sequence

### Step 1: Launch Application & Verify System Environment

```bash
# 1. Activate Python virtual environment
source .venv/bin/activate

# 2. Run the automated 6-step workflow audit (Optional verification check)
python scripts/verify_end_to_end_workflow.py

# 3. Launch Streamlit UI
streamlit run streamlit_app.py
```

- **URL**: Open browser to `http://localhost:8501`
- **Sidebar Check**:
  - **Execution Mode**: `🟢 Full Mode (Local)`
  - **Synthesizer Status**: `Offline Mock (Passage-Anchored RAG)`
  - **Live LLM Note**: `(Live OpenAI API pending credit quota)`
  - **Embeddings & Vector DB**: `sentence-transformers/all-MiniLM-L6-v2` / `FAISS (5,387 patent chunks, 722 paper chunks)`

---

### Step 2: Input Invention / Research Idea

- **Location**: Tab 1 ("🚀 Idea Analysis")
- **Input Text Area**: Provide representative concept:
  > *"An AI system that uses drone-based multispectral images to detect crop diseases and recommend variable-rate chemical treatment."*
- **Domain**: `Agriculture & Computer Vision`
- **Mode Selection**: Select `Mode C: Multi-Agent + RAG` (Invokes the 7-node LangGraph pipeline).
- **Action**: Click **"Analyze Idea"**.

---

### Step 3: Observe Multi-Agent Pipeline Execution

- The status spinner coordinates the execution through sequential LangGraph nodes:
  1. `Feature Extraction`: Parses core technologies, input sensors, algorithms, and technical outputs.
  2. `Retrieval Agent`: Queries FAISS dense vector indexes over patents and scientific literature.
  3. `Patent Analysis Agent`: Extracts technical problems, disclosed solutions, claim relevance, and chunk excerpts.
  4. `Research Analysis Agent`: Extracts scientific methodologies, empirical findings, and excerpts.
  5. `Feature Comparison`: Evaluates claim coverage across prior art.
  6. `Novelty Assessment Agent`: Evaluates multi-factor prior-art overlap heuristic.
  7. `Gap Discovery Agent`: Computes underrepresented features and white-space combinations.
  8. `Report Agent`: Synthesizes the passage-anchored executive summary.

---

### Step 4: Review Executive Summary & Preliminary Novelty Heuristic

- **Executive Summary Box**:
  - Highlights baseline prior art identified in the seed corpus (`[PAT-019]`, `[PAP-003]`, `[PAT-004]`).
  - Identifies documented capabilities (spectral filtering, edge detection, AI architectures).
  - Explicitly states unaddressed features: *"Hybrid CNN-transformer architectures combined with decentralized drone task scheduling remains unaddressed in the examined prior art."*
- **Preliminary Prior-Art Overlap Assessment**:
  - **Heuristic Score**: `0.4557` (or similar depending on query) | **Level**: `Moderate`
  - **Defensive Clarification**:
    > *"This score is an automated engineering heuristic reflecting textual overlap against the local 40-document seed corpus. It is NOT a legal patentability opinion, an official examination result, or a statistically validated novelty probability."*

---

### Step 5: Citation & Source Passage Traceability Explorer

- **Interactive Verification**:
  - Scroll to **"🔗 Citation & Source Passage Traceability Explorer"**.
  - Notice the green badge: `🟢 In Retrieval Pool`.
  - Expand `Reference [PAT-004]`:
    - **Document Title**: `Prior Art Specification for PAT-004`
    - **Verified Chunk ID**: `PAT-004-C138`
    - **Cosine Similarity**: e.g., `0.3704`
    - **Verbatim Excerpt**: *"Standard edge detection algorithms may include, but are not limited to, a Laplacian filter... In some embodiments, spectral analysis of the raster content can be used to identify..."*
  - Expand **"View Full Chunk Text"** to demonstrate complete unedited context.

---

### Step 6: Multi-Agent Upstream Evidence Disclosures

- **Upstream Agent Transparency**:
  - Switch between **"Patent Analysis Agent Output"** and **"Research Analysis Agent Output"**.
  - Show that upstream agents extract structured technical representations and bind exact `chunk_id`s and excerpts *before* the report synthesizer writes a single claim.
  - Demonstrates architectural prevention of hallucination: the generator is strictly constrained to the text excerpts passed to it.

---

### Step 7: Research Whitespace, Corpus Boundaries & Honest Disclaimers

- **Identified Whitespace & Gaps**:
  - *Underrepresented Features*: Multi-agent feedback loops between edge detection and prescription generation.
  - *Unexplored Combinations*: Hybrid CNN-transformer architectures combined with decentralized drone task scheduling.
- **Corpus Limitation Statement**:
  - Emphasize that white space reflects technical underrepresentation within the **40-document seed corpus (20 patents, 20 research papers)**, NOT global scientific absence or proof of patentability.
- **Support Policy**:
  - Conclude with the system policy: *"Claims made by the system that cannot be traced to a corresponding retrieved chunk are not established by the available evidence and must not be treated as verified prior art."*

---

## 🛡️ Viva & Reviewer Defense Q&A

### Q1: "Does your system prove that an idea is patentable or globally novel?"
> **Answer**:  
> No. Patentability is a statutory legal determination under 35 U.S.C. §§ 101, 102, and 103 requiring exhaustive searching across international patent office databases by licensed patent attorneys or examiners. Our prototype performs AI-assisted technical overlap analysis against a local 40-document seed corpus. The novelty score (`0.4557`, Moderate) is an engineering heuristic reflecting cosine and technical feature coverage, not a legal or empirical probability.

### Q2: "What does the 99.13% provenance score mean? Does it mean 99.13% of claims are true?"
> **Answer**:  
> No. Provenance measures whether a cited document ID existed in the query's retrieved candidate pool (228 out of 230 citations in the frozen benchmark). Provenance confirms retrieval validity, not semantic entailment. Semantic correctness requires passage-level verification, which is why we built the Citation Traceability Explorer and passage-anchored synthesis.

### Q3: "Why is the system currently running with an Offline Mock synthesizer?"
> **Answer**:  
> During real-LLM validation testing with `OpenAIProvider` (`gpt-4o-mini`), the external API returned HTTP 429 (`insufficient_quota` / `credit_balance_exhausted`), as formally recorded in `data/evaluation/real_llm_test_record.json`. We designed a deterministic, passage-anchored synthesizer that ingests exact chunk IDs, excerpts, and extracted features, eliminating cross-domain hallucinations in controlled testing. The architecture is fully pre-wired for live LLM execution as soon as account credits are replenished.

### Q4: "How does your multi-agent approach differ from standard RAG?"
> **Answer**:  
> Standard RAG (Mode B) dumps all retrieved text chunks into a single large generation prompt, risking ungrounded synthesis and hallucinated capabilities. Our Multi-Agent approach (Mode C) utilizes LangGraph to decompose the analysis into specialized agents: legal claim analysis, scientific methodology extraction, feature-by-feature matrix comparison, and passage-anchored synthesis, ensuring explainable attribution at every step.
