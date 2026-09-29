# PBL-3 Viva & Supervisor Defense Guide

This document provides clear, technically grounded answers to anticipated questions from project supervisors, examiners, and review committees for the **Agentic AI Research & Patent Gap Discovery Framework** (PBL-3 Lab CSP391).

---

### Q1: What is your project?
**Answer**: Our project is an AI-powered academic prototype that assists researchers and innovators during the early ideation stage. When a user inputs an invention or research idea, the system automatically retrieves related prior-art from both patent literature and scientific research papers, performs in-depth technical feature comparison, computes a preliminary novelty score, identifies potential research/patent gaps, and synthesizes an explainable report. Crucially, it experimentally compares three paradigms: **Single LLM**, **LLM + RAG**, and our proposed **Multi-Agent + RAG** framework.

---

### Q2: What problem are you solving?
**Answer**: Manual prior-art analysis is slow, expensive, and overwhelming. Innovators typically search either patent databases or scientific journals, but rarely cross-reference both efficiently. Standard keyword search misses semantic equivalents, while general-purpose LLMs hallucinate patent numbers and lack grounded evidence. We solve this by automating cross-corpus evidence retrieval, grounded feature alignment, and white-space gap discovery without black-box guessing.

---

### Q3: Why patents?
**Answer**: Patents represent legally protected, commercially applied technologies. They contain structured claims defining the exact boundaries of technological ownership. Evaluating an idea against patents protects innovators from duplicating existing commercial inventions and helps identify unpatented white spaces.

---

### Q4: Why research papers?
**Answer**: Research papers represent the cutting edge of scientific discovery and algorithmic development. Many cutting-edge ideas appear in academic papers (e.g. arXiv) years before they are commercialized or patented. Examining both patents and research papers provides a complete view of existing human knowledge.

---

### Q5: Why RAG (Retrieval-Augmented Generation)?
**Answer**: Standalone LLMs rely only on their static internal training weights (parametric memory). They cannot cite specific verified prior-art documents and often hallucinate citations. RAG injects retrieved, real-world patent and paper excerpts into the LLM context window, ensuring every claim in the output is backed by verifiable citations.

---

### Q6: Why Multi-Agent?
**Answer**: A single monolithic prompt trying to analyze patents, research papers, novelty scores, and gaps produces shallow, conflicting outputs. As demonstrated in *EvoPat* (Wang 2024) and *TCLMA* (Zheng 2025), patent analysis requires legal element-by-element parsing, while research analysis requires methodological scrutiny. Using specialized agents coordinated through LangGraph allows:
1. **Retrieval Agent** to focus purely on dense multi-corpus search.
2. **Patent Agent** to focus on claim scope and legal disclosure.
3. **Research Agent** to evaluate experimental findings and scientific limits.
4. **Novelty Agent** to compute transparent factor scores.
5. **Gap Agent** to spot cross-literature white spaces.
6. **Report Agent** to assemble an explainable final brief.

---

### Q7: Why embeddings?
**Answer**: Traditional keyword matching (lexical search) fails when different terms describe identical concepts (e.g., "UAV multispectral camera" vs "drone aerial spectral imaging"). Dense embeddings (`all-MiniLM-L6-v2`) project texts into a shared 384-dimensional geometric space where conceptual similarity is measured by vector proximity, enabling high-recall semantic retrieval.

---

### Q8: Why FAISS?
**Answer**: FAISS (Facebook AI Similarity Search) is the industry and academic standard for high-performance vector indexing. It provides exact, hardware-accelerated Inner Product (cosine similarity) search with zero external database server overhead, making our system lightweight, fast, and completely reproducible locally.

---

### Q9: What is your dataset?
**Answer**: We strictly distinguish between two datasets:
1. **Knowledge Corpus**: The 20 patents and 20 research papers manually collected for our PBL-3 semester work, covering patent intelligence, multi-agent ideation, and semantic retrieval.
2. **Evaluation Dataset**: 20 curated AI/ML invention test cases (`data/evaluation/test_cases.json`) used to benchmark Precision@K, Recall@K, F1@K, MRR, explainability, and response latency across all three experimental modes.

---

### Q10: How does novelty scoring work?
**Answer**: We reject arbitrary formulas like presenting raw `1 - similarity` as actual patent novelty. Instead, our prototype uses a transparent, configurable Prior-Art Overlap Index ($POI$):
$$POI = 0.40 \cdot \text{SemanticSimilarity} + 0.35 \cdot \text{TechnicalOverlap} + 0.25 \cdot \text{ClaimCoverage}$$
$$\text{Novelty Score} = 1.0 - (POI \cdot \text{EvidenceStrength})$$
The weights and similarity thresholds are configurable prototype parameters that have not been empirically validated on independent human ground truth. All factors and rationales are exposed to the user, and every result includes a strict disclaimer that this is an AI-assisted technical overlap heuristic against the retrieved local corpus, not a legal patentability opinion.

---

### Q11: How do you identify gaps?
**Answer**: Our Gap Discovery Agent constructs a matrix contrasting what patents disclose versus what research papers report in the retrieved evidence. When technical features are missing from both, or when a patent covers feature $A$ and a paper covers feature $B$ but their joint combination is absent from the local seed corpus, the agent flags this as a potential gap or white-space signal, explicitly noting that absence in the retrieved local corpus does not establish global absence.

---

### Q12: How do you evaluate the system?
**Answer**: We evaluate the three systems on 20 test cases using standard quantitative integration diagnostics:
- **Retrieval**: Precision@3, Recall@3, F1@3, and Mean Reciprocal Rank (MRR) reported as *provisional retrieval diagnostics against developer-curated targets* (software integration check, not proof of model superiority).
- **Explainability**: Citation Grounding Ratio (percentage of citations that match real retrieved documents from the vector store).
- **Latency**: True execution time in seconds.
Agreement against independent human ground truth is marked N/A until independent double-blind annotations are conducted.

---

### Q13: What is your research contribution?
**Answer**: Our contribution is an explainable, reproducible research prototype demonstrating how a multi-agent workflow coordinates specialized stages (dense retrieval, claim analysis, research paper analysis, semantic feature comparison, heuristic overlap assessment, and gap discovery) with verified citation provenance. Comparative performance claims remain to be established through independently annotated evaluation benchmarks.

---

### Q14: What does the system DO and NOT DO?
**What the system DOES**:
- Retrieves related patent and research evidence from the 40-document seed corpus.
- Performs semantic feature comparison against candidate prior-art text.
- Analyzes patent claims and research papers through specialized workflow stages.
- Computes a transparent, prototype prior-art overlap heuristic with factor breakdowns.
- Identifies potential corpus-level research gaps and patent white-space signals.
- Generates an explainable report with verified citations.

**What the system DOES NOT DO**:
- Determine legal patentability under 35 U.S.C. §§ 101, 102, 103.
- Guarantee novelty or inventiveness.
- Perform an exhaustive global prior-art search.
- Prove the absence of prior art globally.
- Provide a legal opinion or prove Multi-Agent RAG superiority.

