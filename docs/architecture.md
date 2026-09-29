# System Architecture: AI-Based Patent & Research Gap Discovery

## 1. Executive Architecture Overview

The **AI-Based Patent and Research Gap Discovery Framework** is an explainable, multi-agent AI system designed to conduct early-stage technical novelty assessments and cross-corpus gap identification for AI/ML inventions.

```
                    ┌────────────────────────┐
                    │  USER INVENTION IDEA   │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Input Processing &   │
                    │   Feature Extraction   │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    RETRIEVAL AGENT     │
                    └─────┬────────────┬─────┘
                          │            │
             Dense Vector │            │ Dense Vector
             Query        │            │ Query
                          ▼            ▼
                 ┌──────────────┐┌──────────────┐
                 │    FAISS     ││    FAISS     │
                 │ Patent Index ││ Paper Index  │
                 └──────┬───────┘└──────┬───────┘
                        │               │
                        ▼               ▼
                 ┌──────────────┐┌──────────────┐
                 │ Top-K Patent ││ Top-K Paper  │
                 │   Evidence   ││   Evidence   │
                 └──────┬───────┘└──────┬───────┘
                        │               │
                        ▼               ▼
                 ┌──────────────┐┌──────────────┐
                 │ Patent Agent ││Research Agent│
                 └──────┬───────┘└──────┬───────┘
                        │               │
                        └───────┬───────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │  Technical Feature Comparator│
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │     NOVELTY AGENT            │
                 │ Multi-Factor Transparent POI │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │     GAP DISCOVERY AGENT      │
                 │ Patent vs Paper White Spaces │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │        REPORT AGENT          │
                 │ Synthesizes Grounded Summary │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │   Final Explainable Report   │
                 └──────────────────────────────┘
```

---

## 2. Component Descriptions

### 2.1 Input Processing & Feature Extraction (`src/analysis/feature_extraction.py`)
- Decomposes informal natural language idea descriptions into a standardized structured schema (`ExtractedFeatures`).
- Extracts title, domain, core technologies, technical mechanisms, procedural methods, input streams, output artifacts, and architectural subsystems.

### 2.2 Semantic Retrieval Layer (`src/retrieval/`)
- Uses dense Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) mapping 384-dimensional dense vectors.
- Embeddings are L2-normalized, enabling exact cosine similarity computation via FAISS Inner Product indexing (`IndexFlatIP`).
- Separate logical indexes are maintained for patents (`vector_store/patents/`) and research papers (`vector_store/papers/`).
- `rank_and_deduplicate` enforces document diversity by limiting chunks from a single document to prevent high-similarity clustering.

### 2.3 Multi-Agent LangGraph System (`src/workflows/multi_agent_workflow.py`)
State transitions are coordinated through a strongly-typed `MultiAgentState`:
1. **Retrieval Agent**: Enriches query with technical features and performs simultaneous multi-corpus retrieval.
2. **Patent Analysis Agent**: Scrutinizes patent specifications, independent claims, and limitation elements against the proposed features.
3. **Research Analysis Agent**: Evaluates academic methodology, experimental contributions, and reported empirical limits.
4. **Technical Feature Comparator**: Constructs an $N \times M$ cross-corpus feature coverage matrix.
5. **Novelty Agent**: Implements the multi-factor Prior-Art Overlap Index ($POI$).
6. **Gap Discovery Agent**: Highlights underrepresented technical features and less-explored cross-domain intersections.
7. **Report Agent**: Synthesizes the complete explainable final report with strict citation anchors.

---

## 3. The Three Experimental Modes

| Dimension | Mode A: Single LLM | Mode B: Standard LLM + RAG | Mode C: Multi-Agent + RAG |
|---|---|---|---|
| **Retrieval Corpus** | None (Parametric memory only) | Dual-corpus (Patents + Papers) | Dual-corpus (Patents + Papers) |
| **Reasoning Architecture** | Single zero-shot prompt | Single context-stuffed prompt | 6 Specialized collaborative agents |
| **Claim Decomposition** | No | Coarse / Unstructured | Yes (Independent claim mapping) |
| **Novelty Assessment** | None (Parametric baseline) | Prototype retrieval overlap signal | Transparent Multi-Factor Formula |
| **Gap Discovery** | Speculative | None / Unstructured | Cross-literature white-space matrix |
| **Hallucination Risk** | High | Low-Moderate | Low (Citations strictly anchored) |
| **Latency** | ~0.01s | ~0.02s - 0.05s | ~0.08s - 0.20s |

---

## 4. Research Prototype Disclaimer
> *Research prototype only. Results represent AI-assisted technical overlap analysis against the retrieved local corpus. They are not a legal patentability opinion, exhaustive prior-art search, or guarantee of novelty.*

