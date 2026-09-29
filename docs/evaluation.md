# Evaluation Framework & Experimental Protocol

## 1. Research Question
This project experimentally investigates:
> *"Does combining semantic retrieval, RAG, and specialized Multi-Agent reasoning provide better patent/research novelty and gap analysis than a standalone LLM or a basic LLM+RAG pipeline?"*

To answer this, three systems are evaluated on identical test cases:
- **System A**: Single LLM Baseline (no retrieval)
- **System B**: LLM + RAG (single-prompt retrieval)
- **System C**: Multi-Agent + RAG (proposed LangGraph architecture)

---

## 2. Evaluation Metrics

### 2.1 Information Retrieval Metrics
- **Precision@K**:
  $$\text{Precision}@K = \frac{|\mathcal{R}_{retrieved}@K \cap \mathcal{R}_{relevant}|}{K}$$
- **Recall@K**:
  $$\text{Recall}@K = \frac{|\mathcal{R}_{retrieved}@K \cap \mathcal{R}_{relevant}|}{|\mathcal{R}_{relevant}|}$$
- **F1@K**:
  $$\text{F1}@K = 2 \cdot \frac{\text{Precision}@K \cdot \text{Recall}@K}{\text{Precision}@K + \text{Recall}@K}$$
- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the position of the first relevant prior-art document.

### 2.2 Explainability & Hallucination Metrics
- **Citation Grounding Ratio**:
  $$\text{Grounding Ratio} = \frac{|\text{Valid Citations Matching Retrieved Documents}|}{|\text{Total Generated Citations}|}$$
- **Hallucinated Citations Count**: Any citation referencing documents not actually present in the retrieved set.

### 2.3 Novelty Assessment Agreement
Measures categorical concordance between AI predicted novelty level (`Low`, `Moderate`, `High`) and human ground-truth evaluations.

### 2.4 System Efficiency
Measures elapsed execution time (seconds) per test case across each architecture.

---

## 3. Running the Benchmark
```bash
.venv/bin/python scripts/run_evaluation.py
```
Outputs are serialized to `data/evaluation/benchmark_results.json`.
