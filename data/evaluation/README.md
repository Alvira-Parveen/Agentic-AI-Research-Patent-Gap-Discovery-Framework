# PBL-3 Evaluation Dataset and Benchmark Guidelines

## 1. Overview
This dataset contains 20 curated AI/ML invention and research test cases designed to evaluate the three experimental systems:
- **System A**: Single LLM Baseline (no retrieval)
- **System B**: Standard LLM + RAG (single-prompt retrieval)
- **System C**: Multi-Agent + RAG (proposed LangGraph framework)

## 2. Schema Specification
Each record in `test_cases.json` includes:
- `case_id`: Unique identifier (e.g. `TC-001`)
- `idea_title`: Concise title of the proposed invention
- `idea_description`: Natural-language problem and proposed solution
- `domain`: Specific AI/ML sub-domain
- `technical_features`: List of technical elements defining the architecture
- `relevant_patents`: Ground-truth list of relevant patents in the PBL-3 corpus (e.g. `PAT-003`)
- `relevant_papers`: Ground-truth list of relevant research papers (e.g. `PAP-019`)
- `human_assessment`: Qualitative human expert assessment (`TODO` where pending)
- `expected_novelty`: Expected novelty level: `Low`, `Moderate`, `High` (`TODO` where pending)
- `expected_gap`: Specific technical gap identified by domain experts (`TODO` where pending)
- `notes`: Metadata regarding calibration
- `evidence`: Specific citation anchors

## 3. Ground-Truth Policy (Anti-Fabrication Rule)
In strict compliance with academic research standards:
- Ground-truth annotations are marked as `TODO` until verified by human patent/technical experts.
- Automated benchmarks report `"Evaluation unavailable — ground truth not yet provided"` for unannotated cases rather than fabricating artificial scores.

## 4. Evaluation Metrics Computed
1. **Retrieval**:
   - `Precision@K` ($K \in \{1, 3, 5\}$)
   - `Recall@K`
   - `F1@K`
   - `Mean Reciprocal Rank (MRR)`
2. **Novelty Assessment**:
   - Agreement rate with human evaluation
3. **Explainability & Grounding**:
   - Citation validity ratio
   - Hallucination detection rate (citing non-retrieved documents)
4. **Efficiency**:
   - End-to-end execution latency (seconds)
