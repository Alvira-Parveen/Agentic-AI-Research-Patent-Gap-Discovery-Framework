# Dataset Specification: Knowledge Corpus & Evaluation Benchmark

## 1. Two Distinct Dataset Concepts

A fundamental principle of this project is separating the **Knowledge Corpus** from the **Evaluation Dataset**:

| Concept | Purpose | Size & Contents | Location |
|---|---|---|---|
| **A. Knowledge Corpus** | Searchable reference knowledge base for dense vector retrieval | 20 USPTO Patents + 20 Academic Research Papers | `data/raw/patents/`, `data/raw/papers/` |
| **B. Evaluation Dataset** | Verified test queries to empirically benchmark the 3 systems | 20 Curated AI/ML Invention Cases | `data/evaluation/test_cases.json` |

---

## 2. Knowledge Corpus Details

### 2.1 Research Papers (20 Papers)
Directly aligned with the literature survey in `PBL-3_Report.pdf`:
- Ingested from arXiv and academic journals.
- Formats: High-resolution digital text PDFs.
- Covered topics: Patent Intelligence, Multi-Agent Ideation, EvoPat, Patent Claim Correspondence, Graph RAG, Tree-of-Claims, TCLMA, IPAS-CARNA, Siamese Networks, Memory Graph, and RAG for Intellectual Property.

### 2.2 Patent Documents (20 Patents)
- Source: USPTO patent publications (Google Patents).
- Formats:
  - 5 Digital PDFs (`Patent 1, 2, 13, 14, 15`)
  - 15 Scanned USPTO publication PDFs (`Patent 3-12, 16-20`) processed via Tesseract OCR.
- Section Extraction: Front matter (Title, Patent Number, Date, Assignee, Abstract), Detailed Specification, and Numbered Claims.

---

## 3. Evaluation Dataset Schema

The 20 test cases in `data/evaluation/test_cases.json` follow this schema:
```json
{
  "case_id": "TC-001",
  "idea_title": "Drone Multispectral Crop Disease Detection and Prescription",
  "idea_description": "...",
  "domain": "Computer Vision & Smart Agriculture",
  "technical_features": [...],
  "relevant_patents": ["PAT-003", "PAT-015"],
  "relevant_papers": ["PAP-019"],
  "human_assessment": "...",
  "expected_novelty": "Moderate",
  "expected_gap": "...",
  "notes": "Anchor test case",
  "evidence": ["[PAT-003, Description]"]
}
```

Ground-truth policy: Missing annotations are explicitly tagged with `TODO` to prevent fabricated evaluations.
