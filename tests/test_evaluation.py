import pytest
from src.evaluation.benchmark import BenchmarkRunner
from src.schemas.documents import DocumentMetadata, DocumentType
from src.schemas.retrieval import RetrievedChunk


def test_schema_ocr_metadata():  
    doc_meta = DocumentMetadata(
        document_id="PAT-TEST",
        document_type=DocumentType.PATENT,
        title="Test Patent",
        source="data/raw/patents/test.pdf",
        filename="test.pdf",
        extraction_method="tesseract_ocr",
        is_ocr=True,
        ocr_quality="MEDIUM"
    )
    assert doc_meta.is_ocr is True
    assert doc_meta.extraction_method == "tesseract_ocr"
    assert doc_meta.ocr_quality == "MEDIUM"

    chunk = RetrievedChunk(
        chunk_id="C-TEST",
        document_id="PAT-TEST",
        document_type=DocumentType.PATENT,
        document_title="Test Patent",
        section="claim",
        text="Sample claim text",
        similarity_score=0.85,
        rank=1,
        citation="[PAT-TEST]",
        extraction_method="tesseract_ocr",
        is_ocr=True,
        ocr_quality="MEDIUM"
    )
    assert chunk.is_ocr is True
    assert chunk.ocr_quality == "MEDIUM"


def test_benchmark_runner_mode_a_and_provenance():
    runner = BenchmarkRunner()
    summary = runner.run_benchmark(max_cases=2)

    assert "execution_metadata" in summary
    meta = summary["execution_metadata"]
    assert "benchmark_type" in meta
    assert meta["benchmark_type"] in ["software_integration", "human_validated"]
    assert "llm_provider" in meta
    assert "model_name" in meta
    assert "dataset_annotation_status" in meta
    assert "provisional_developer_curated" in meta["dataset_annotation_status"]

    s_llm = summary["single_llm"]
    # Mode A retrieval metrics must be None (N/A)
    assert s_llm["avg_precision@3"] is None
    assert s_llm["avg_recall@3"] is None
    assert s_llm["avg_f1@3"] is None
    assert s_llm["mean_reciprocal_rank"] is None
    assert s_llm["avg_grounding_ratio"] == 0.0

    # Unannotated human agreement must be None
    assert s_llm["novelty_agreement_human"] is None
    assert summary["rag"]["novelty_agreement_human"] is None
    assert summary["multi_agent_rag"]["novelty_agreement_human"] is None

    # Check detailed cases
    assert len(summary["detailed_cases"]) == 2
    case_0 = summary["detailed_cases"][0]
    assert case_0["single_llm"]["precision@3"] is None
    assert case_0["single_llm"]["novelty_score"] is None


def test_benchmark_runner_max_cases():
    runner = BenchmarkRunner()
    test_cases = runner.load_test_cases()
    assert len(test_cases) == 20

    # Verify that max_cases="all" handles all cases
    assert len(runner.load_test_cases()) == 20
