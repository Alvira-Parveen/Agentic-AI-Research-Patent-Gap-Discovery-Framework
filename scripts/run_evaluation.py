import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import BenchmarkRunner
from src.config.settings import settings
from src.utils.logging import logger


def format_val(val, fmt=".4f", na_str="N/A"):
    if val is None:
        return na_str
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="PBL-3 Comparative Evaluation Benchmark")
    parser.add_argument(
        "--max-cases",
        type=str,
        default="20",
        help="Number of test cases to evaluate (e.g. 5, 20, or 'all'). Default is 20."
    )
    parser.add_argument("--provider", type=str, default=None, help="Override LLM provider (e.g., 'mock', 'openai')")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model name")
    args = parser.parse_args()

    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.llm_model = args.model

    logger.info("==================================================================")
    logger.info(f"Starting PBL-3 Comparative Evaluation Benchmark (max_cases={args.max_cases})")
    logger.info(f"Provider: {settings.llm_provider} | Model: {settings.llm_model}")
    logger.info("==================================================================")

    runner = BenchmarkRunner()
    max_cases_arg = "all" if args.max_cases.lower() in ["all", "max"] else int(args.max_cases)
    summary = runner.run_benchmark(max_cases=max_cases_arg)

    if "error" in summary:
        print(f"Benchmark Error: {summary['error']}")
        sys.exit(1)

    meta = summary.get("execution_metadata", {})
    num_cases = summary.get("num_cases_evaluated", 0)
    s_llm = summary.get("single_llm", {})
    rag = summary.get("rag", {})
    ma_rag = summary.get("multi_agent_rag", {})

    print("\n" + "=" * 82)
    print("           PBL-3 COMPARATIVE BENCHMARK EVALUATION RESULTS")
    print("=" * 82)
    print(f"Benchmark Type        : {meta.get('benchmark_type', 'software_integration')}")
    print(f"Diagnostic Label      : {meta.get('diagnostic_label', 'Provisional retrieval diagnostics against developer-curated targets')}")
    print(f"Human Validated       : {meta.get('human_validated', False)}")
    print(f"LLM Provider / Model  : {meta.get('llm_provider')} / {meta.get('model_name')}")
    print(f"Evaluation Cases      : {num_cases} (from {meta.get('test_cases_file')})")
    print(f"Annotation Status     : {meta.get('dataset_annotation_status')}")
    print(f"Execution Timestamp   : {meta.get('execution_timestamp')}")
    print("-" * 82)
    print(f"{'Metric':<36} | {'Single LLM (A)':<14} | {'LLM + RAG (B)':<14} | {'Multi-Agent (C)':<14}")
    print("-" * 82)

    lat_a = f"{format_val(s_llm.get('avg_latency_seconds'), '.2f')}s"
    lat_b = f"{format_val(rag.get('avg_latency_seconds'), '.2f')}s"
    lat_c = f"{format_val(ma_rag.get('avg_latency_seconds'), '.2f')}s"
    print(f"{'Avg Latency':<36} | {lat_a:<14} | {lat_b:<14} | {lat_c:<14}")

    p3_a = "N/A (no ret)"
    p3_b = format_val(rag.get("avg_precision@3_provisional"), ".4f")
    p3_c = format_val(ma_rag.get("avg_precision@3_provisional"), ".4f")
    print(f"{'Precision@3 (Prov. Diagnostic)':<36} | {p3_a:<14} | {p3_b:<14} | {p3_c:<14}")

    r3_a = "N/A (no ret)"
    r3_b = format_val(rag.get("avg_recall@3_provisional"), ".4f")
    r3_c = format_val(ma_rag.get("avg_recall@3_provisional"), ".4f")
    print(f"{'Recall@3 (Prov. Diagnostic)':<36} | {r3_a:<14} | {r3_b:<14} | {r3_c:<14}")

    f1_a = "N/A (no ret)"
    f1_b = format_val(rag.get("avg_f1@3_provisional"), ".4f")
    f1_c = format_val(ma_rag.get("avg_f1@3_provisional"), ".4f")
    print(f"{'F1@3 (Prov. Diagnostic)':<36} | {f1_a:<14} | {f1_b:<14} | {f1_c:<14}")

    mrr_a = "N/A (no ret)"
    mrr_b = format_val(rag.get("mean_reciprocal_rank_provisional"), ".4f")
    mrr_c = format_val(ma_rag.get("mean_reciprocal_rank_provisional"), ".4f")
    print(f"{'MRR (Prov. Diagnostic)':<36} | {mrr_a:<14} | {mrr_b:<14} | {mrr_c:<14}")

    gr_a = "0.0 (param)"
    gr_b = format_val(rag.get("avg_grounding_ratio"), ".2f")
    gr_c = format_val(ma_rag.get("avg_grounding_ratio"), ".2f")
    print(f"{'Citation Grounding Ratio':<36} | {gr_a:<14} | {gr_b:<14} | {gr_c:<14}")

    gaps_a = "0.0"
    gaps_b = "0.0"
    gaps_c = format_val(ma_rag.get("avg_gaps_discovered"), ".1f")
    print(f"{'Avg Gaps Discovered':<36} | {gaps_a:<14} | {gaps_b:<14} | {gaps_c:<14}")

    nov_a = format_val(s_llm.get("novelty_agreement_human"), ".4f", na_str="N/A (unannot)")
    nov_b = format_val(rag.get("novelty_agreement_human"), ".4f", na_str="N/A (unannot)")
    nov_c = format_val(ma_rag.get("novelty_agreement_human"), ".4f", na_str="N/A (unannot)")
    print(f"{'Novelty Agree (Human Ground Truth)':<36} | {nov_a:<14} | {nov_b:<14} | {nov_c:<14}")

    prov_a = format_val(s_llm.get("provisional_agreement_developer"), ".4f", na_str="N/A")
    prov_b = format_val(rag.get("provisional_agreement_developer"), ".4f", na_str="N/A")
    prov_c = format_val(ma_rag.get("provisional_agreement_developer"), ".4f", na_str="N/A")
    print(f"{'Novelty Agree (Dev Provisional)':<36} | {prov_a:<14} | {prov_b:<14} | {prov_c:<14}")

    print("=" * 82)
    print("PROVENANCE & QUALIFICATIONS:")
    print("1. Mode A (Single LLM) retrieval metrics are reported as N/A because Mode A is")
    print("   a purely parametric baseline that does not perform retrieval.")
    print("2. Precision, Recall, and MRR are provisional retrieval diagnostics evaluated against")
    print("   20 developer-curated targets (software integration check), NOT an independently")
    print("   validated gold standard or proof of system superiority.")
    print("3. Novelty agreement against human annotations reports N/A until independent")
    print("   double-blind annotation is conducted per docs/EVALUATION_PROTOCOL.md.")
    print("4. Comparative performance remains to be established using independently annotated")
    print("   evaluation data.")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
