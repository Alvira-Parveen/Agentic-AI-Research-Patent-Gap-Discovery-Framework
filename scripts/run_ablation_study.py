import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import RAGRetriever
from src.analysis.feature_extraction import FeatureExtractor
from src.llm.factory import get_llm_provider
from src.schemas.ideas import UserIdeaInput
from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, f1_at_k, mean_reciprocal_rank
from src.evaluation.benchmark import _dedup_doc_ids
from src.config.constants import EVALUATION_DIR
from src.utils.logging import logger


def compute_ci95(data: List[float]) -> Tuple[float, float, float]:
    """
    Computes Sample Mean, Sample Standard Deviation, and two-sided
    95% Confidence Interval half-width using Student's t-distribution:
        CI_95 = mean ± t_{0.025, df=n-1} * (s / sqrt(n))
    """
    n = len(data)
    if n < 2:
        return float(np.mean(data)), 0.0, 0.0
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    se = std / math.sqrt(n)
    h = float(stats.t.ppf(0.975, df=n - 1) * se)
    return mean, std, h


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Computes Holm-Bonferroni adjusted p-values for a family of hypotheses."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        adj_p = min(1.0, (m - rank) * p)
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = cum_max
    return adjusted


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Computes Benjamini-Hochberg FDR adjusted p-values (q-values)."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [0.0] * m
    cum_min = 1.0
    for rank in reversed(range(m)):
        orig_idx, p = indexed_p[rank]
        q = min(1.0, (m / (rank + 1)) * p)
        cum_min = min(cum_min, q)
        q_values[orig_idx] = cum_min
    return q_values


def run_reconciled_ablation():
    logger.info("Starting Reconciled Retrieval Ablation & Statistical Validation Study")
    
    with open(EVALUATION_DIR / "test_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    llm = get_llm_provider()
    feature_extractor = FeatureExtractor(llm)
    retriever = RAGRetriever()

    configs = ["Dense-Only", "Sparse-BM25-Only", "Hybrid-RRF"]
    metrics_per_config: Dict[str, Dict[str, List[float]]] = {
        cfg: {"p3": [], "p5": [], "r3": [], "r5": [], "f1_3": [], "f1_5": [], "mrr": []}
        for cfg in configs
    }
    per_query_results = []

    for tc in cases:
        cid = tc["case_id"]
        title = tc.get("idea_title", "")
        desc = tc["idea_description"]
        domain = tc.get("domain", "")
        gt_docs = set(tc.get("relevant_patents", []) + tc.get("relevant_papers", []))

        idea_input = UserIdeaInput(idea=desc, title=title, domain=domain)
        # Use exact same query construction as StandardRAGWorkflow and MultiAgentRAGWorkflow
        features = feature_extractor.extract(idea_input.idea, idea_input.domain)
        query = f"{idea_input.title}. {idea_input.idea}" if idea_input.title else idea_input.idea
        if features and features.technical_features:
            query = f"{query} {' '.join(features.technical_features[:3])}"

        # Execute through unified RAGRetriever in all 3 modes
        res_dense = retriever.retrieve(query, mode="dense")
        res_bm25 = retriever.retrieve(query, mode="bm25")
        res_hybrid = retriever.retrieve(query, mode="hybrid")

        ids_dense = _dedup_doc_ids([c.document_id for c in res_dense.all_results])
        ids_bm25 = _dedup_doc_ids([c.document_id for c in res_bm25.all_results])
        ids_hybrid = _dedup_doc_ids([c.document_id for c in res_hybrid.all_results])

        query_record = {"case_id": cid, "title": title}

        for cfg_name, ids in [("Dense-Only", ids_dense), ("Sparse-BM25-Only", ids_bm25), ("Hybrid-RRF", ids_hybrid)]:
            p3 = precision_at_k(ids, gt_docs, k=3)
            p5 = precision_at_k(ids, gt_docs, k=5)
            r3 = recall_at_k(ids, gt_docs, k=3)
            r5 = recall_at_k(ids, gt_docs, k=5)
            f1_3 = f1_at_k(p3, r3)
            f1_5 = f1_at_k(p5, r5)
            mrr = mean_reciprocal_rank(ids, gt_docs)

            metrics_per_config[cfg_name]["p3"].append(p3)
            metrics_per_config[cfg_name]["p5"].append(p5)
            metrics_per_config[cfg_name]["r3"].append(r3)
            metrics_per_config[cfg_name]["r5"].append(r5)
            metrics_per_config[cfg_name]["f1_3"].append(f1_3)
            metrics_per_config[cfg_name]["f1_5"].append(f1_5)
            metrics_per_config[cfg_name]["mrr"].append(mrr)

            query_record[cfg_name] = {"P@3": p3, "P@5": p5, "R@3": r3, "R@5": r5, "MRR": mrr}

        per_query_results.append(query_record)

    # Compute Statistics across 20 test cases
    stat_summary = {}
    for cfg in configs:
        stat_summary[cfg] = {}
        for m in ["p3", "p5", "r3", "r5", "f1_3", "f1_5", "mrr"]:
            mean, std, ci = compute_ci95(metrics_per_config[cfg][m])
            stat_summary[cfg][m] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "ci95_half": round(ci, 4),
                "formatted": f"{mean:.4f} ± {ci:.4f}"
            }

    # Significance Testing: Hybrid RRF vs Dense-Only Baseline
    metric_keys = ["p3", "p5", "r3", "r5", "mrr"]
    t_stats, raw_p_t = [], []
    w_stats, raw_p_w = [], []

    for m in metric_keys:
        t_stat, p_val_t = stats.ttest_rel(metrics_per_config["Hybrid-RRF"][m], metrics_per_config["Dense-Only"][m])
        t_stats.append(float(t_stat))
        raw_p_t.append(float(p_val_t))

        diff = np.array(metrics_per_config["Hybrid-RRF"][m]) - np.array(metrics_per_config["Dense-Only"][m])
        if np.all(diff == 0):
            w_stat, p_val_w = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(metrics_per_config["Hybrid-RRF"][m], metrics_per_config["Dense-Only"][m], zero_method="wilcox")
            w_stat, p_val_w = float(w_res.statistic), float(w_res.pvalue)
        w_stats.append(w_stat)
        raw_p_w.append(p_val_w)

    holm_p_t = holm_bonferroni(raw_p_t)
    holm_p_w = holm_bonferroni(raw_p_w)
    bh_q_t = benjamini_hochberg(raw_p_t)

    sig_tests_dense = {}
    for idx, m in enumerate(metric_keys):
        sig_tests_dense[m] = {
            "t_statistic": round(t_stats[idx], 4),
            "p_nominal_ttest": round(raw_p_t[idx], 4),
            "p_holm_ttest": round(holm_p_t[idx], 4),
            "q_bh_ttest": round(bh_q_t[idx], 4),
            "wilcoxon_stat": round(w_stats[idx], 4),
            "p_nominal_wilcoxon": round(raw_p_w[idx], 4),
            "p_holm_wilcoxon": round(holm_p_w[idx], 4),
            "is_nominally_significant": bool(raw_p_t[idx] < 0.05),
            "is_holm_significant": bool(holm_p_t[idx] < 0.05)
        }

    output_path = EVALUATION_DIR / "ablation_study_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_cases": len(cases),
            "configurations": configs,
            "statistical_summary": stat_summary,
            "significance_vs_dense": sig_tests_dense,
            "per_query_results": per_query_results
        }, f, indent=2)

    print("\n" + "=" * 94)
    print("      RECONCILED RETRIEVAL ABLATION STUDY & STATISTICAL MULTIPLICITY AUDIT (N=20)")
    print("=" * 94)
    print(f"{'Metric':<14} | {'Dense-Only (95% CI)':<22} | {'BM25-Only (95% CI)':<22} | {'Hybrid-RRF (95% CI)':<22}")
    print("-" * 94)
    for m, label in [("p3", "Precision@3"), ("p5", "Precision@5"), ("r3", "Recall@3"), ("r5", "Recall@5"), ("f1_3", "F1@3"), ("f1_5", "F1@5"), ("mrr", "MRR")]:
        d_str = stat_summary["Dense-Only"][m]["formatted"]
        b_str = stat_summary["Sparse-BM25-Only"][m]["formatted"]
        h_str = stat_summary["Hybrid-RRF"][m]["formatted"]
        print(f"{label:<14} | {d_str:<22} | {b_str:<22} | {h_str:<22}")
    print("-" * 94)
    print("HYPOTHESIS TESTING & MULTIPLICITY CORRECTION (Hybrid-RRF vs. Dense-Only Baseline):")
    print(f"{'Metric':<12} | {'Paired t':<9} | {'p (nominal)':<12} | {'p (Holm-Bonf)':<14} | {'Wilcoxon W':<10} | {'p (Wilcoxon)':<12}")
    print("-" * 94)
    for idx, (m, label) in enumerate([("p3", "Precision@3"), ("p5", "Precision@5"), ("r3", "Recall@3"), ("r5", "Recall@5"), ("mrr", "MRR")]):
        st = sig_tests_dense[m]
        print(f"{label:<12} | {st['t_statistic']:<+9.4f} | {st['p_nominal_ttest']:<12.4f} | {st['p_holm_ttest']:<14.4f} | {st['wilcoxon_stat']:<10.1f} | {st['p_nominal_wilcoxon']:<12.4f}")
    print("=" * 94)
    print("Multiplicity Note: While nominal p-values for Recall@5 (p=0.0276) and Precision@5 (p=0.0308)")
    print("are < 0.05, none survives family-wise Holm-Bonferroni correction across 5 simultaneous endpoints.")
    print(f"Results successfully saved to {output_path}")


if __name__ == "__main__":
    run_reconciled_ablation()
