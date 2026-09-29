import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.schemas.ideas import UserIdeaInput
from src.workflows.orchestrator import IdeaAnalysisOrchestrator
from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, f1_at_k, mean_reciprocal_rank
from src.evaluation.novelty_metrics import evaluate_novelty_agreement
from src.evaluation.explainability import evaluate_explainability
from src.config.constants import EVALUATION_DIR
from src.config.settings import settings
from src.utils.logging import logger


class BenchmarkRunner:
    def __init__(self, orchestrator: Optional[IdeaAnalysisOrchestrator] = None):
        self.orchestrator = orchestrator or IdeaAnalysisOrchestrator()

    def load_test_cases(self, file_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        path = file_path or (EVALUATION_DIR / "test_cases.json")
        if not path.exists():
            logger.error(f"Test cases file not found at {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self, max_cases: Union[int, str] = 20, test_cases_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Runs comparative evaluation benchmark across Single LLM, RAG, and Multi-Agent RAG.

        Supports evaluating up to all 20 test cases.
        Distinguishes provisional developer targets from verified human ground truth.
        Mode A retrieval metrics report None (N/A) rather than 0.0.
        Human novelty and gap agreements report None (N/A) if independent human labels are absent.
        """
        test_cases = self.load_test_cases(test_cases_path)
        if not test_cases:
            return {"error": "No test cases available"}

        if isinstance(max_cases, str) and max_cases.lower() in ["all", "max"]:
            num_cases = len(test_cases)
        else:
            try:
                num_cases = min(int(max_cases), len(test_cases))
            except (ValueError, TypeError):
                num_cases = len(test_cases)

        selected_cases = test_cases[:num_cases]
        logger.info(f"Running benchmark on {len(selected_cases)}/{len(test_cases)} test cases...")

        has_verified_annotations = any(tc.get("human_verified", False) for tc in selected_cases)

        execution_metadata = {
            "benchmark_type": "software_integration",
            "dataset_annotation_status": "provisional_developer_curated",
            "human_validated": False,
            "diagnostic_label": "Provisional retrieval diagnostics against developer-curated targets",
            "llm_provider": settings.llm_provider,
            "model_name": settings.llm_model,
            "temperature": settings.temperature,
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_cases_file": str((test_cases_path or (EVALUATION_DIR / "test_cases.json")).name),
            "ground_truth_qualification": (
                "Retrieval metrics are provisional retrieval diagnostics evaluated against developer-curated targets (N=20). "
                "They represent software integration diagnostics, NOT validated performance or proof of model superiority. "
                "Agreement against human labels reports N/A until independent human annotation is completed."
            )
        }

        results_by_mode = {
            "single_llm": {
                "latencies": [],
                "novelty_agreements_human": [],
                "provisional_agreements": []
            },
            "rag": {
                "latencies": [],
                "precision_3": [],
                "recall_3": [],
                "f1_3": [],
                "mrr": [],
                "novelty_agreements_human": [],
                "provisional_agreements": [],
                "grounding_ratios": []
            },
            "multi_agent_rag": {
                "latencies": [],
                "precision_3": [],
                "recall_3": [],
                "f1_3": [],
                "mrr": [],
                "novelty_agreements_human": [],
                "provisional_agreements": [],
                "grounding_ratios": [],
                "gaps_found": []
            }
        }

        detailed_cases = []

        for tc in selected_cases:
            case_id = tc["case_id"]
            idea_text = tc["idea_description"]
            idea_input = UserIdeaInput(idea=idea_text, title=tc.get("idea_title", ""), domain=tc.get("domain", ""))
            
            # Developer-curated provisional targets
            ground_truth_docs = set(tc.get("relevant_patents", []) + tc.get("relevant_papers", []))
            provisional_novelty = tc.get("expected_novelty")
            human_novelty = tc.get("human_novelty_score") if tc.get("human_verified", False) else None

            logger.info(f"Evaluating {case_id}: '{tc.get('idea_title', '')}'")
            comp_res = self.orchestrator.compare(idea_input)

            # Evaluate Mode A: Single LLM (Parametric baseline - no retrieval)
            rep_a = comp_res.single_llm
            results_by_mode["single_llm"]["latencies"].append(rep_a.processing_time_seconds)
            if human_novelty is not None:
                nov_a_human = evaluate_novelty_agreement(rep_a.novelty.novelty_level if rep_a.novelty else "Unassessed", human_novelty)
                results_by_mode["single_llm"]["novelty_agreements_human"].append(nov_a_human["is_match"])
            if provisional_novelty:
                nov_a_prov = evaluate_novelty_agreement(rep_a.novelty.novelty_level if rep_a.novelty else "Unassessed", provisional_novelty)
                results_by_mode["single_llm"]["provisional_agreements"].append(nov_a_prov["is_match"])

            # Evaluate Mode B: RAG
            rep_b = comp_res.rag
            results_by_mode["rag"]["latencies"].append(rep_b.processing_time_seconds)
            ret_b_ids = [c.document_id for c in rep_b.retrieval.all_results] if rep_b.retrieval else []
            p3_b = precision_at_k(ret_b_ids, ground_truth_docs, k=3)
            r3_b = recall_at_k(ret_b_ids, ground_truth_docs, k=3)
            results_by_mode["rag"]["precision_3"].append(p3_b)
            results_by_mode["rag"]["recall_3"].append(r3_b)
            results_by_mode["rag"]["f1_3"].append(f1_at_k(p3_b, r3_b))
            results_by_mode["rag"]["mrr"].append(mean_reciprocal_rank(ret_b_ids, ground_truth_docs))
            if human_novelty is not None:
                nov_b_human = evaluate_novelty_agreement(rep_b.novelty.novelty_level if rep_b.novelty else "Unassessed", human_novelty)
                results_by_mode["rag"]["novelty_agreements_human"].append(nov_b_human["is_match"])
            if provisional_novelty:
                nov_b_prov = evaluate_novelty_agreement(rep_b.novelty.novelty_level if rep_b.novelty else "Unassessed", provisional_novelty)
                results_by_mode["rag"]["provisional_agreements"].append(nov_b_prov["is_match"])
            exp_b = evaluate_explainability(rep_b.novelty.supporting_evidence if rep_b.novelty else [], set(ret_b_ids))
            results_by_mode["rag"]["grounding_ratios"].append(exp_b["grounding_ratio"])

            # Evaluate Mode C: Multi-Agent RAG
            rep_c = comp_res.multi_agent_rag
            results_by_mode["multi_agent_rag"]["latencies"].append(rep_c.processing_time_seconds)
            ret_c_ids = [c.document_id for c in rep_c.retrieval.all_results] if rep_c.retrieval else []
            p3_c = precision_at_k(ret_c_ids, ground_truth_docs, k=3)
            r3_c = recall_at_k(ret_c_ids, ground_truth_docs, k=3)
            results_by_mode["multi_agent_rag"]["precision_3"].append(p3_c)
            results_by_mode["multi_agent_rag"]["recall_3"].append(r3_c)
            results_by_mode["multi_agent_rag"]["f1_3"].append(f1_at_k(p3_c, r3_c))
            results_by_mode["multi_agent_rag"]["mrr"].append(mean_reciprocal_rank(ret_c_ids, ground_truth_docs))
            if human_novelty is not None:
                nov_c_human = evaluate_novelty_agreement(rep_c.novelty.novelty_level if rep_c.novelty else "Unassessed", human_novelty)
                results_by_mode["multi_agent_rag"]["novelty_agreements_human"].append(nov_c_human["is_match"])
            if provisional_novelty:
                nov_c_prov = evaluate_novelty_agreement(rep_c.novelty.novelty_level if rep_c.novelty else "Unassessed", provisional_novelty)
                results_by_mode["multi_agent_rag"]["provisional_agreements"].append(nov_c_prov["is_match"])
            exp_c = evaluate_explainability(rep_c.novelty.supporting_evidence if rep_c.novelty else [], set(ret_c_ids))
            results_by_mode["multi_agent_rag"]["grounding_ratios"].append(exp_c["grounding_ratio"])
            results_by_mode["multi_agent_rag"]["gaps_found"].append(len(rep_c.gaps.unexplored_combinations) if rep_c.gaps else 0)

            detailed_cases.append({
                "case_id": case_id,
                "title": tc.get("idea_title", ""),
                "domain": tc.get("domain", ""),
                "annotation_status": tc.get("annotation_status", "provisional_developer_curated"),
                "human_verified": tc.get("human_verified", False),
                "single_llm": {
                    "latency_seconds": rep_a.processing_time_seconds,
                    "novelty_score": rep_a.novelty.novelty_score if rep_a.novelty else None,
                    "novelty_level": rep_a.novelty.novelty_level if rep_a.novelty else "Unassessed",
                    "precision@3": None,
                    "recall@3": None,
                    "citation_grounding": 0.0,
                    "citation_note": "Parametric generation — no retrieved citations"
                },
                "rag": {
                    "latency_seconds": rep_b.processing_time_seconds,
                    "novelty_score": rep_b.novelty.novelty_score if rep_b.novelty else None,
                    "novelty_level": rep_b.novelty.novelty_level if rep_b.novelty else "Unassessed",
                    "precision@3_provisional": p3_b,
                    "recall@3_provisional": r3_b,
                    "f1@3_provisional": f1_at_k(p3_b, r3_b),
                    "mrr_provisional": mean_reciprocal_rank(ret_b_ids, ground_truth_docs),
                    "citation_grounding": exp_b["grounding_ratio"]
                },
                "multi_agent_rag": {
                    "latency_seconds": rep_c.processing_time_seconds,
                    "novelty_score": rep_c.novelty.novelty_score if rep_c.novelty else None,
                    "novelty_level": rep_c.novelty.novelty_level if rep_c.novelty else "Unassessed",
                    "precision@3_provisional": p3_c,
                    "recall@3_provisional": r3_c,
                    "f1@3_provisional": f1_at_k(p3_c, r3_c),
                    "mrr_provisional": mean_reciprocal_rank(ret_c_ids, ground_truth_docs),
                    "citation_grounding": exp_c["grounding_ratio"],
                    "gaps_count": len(rep_c.gaps.unexplored_combinations) if rep_c.gaps else 0
                }
            })

        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
        def agreement_ratio(lst):
            valid = [x for x in lst if x is not None]
            return round(sum(1 for x in valid if x) / len(valid), 4) if valid else None

        summary = {
            "execution_metadata": execution_metadata,
            "num_cases_evaluated": len(selected_cases),
            "single_llm": {
                "avg_latency_seconds": avg(results_by_mode["single_llm"]["latencies"]),
                "avg_precision@3": None,
                "avg_recall@3": None,
                "avg_f1@3": None,
                "mean_reciprocal_rank": None,
                "avg_grounding_ratio": 0.0,
                "grounding_note": "Parametric generation (zero retrieved citations)",
                "novelty_agreement_human": agreement_ratio(results_by_mode["single_llm"]["novelty_agreements_human"]),
                "provisional_agreement_developer": agreement_ratio(results_by_mode["single_llm"]["provisional_agreements"])
            },
            "rag": {
                "avg_latency_seconds": avg(results_by_mode["rag"]["latencies"]),
                "avg_precision@3_provisional": avg(results_by_mode["rag"]["precision_3"]),
                "avg_recall@3_provisional": avg(results_by_mode["rag"]["recall_3"]),
                "avg_f1@3_provisional": avg(results_by_mode["rag"]["f1_3"]),
                "mean_reciprocal_rank_provisional": avg(results_by_mode["rag"]["mrr"]),
                "avg_grounding_ratio": avg(results_by_mode["rag"]["grounding_ratios"]),
                "novelty_agreement_human": agreement_ratio(results_by_mode["rag"]["novelty_agreements_human"]),
                "provisional_agreement_developer": agreement_ratio(results_by_mode["rag"]["provisional_agreements"]),
                "avg_gaps_discovered": 0.0
            },
            "multi_agent_rag": {
                "avg_latency_seconds": avg(results_by_mode["multi_agent_rag"]["latencies"]),
                "avg_precision@3_provisional": avg(results_by_mode["multi_agent_rag"]["precision_3"]),
                "avg_recall@3_provisional": avg(results_by_mode["multi_agent_rag"]["recall_3"]),
                "avg_f1@3_provisional": avg(results_by_mode["multi_agent_rag"]["f1_3"]),
                "mean_reciprocal_rank_provisional": avg(results_by_mode["multi_agent_rag"]["mrr"]),
                "avg_grounding_ratio": avg(results_by_mode["multi_agent_rag"]["grounding_ratios"]),
                "novelty_agreement_human": agreement_ratio(results_by_mode["multi_agent_rag"]["novelty_agreements_human"]),
                "provisional_agreement_developer": agreement_ratio(results_by_mode["multi_agent_rag"]["provisional_agreements"]),
                "avg_gaps_discovered": avg(results_by_mode["multi_agent_rag"]["gaps_found"])
            },
            "provisional_targets_disclaimer": (
                "NOTE: Retrieval metrics are provisional retrieval diagnostics evaluated against developer-curated targets (N=20), "
                "which serve as a software integration diagnostic rather than an independently verified gold standard. "
                "These metrics must NOT be interpreted as validated accuracy, research performance, or proof of model superiority. "
                "Agreement against human novelty annotations is marked N/A until independent double-blind annotations are collected."
            ),
            "detailed_cases": detailed_cases
        }

        # Save results to disk
        out_path = EVALUATION_DIR / "benchmark_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved benchmark results to {out_path}")

        return summary
