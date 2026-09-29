from typing import List, Set


def precision_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Calculates Precision@K = (retrieved & ground_truth)[:k] / k."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return round(hits / len(top_k), 4)


def recall_at_k(retrieved: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Calculates Recall@K = (retrieved & ground_truth)[:k] / len(ground_truth)."""
    if not ground_truth:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return round(hits / len(ground_truth), 4)


def f1_at_k(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)


def mean_reciprocal_rank(retrieved: List[str], ground_truth: Set[str]) -> float:
    """Calculates Reciprocal Rank of the first relevant retrieved document."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in ground_truth:
            return round(1.0 / rank, 4)
    return 0.0
