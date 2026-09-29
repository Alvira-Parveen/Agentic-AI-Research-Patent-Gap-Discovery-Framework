from typing import Optional, Dict, Any


def evaluate_novelty_agreement(predicted_level: str, ground_truth_level: Optional[str]) -> Dict[str, Any]:
    """
    Evaluates agreement between model predicted novelty level and ground truth.
    If ground truth is not provided or marked as TODO, explicitly returns unavailable status.
    """
    if not ground_truth_level or ground_truth_level.upper() in ["TODO", "NONE", "UNAVAILABLE"]:
        return {
            "status": "Evaluation unavailable — ground truth not yet provided.",
            "is_match": None,
            "predicted": predicted_level,
            "ground_truth": None
        }

    p_clean = predicted_level.strip().lower()
    gt_clean = ground_truth_level.strip().lower()
    is_match = (p_clean == gt_clean)

    return {
        "status": "Evaluated",
        "is_match": is_match,
        "predicted": predicted_level,
        "ground_truth": ground_truth_level
    }
