import re
from typing import List, Union

def exact_match_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    pred_clean = prediction.strip().lower()
    for gt in ground_truths:
        gt_clean = gt.strip().lower()
        if gt_clean in pred_clean or pred_clean in gt_clean:
            return 1.0
    return 0.0

def accuracy_score_metric(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
