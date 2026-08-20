from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class EvaluationResult:
    """
    Schema for evaluation result of a single (question, passage) pair evaluated by Our T5 Evaluator.
    """
    label: str                     # "1" or "0"
    is_relevant: bool              # True if label == "1"
    relevance_probability: float   # P(relevant) in [0.0, 1.0]
    crag_score: float              # 2 * P(relevant) - 1.0 in [-1.0, +1.0]
    raw_output: str                # Generated raw output text
    logit_1: float = 0.0           # Decoder logit for '1'
    logit_0: float = 0.0           # Decoder logit for '0'
    metadata: Dict[str, Any] = field(default_factory=dict)
