from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
from crag_custom.evaluator.schemas import EvaluationResult

class CRAGDecision(Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"

@dataclass
class DecisionResult:
    decision: CRAGDecision
    confidence_score: float
    individual_results: List[EvaluationResult]
    max_score: float
    min_score: float
    upper_threshold: float
    lower_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
