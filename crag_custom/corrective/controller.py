from typing import List
from crag_custom.evaluator.schemas import EvaluationResult
from crag_custom.corrective.decision import CRAGDecision, DecisionResult
from crag_custom.config.thresholds import thresholds

class CRAGDecisionController:
    """
    Faithful CRAG 3-Way Decision Controller.
    Reproduces the exact reference implementation logic from CRAG_Inference.py (process_flag).

    Reference Logic:
      - CORRECT:   if max(scores) >= upper_threshold (e.g. >= 0.592)
      - INCORRECT: if max(scores) < -lower_threshold (e.g. < -0.995)
      - AMBIGUOUS: otherwise (lower_cutoff <= max(scores) < upper_threshold)
    """
    def __init__(
        self,
        upper_threshold: float = thresholds.DEFAULT_UPPER,
        lower_threshold: float = thresholds.DEFAULT_LOWER
    ):
        self.upper_threshold = upper_threshold
        # Reference script run_crag_inference.sh passes --lower_threshold 0.995 and performs:
        # args.lower_threshold = -args.lower_threshold => -0.995
        if lower_threshold > 0:
            self.lower_cutoff = -lower_threshold
            self.lower_threshold_param = lower_threshold
        else:
            self.lower_cutoff = lower_threshold
            self.lower_threshold_param = abs(lower_threshold)

    def decide(self, eval_results: List[EvaluationResult]) -> DecisionResult:
        if not eval_results:
            return DecisionResult(
                decision=CRAGDecision.INCORRECT,
                confidence_score=-1.0,
                individual_results=[],
                max_score=-1.0,
                min_score=-1.0,
                upper_threshold=self.upper_threshold,
                lower_threshold=self.lower_cutoff
            )
            
        scores = [r.crag_score for r in eval_results]
        max_score = max(scores)
        min_score = min(scores)
        
        # Exact reference CRAG process_flag decision logic
        if max_score >= self.upper_threshold:
            decision = CRAGDecision.CORRECT
        elif max_score < self.lower_cutoff:
            decision = CRAGDecision.INCORRECT
        else:
            decision = CRAGDecision.AMBIGUOUS
            
        return DecisionResult(
            decision=decision,
            confidence_score=max_score,
            individual_results=eval_results,
            max_score=max_score,
            min_score=min_score,
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_cutoff
        )
