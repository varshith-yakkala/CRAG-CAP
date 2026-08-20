from abc import ABC, abstractmethod
from typing import List
from crag_custom.evaluator.schemas import EvaluationResult

class BaseRetrievalEvaluator(ABC):
    """
    Abstract Base Class for Retrieval Quality Evaluators.
    """
    
    @abstractmethod
    def evaluate(self, question: str, passage: str) -> EvaluationResult:
        """
        Evaluate a single (question, passage) pair.
        """
        pass
    
    @abstractmethod
    def evaluate_batch(self, question: str, passages: List[str]) -> List[EvaluationResult]:
        """
        Evaluate multiple retrieved passages for a single question individually.
        """
        pass
