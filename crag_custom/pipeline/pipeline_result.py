from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.evaluator.schemas import EvaluationResult
from crag_custom.corrective.decision import CRAGDecision

@dataclass
class PipelineResult:
    query: str
    retrieved_documents: List[RetrievedDocument]
    eval_results: List[EvaluationResult]
    decision: CRAGDecision
    confidence_score: float
    selected_pathway: str          # "INTERNAL_KNOWLEDGE", "EXTERNAL_SEARCH", "COMBINED_KNOWLEDGE"
    processed_knowledge: str
    rewritten_query: Optional[str] = None
    web_snippets: List[str] = field(default_factory=list)
    final_answer: str = ""
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
