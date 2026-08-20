from typing import List
from crag_custom.evaluator.base import BaseRetrievalEvaluator

def filter_relevant_strips(
    question: str,
    strips: List[str],
    evaluator: BaseRetrievalEvaluator,
    top_n: int = 5
) -> List[str]:
    """
    Filters and ranks sub-strips using Our T5 Evaluator.
    """
    if not strips:
        return []
    
    scored_strips = []
    for strip in strips:
        eval_res = evaluator.evaluate(question, strip)
        scored_strips.append((eval_res.crag_score, strip))
        
    # Sort by CRAG score descending
    scored_strips.sort(key=lambda x: x[0], reverse=True)
    
    # Return top_n relevant strips
    selected = [strip for score, strip in scored_strips[:top_n] if score > -0.99]
    if not selected and scored_strips:
        selected = [scored_strips[0][1]]
    return selected
