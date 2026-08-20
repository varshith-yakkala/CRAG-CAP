from typing import List
from crag_custom.evaluator.base import BaseRetrievalEvaluator
from crag_custom.knowledge.filtering import filter_relevant_strips
from crag_custom.knowledge.recomposition import recompose_knowledge

def process_external_knowledge(
    question: str,
    web_snippets: List[str],
    evaluator: BaseRetrievalEvaluator,
    top_n: int = 5
) -> str:
    """
    External Knowledge Processing Pipeline:
    1. Filter web search snippets using Our T5 Evaluator.
    2. Recompose into external knowledge context.
    """
    if not web_snippets:
        return ""
    relevant_strips = filter_relevant_strips(question, web_snippets, evaluator, top_n=top_n)
    return recompose_knowledge(relevant_strips)
