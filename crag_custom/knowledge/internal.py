from typing import List
from crag_custom.evaluator.base import BaseRetrievalEvaluator
from crag_custom.knowledge.decomposition import decompose_passage
from crag_custom.knowledge.filtering import filter_relevant_strips
from crag_custom.knowledge.recomposition import recompose_knowledge

def process_internal_knowledge(
    question: str,
    passages: List[str],
    evaluator: BaseRetrievalEvaluator,
    decompose_mode: str = "selection",
    top_n: int = 5
) -> str:
    """
    Internal Knowledge Processing Pipeline:
    1. Decompose passages into sub-strips.
    2. Filter relevant strips using Our T5 Evaluator.
    3. Recompose into refined internal knowledge string.
    """
    all_strips = []
    for psg in passages:
        all_strips.extend(decompose_passage(psg, mode=decompose_mode))
        
    relevant_strips = filter_relevant_strips(question, all_strips, evaluator, top_n=top_n)
    return recompose_knowledge(relevant_strips)
