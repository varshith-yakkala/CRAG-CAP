import pytest
import os
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.evaluator.schemas import EvaluationResult
from crag_custom.config.settings import settings

def test_t5_evaluator_loading():
    if not os.path.exists(settings.T5_MODEL_PATH):
        pytest.skip(f"T5 model checkpoint not found at {settings.T5_MODEL_PATH}")
        
    evaluator = T5RetrievalEvaluator()
    assert evaluator.model is not None
    assert evaluator.id_1 is not None
    assert evaluator.id_0 is not None

def test_t5_evaluator_inference():
    if not os.path.exists(settings.T5_MODEL_PATH):
        pytest.skip(f"T5 model checkpoint not found at {settings.T5_MODEL_PATH}")
        
    evaluator = T5RetrievalEvaluator()
    question = "What is George Rankin's occupation?"
    passage_rel = "George Rankin was an Australian soldier and politician. He attended the local state school and became a farmer."
    passage_irrel = "Bangai-O Spirits is an action game for the Nintendo DS."

    res_rel = evaluator.evaluate(question, passage_rel)
    assert isinstance(res_rel, EvaluationResult)
    assert res_rel.relevance_probability > 0.5
    assert res_rel.crag_score > 0.0
    assert res_rel.label == "1"

    res_irrel = evaluator.evaluate(question, passage_irrel)
    assert res_irrel.relevance_probability < 0.5
    assert res_irrel.crag_score < 0.0
    assert res_irrel.label == "0"
