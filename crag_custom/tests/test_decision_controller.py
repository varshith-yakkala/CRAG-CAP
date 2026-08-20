import pytest
from crag_custom.corrective.controller import CRAGDecisionController
from crag_custom.corrective.decision import CRAGDecision
from crag_custom.evaluator.schemas import EvaluationResult

def test_decision_controller_correct():
    controller = CRAGDecisionController(upper_threshold=0.5, lower_threshold=-0.9)
    res_list = [
        EvaluationResult(label="1", is_relevant=True, relevance_probability=0.85, crag_score=0.70, raw_output="1"),
        EvaluationResult(label="0", is_relevant=False, relevance_probability=0.05, crag_score=-0.90, raw_output="0")
    ]
    dec_res = controller.decide(res_list)
    assert dec_res.decision == CRAGDecision.CORRECT
    assert dec_res.max_score == 0.70

def test_decision_controller_incorrect():
    controller = CRAGDecisionController(upper_threshold=0.5, lower_threshold=-0.9)
    res_list = [
        EvaluationResult(label="0", is_relevant=False, relevance_probability=0.01, crag_score=-0.98, raw_output="0"),
        EvaluationResult(label="0", is_relevant=False, relevance_probability=0.02, crag_score=-0.96, raw_output="0")
    ]
    dec_res = controller.decide(res_list)
    assert dec_res.decision == CRAGDecision.INCORRECT
    assert dec_res.max_score == -0.96

def test_decision_controller_ambiguous():
    controller = CRAGDecisionController(upper_threshold=0.5, lower_threshold=-0.9)
    res_list = [
        EvaluationResult(label="1", is_relevant=True, relevance_probability=0.60, crag_score=0.20, raw_output="1"),
        EvaluationResult(label="0", is_relevant=False, relevance_probability=0.10, crag_score=-0.80, raw_output="0")
    ]
    dec_res = controller.decide(res_list)
    assert dec_res.decision == CRAGDecision.AMBIGUOUS
    assert dec_res.max_score == 0.20
