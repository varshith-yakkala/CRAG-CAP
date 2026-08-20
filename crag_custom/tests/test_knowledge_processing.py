import pytest
from crag_custom.knowledge.decomposition import decompose_passage
from crag_custom.knowledge.recomposition import recompose_knowledge

def test_passage_decomposition_selection():
    text = "Sentence 1. Sentence 2. Sentence 3."
    strips = decompose_passage(text, mode="selection")
    assert strips == [text]

def test_passage_decomposition_fixed_num():
    words = ["word"] * 120
    text = " ".join(words)
    strips = decompose_passage(text, mode="fixed_num")
    assert len(strips) >= 2

def test_knowledge_recomposition():
    strips = ["Knowledge snippet 1", "Knowledge snippet 2"]
    recomposed = recompose_knowledge(strips)
    assert "Knowledge snippet 1 ; Knowledge snippet 2" == recomposed
