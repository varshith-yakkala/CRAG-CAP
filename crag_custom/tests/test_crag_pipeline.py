import pytest
import os
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.config.settings import settings

def test_pipeline_execution():
    if not os.path.exists(settings.T5_MODEL_PATH):
        pytest.skip(f"T5 model checkpoint not found at {settings.T5_MODEL_PATH}")

    corpus = [
        RetrievedDocument(doc_id="1", text="George Rankin was an Australian soldier and politician. He became a farmer.")
    ]
    retriever = LocalRetriever(corpus=corpus)
    evaluator = T5RetrievalEvaluator()
    pipeline = CRAGPipeline(retriever=retriever, evaluator=evaluator)
    
    result = pipeline.run(query="What is George Rankin's occupation?", top_k=1, verbose=False)
    assert result.query == "What is George Rankin's occupation?"
    assert result.decision is not None
    assert result.final_answer != ""
