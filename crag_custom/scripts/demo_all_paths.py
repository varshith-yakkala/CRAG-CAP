import os
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.corrective.controller import CRAGDecisionController

def main():
    print("==================================================================")
    print("DEMONSTRATING ALL 3 CRAG PATHWAYS UNDER ONE FIXED CONFIGURATION")
    print("Fixed Threshold Configuration: upper_threshold=+0.592, lower_cutoff=-0.995")
    print("==================================================================")

    evaluator = T5RetrievalEvaluator()
    # Single fixed PopQA threshold configuration for ALL 3 demonstrations
    controller = CRAGDecisionController(upper_threshold=0.592, lower_threshold=0.995)

    # 1. Test CORRECT Path (Natural High Relevance Score >= +0.592)
    print("\n\n" + "#" * 70)
    print("DEMO 1: NATURAL CORRECT PATHWAY (max_score >= +0.592)")
    print("#" * 70)
    q1 = "What is George Rankin's occupation?"
    corpus_correct = [
        RetrievedDocument(doc_id="1", text="George Rankin was an Australian soldier and politician. He attended local school and became a farmer.")
    ]
    retriever_c = LocalRetriever(corpus=corpus_correct)
    pipeline_c = CRAGPipeline(retriever=retriever_c, evaluator=evaluator, controller=controller)
    pipeline_c.run(query=q1, top_k=1, verbose=True)

    # 2. Test AMBIGUOUS Path (Natural Medium Relevance Score: -0.995 <= max_score < +0.592)
    print("\n\n" + "#" * 70)
    print("DEMO 2: NATURAL AMBIGUOUS PATHWAY (-0.995 <= max_score < +0.592)")
    print("#" * 70)
    q2 = "In what city was Billy Carlson born?"
    corpus_ambiguous = [
        RetrievedDocument(doc_id="1", text="The Golden Gate Bridge is a suspension bridge spanning the Golden Gate in San Francisco.")
    ]
    retriever_a = LocalRetriever(corpus=corpus_ambiguous)
    pipeline_a = CRAGPipeline(retriever=retriever_a, evaluator=evaluator, controller=controller)
    pipeline_a.run(query=q2, top_k=1, verbose=True)

    # 3. Test INCORRECT Path (Natural Low Relevance Score < -0.995)
    print("\n\n" + "#" * 70)
    print("DEMO 3: NATURAL INCORRECT PATHWAY (max_score < -0.995)")
    print("#" * 70)
    q3 = "What is George Rankin's occupation?"
    corpus_incorrect = [
        RetrievedDocument(doc_id="1", text="Bangai-O Spirits is an action game for the Nintendo DS with 160 levels.")
    ]
    retriever_i = LocalRetriever(corpus=corpus_incorrect)
    pipeline_i = CRAGPipeline(retriever=retriever_i, evaluator=evaluator, controller=controller)
    pipeline_i.run(query=q3, top_k=1, verbose=True)

if __name__ == "__main__":
    main()
