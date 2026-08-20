import json
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.evaluation.evaluate import ExperimentRunner

def main():
    print("==========================================================")
    print("CRAG EXPERIMENTAL COMPARISON: Vanilla RAG vs CRAG (Our T5)")
    print("==========================================================")

    sample_corpus = [
        RetrievedDocument(
            doc_id="1",
            text="George Rankin was an Australian soldier and politician. He attended the local state school and became a farmer."
        ),
        RetrievedDocument(
            doc_id="2",
            text="Measles outbreak kills more than 1,200 in Madagascar."
        ),
        RetrievedDocument(
            doc_id="3",
            text="Ric Flair was declared brain dead on 16 May 2019 is a false rumor."
        )
    ]

    test_dataset = [
        {
            "question": "What is George Rankin's occupation?",
            "answers": ["soldier", "politician", "farmer"]
        },
        {
            "question": "In what city was Billy Carlson born?", # Irrelevant passage trigger
            "answers": ["San Diego"]
        }
    ]

    retriever = LocalRetriever(corpus=sample_corpus)
    pipeline = CRAGPipeline(retriever=retriever)
    runner = ExperimentRunner(pipeline=pipeline, retriever=retriever)

    summary = runner.run_comparison(test_dataset=test_dataset, top_k=2)
    print("\n[EXPERIMENTAL SUMMARY RESULT]")
    print(json.dumps(summary, indent=4))

if __name__ == "__main__":
    main()
