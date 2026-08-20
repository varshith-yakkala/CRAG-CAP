import sys
import argparse
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument

def main():
    parser = argparse.ArgumentParser(description="CLI Runner for Custom CRAG Pipeline")
    parser.add_argument("--query", type=str, default="What is George Rankin's occupation?", help="User question")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top documents to retrieve")
    args = parser.parse_args()

    # Sample corpus
    sample_corpus = [
        RetrievedDocument(
            doc_id="1",
            text="George Rankin was an Australian soldier and politician. He attended the local state school and became a farmer."
        ),
        RetrievedDocument(
            doc_id="2",
            text="Bangai-O Spirits is an action game for the Nintendo DS. The game features a level editor."
        ),
        RetrievedDocument(
            doc_id="3",
            text="George Claus Rankin was a British judge in India."
        )
    ]

    retriever = LocalRetriever(corpus=sample_corpus)
    pipeline = CRAGPipeline(retriever=retriever)
    
    result = pipeline.run(query=args.query, top_k=args.top_k, verbose=True)
    print("\n[SUMMARY]")
    print(f"Decision: {result.decision.value}")
    print(f"Pathway:  {result.selected_pathway}")
    print(f"Answer:   {result.final_answer}")

if __name__ == "__main__":
    main()
