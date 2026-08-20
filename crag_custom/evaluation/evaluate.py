import time
from typing import List, Dict, Any
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.base import BaseRetriever
from crag_custom.evaluation.metrics import exact_match_score
from crag_custom.llm.generator import GroqGenerator

class ExperimentRunner:
    """
    Experimental Comparison Runner:
    Baseline: Vanilla RAG
    vs.
    System: CRAG + Our T5-Small Evaluator + Groq
    """
    def __init__(self, pipeline: CRAGPipeline, retriever: BaseRetriever):
        self.pipeline = pipeline
        self.retriever = retriever
        self.generator = GroqGenerator()

    def run_vanilla_rag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        start = time.time()
        docs = self.retriever.retrieve(query, top_k=top_k)
        context = " ".join([d.text for d in docs])
        answer = self.generator.generate(query, context=context)
        latency = time.time() - start
        return {
            "query": query,
            "answer": answer,
            "latency": latency,
            "context": context
        }

    def run_comparison(self, test_dataset: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        results_vanilla = []
        results_crag = []

        correct_count = 0
        ambiguous_count = 0
        incorrect_count = 0

        for item in test_dataset:
            q = item["question"]
            gt = item.get("answers", item.get("answer", []))

            # 1. Run Vanilla RAG
            v_res = self.run_vanilla_rag(q, top_k=top_k)
            v_match = exact_match_score(v_res["answer"], gt)
            v_res["match"] = v_match
            results_vanilla.append(v_res)

            # 2. Run CRAG (Our T5 Evaluator + Groq)
            c_res = self.pipeline.run(q, top_k=top_k, verbose=False)
            c_match = exact_match_score(c_res.final_answer, gt)
            c_res_dict = {
                "query": q,
                "answer": c_res.final_answer,
                "decision": c_res.decision.value,
                "pathway": c_res.selected_pathway,
                "match": c_match,
                "latency": c_res.latency_seconds
            }
            results_crag.append(c_res_dict)

            if c_res.decision.value == "CORRECT":
                correct_count += 1
            elif c_res.decision.value == "AMBIGUOUS":
                ambiguous_count += 1
            else:
                incorrect_count += 1

        total = len(test_dataset) if test_dataset else 1
        summary = {
            "total_examples": len(test_dataset),
            "vanilla_rag": {
                "mean_match": sum(r["match"] for r in results_vanilla) / total,
                "mean_latency": sum(r["latency"] for r in results_vanilla) / total,
            },
            "crag_our_t5": {
                "mean_match": sum(r["match"] for r in results_crag) / total,
                "mean_latency": sum(r["latency"] for r in results_crag) / total,
                "correct_rate": (correct_count / total) * 100,
                "ambiguous_rate": (ambiguous_count / total) * 100,
                "incorrect_rate": (incorrect_count / total) * 100,
                "external_search_trigger_rate": ((ambiguous_count + incorrect_count) / total) * 100
            }
        }
        return summary
