import os
import json
import time
import numpy as np
from typing import List, Dict, Any

from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.corrective.controller import CRAGDecisionController

def load_popqa_test_queries(num_queries: int = 100):
    dataset_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\train_popqa.txt"
    query_map = {}
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            content, label_str = line.strip().split("\t")
            q, p = content.split(" [SEP] ")
            q = q.strip()
            p = p.strip()
            label = int(label_str.strip())
            
            if q not in query_map:
                query_map[q] = []
            query_map[q].append(p)
            if len(query_map) >= num_queries:
                break

    return query_map

def main():
    print("==================================================================")
    print("POPQA END-TO-END CRAG MATCH SCORE & LATENCY EVALUATION (100 QUERIES)")
    print("==================================================================")

    query_map = load_popqa_test_queries(num_queries=100)
    print(f"Loaded {len(query_map)} unique PopQA benchmark queries.")

    evaluator = T5RetrievalEvaluator()
    controller = CRAGDecisionController(upper_threshold=0.592, lower_threshold=0.995)

    latencies = []
    decisions = []
    answers = []

    start_total = time.time()
    for idx, (q, psgs) in enumerate(query_map.items(), start=1):
        docs = [RetrievedDocument(doc_id=str(i), text=p) for i, p in enumerate(psgs[:5])]
        retriever = LocalRetriever(corpus=docs)
        pipeline = CRAGPipeline(retriever=retriever, evaluator=evaluator, controller=controller)

        t0 = time.time()
        res = pipeline.run(query=q, top_k=5, verbose=False)
        t1 = time.time()

        elapsed = t1 - t0
        latencies.append(elapsed)
        decisions.append(res.decision.value)
        answers.append(res.final_answer)

    total_time = time.time() - start_total

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    correct_cnt = decisions.count("CORRECT")
    ambiguous_cnt = decisions.count("AMBIGUOUS")
    incorrect_cnt = decisions.count("INCORRECT")
    ext_cnt = ambiguous_cnt + incorrect_cnt
    total_q = len(decisions)

    print("\n[POPQA END-TO-END CRAG BENCHMARK RESULTS]")
    print(f"Total Queries Evaluated:    {total_q}")
    print(f"Mean Pipeline Latency:      {mean_lat:.3f} sec")
    print(f"Median Pipeline Latency:    {median_lat:.3f} sec")
    print(f"P95 Pipeline Latency:       {p95_lat:.3f} sec")
    print(f"CORRECT Decisions:          {correct_cnt} ({correct_cnt/total_q*100:.1f}%)")
    print(f"AMBIGUOUS Decisions:        {ambiguous_cnt} ({ambiguous_cnt/total_q*100:.1f}%)")
    print(f"INCORRECT Decisions:        {incorrect_cnt} ({incorrect_cnt/total_q*100:.1f}%)")
    print(f"External Search Trigger:    {ext_cnt} ({ext_cnt/total_q*100:.1f}%)")

if __name__ == "__main__":
    main()
