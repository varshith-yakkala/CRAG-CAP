import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.pipeline.crag_pipeline import CRAGPipeline
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.evaluation.evaluate import ExperimentRunner
from crag_custom.evaluation.metrics import exact_match_score
from crag_custom.config.settings import settings

def run_experiment_1_standalone_evaluator(sample_size: int = 500):
    print("\n" + "=" * 80)
    print(f"LEVEL 2 - EXPERIMENT 1: STANDALONE T5 EVALUATOR CLASSIFICATION PERFORMANCE")
    print(f"Evaluating Our T5-Small Evaluator on {sample_size} labelled (Question, Passage) pairs")
    print("=" * 80)

    dataset_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\train_popqa.txt"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        return

    # Load lines
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()][:sample_size]

    questions = []
    passages = []
    true_labels = []

    for line in lines:
        try:
            content, label_str = line.split("\t")
            q, p = content.split(" [SEP] ")
            questions.append(q.strip())
            passages.append(p.strip())
            true_labels.append(int(label_str.strip()))
        except Exception:
            continue

    print(f"Loaded {len(true_labels)} valid test pairs (Positive: {sum(true_labels)}, Negative: {len(true_labels)-sum(true_labels)}).")

    evaluator = T5RetrievalEvaluator()

    pred_labels = []
    pred_probs = []
    crag_scores = []
    latencies = []

    start_total = time.time()
    for q, p in zip(questions, passages):
        t0 = time.time()
        res = evaluator.evaluate(q, p)
        t1 = time.time()
        
        pred_labels.append(1 if res.is_relevant else 0)
        pred_probs.append(res.relevance_probability)
        crag_scores.append(res.crag_score)
        latencies.append(t1 - t0)

    total_time = time.time() - start_total

    # Metrics computation
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, zero_division=0)
    rec = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels)
    
    try:
        roc_auc = roc_auc_score(true_labels, pred_probs)
    except Exception:
        roc_auc = 0.0

    try:
        precision_arr, recall_arr, _ = precision_recall_curve(true_labels, pred_probs)
        pr_auc = auc(recall_arr, precision_arr)
    except Exception:
        pr_auc = 0.0

    mean_latency = np.mean(latencies) * 1000.0 # ms

    print("\n[STANDALONE EVALUATOR CLASSIFICATION RESULTS]")
    print(f"Total Evaluated Pairs: {len(true_labels)}")
    print(f"Accuracy:              {acc * 100.0:.2f}%")
    print(f"Precision:             {prec * 100.0:.2f}%")
    print(f"Recall:                {rec * 100.0:.2f}%")
    print(f"F1-Score:              {f1 * 100.0:.2f}%")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"PR-AUC:                {pr_auc:.4f}")
    print(f"Mean Latency / Pair:   {mean_latency:.2f} ms")
    print("\nConfusion Matrix:")
    print("                Predicted (0)  Predicted (1)")
    print(f"  Actual (0):      {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"  Actual (1):      {cm[1][0]:<12} {cm[1][1]:<12}")

    return {
        "sample_size": len(true_labels),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mean_latency_ms": mean_latency,
        "confusion_matrix": cm.tolist()
    }


def run_experiment_2_crag_vs_vanilla(num_queries: int = 50):
    print("\n" + "=" * 80)
    print(f"LEVEL 2 - EXPERIMENT 2: RESEARCH BENCHMARK — VANILLA RAG vs. CRAG (OUR T5)")
    print(f"Evaluating {num_queries} PopQA queries to measure corrective effectiveness")
    print("=" * 80)

    dataset_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\train_popqa.txt"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        return

    # Group passages by question
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
                query_map[q] = {"question": q, "passages": [], "has_relevant": False}
            query_map[q]["passages"].append((p, label))
            if label == 1:
                query_map[q]["has_relevant"] = True
            if len(query_map) >= num_queries:
                break

    benchmark_queries = list(query_map.values())[:num_queries]
    print(f"Collected {len(benchmark_queries)} benchmark queries for evaluation.")

    evaluator = T5RetrievalEvaluator()

    results_vanilla = []
    results_crag = []

    correct_count = 0
    ambiguous_count = 0
    incorrect_count = 0

    for idx, q_item in enumerate(benchmark_queries, start=1):
        q = q_item["question"]
        passages = [p for p, _ in q_item["passages"][:5]]
        
        # Build local retriever for this query
        docs = [RetrievedDocument(doc_id=str(i), text=p) for i, p in enumerate(passages)]
        retriever = LocalRetriever(corpus=docs)
        pipeline = CRAGPipeline(retriever=retriever, evaluator=evaluator)

        # Ground truth answer keywords extracted from question topic
        # e.g. "What is George Rankin's occupation?" -> ["soldier", "politician", "farmer"]
        # We perform end-to-end pipeline run
        c_res = pipeline.run(query=q, top_k=5, verbose=False)
        
        c_res_dict = {
            "query": q,
            "decision": c_res.decision.value,
            "pathway": c_res.selected_pathway,
            "confidence_score": c_res.confidence_score,
            "latency": c_res.latency_seconds
        }
        results_crag.append(c_res_dict)

        if c_res.decision.value == "CORRECT":
            correct_count += 1
        elif c_res.decision.value == "AMBIGUOUS":
            ambiguous_count += 1
        else:
            incorrect_count += 1

    total = len(benchmark_queries)
    ext_trigger_count = ambiguous_count + incorrect_count
    ext_trigger_rate = (ext_trigger_count / total) * 100.0

    print("\n[RESEARCH BENCHMARK DISTRIBUTION & SUMMARY]")
    print(f"Total Queries Evaluated:    {total}")
    print(f"  CORRECT Decisions:        {correct_count} ({correct_count/total*100:.1f}%)")
    print(f"  AMBIGUOUS Decisions:      {ambiguous_count} ({ambiguous_count/total*100:.1f}%)")
    print(f"  INCORRECT Decisions:      {incorrect_count} ({incorrect_count/total*100:.1f}%)")
    print(f"External Search Trigger Rate: {ext_trigger_rate:.1f}%")
    print(f"Mean Pipeline Latency:      {np.mean([r['latency'] for r in results_crag]):.3f} sec")

    # Table format
    print("\n" + "=" * 80)
    print(f"{'System Configuration':<30} | {'CORRECT %':<10} | {'AMBIGUOUS %':<12} | {'INCORRECT %':<12} | {'Ext Search %':<12}")
    print("-" * 80)
    print(f"{'Vanilla RAG (Baseline)':<30} | {'N/A':<10} | {'N/A':<12} | {'N/A':<12} | {'0.0%':<12}")
    print(f"{'CRAG + T5-Small + Groq':<30} | {correct_count/total*100:<9.1f}% | {ambiguous_count/total*100:<11.1f}% | {incorrect_count/total*100:<11.1f}% | {ext_trigger_rate:<11.1f}%")
    print("=" * 80 + "\n")

    return {
        "total_queries": total,
        "correct_rate": (correct_count / total) * 100.0,
        "ambiguous_rate": (ambiguous_count / total) * 100.0,
        "incorrect_rate": (incorrect_count / total) * 100.0,
        "ext_search_trigger_rate": ext_trigger_rate,
        "mean_latency_sec": float(np.mean([r['latency'] for r in results_crag]))
    }

if __name__ == "__main__":
    run_experiment_1_standalone_evaluator(sample_size=500)
    run_experiment_2_crag_vs_vanilla(num_queries=50)
