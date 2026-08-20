import os
import time
import numpy as np
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

def main():
    print("\n" + "=" * 80)
    print("LEVEL 2 EXPERIMENT 1: STANDALONE T5 EVALUATOR CLASSIFICATION (100 PAIRS)")
    print("=" * 80)

    dataset_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\train_popqa.txt"
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        return

    questions = []
    passages = []
    true_labels = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 100:
                break
            if not line.strip():
                continue
            content, label_str = line.strip().split("\t")
            q, p = content.split(" [SEP] ")
            questions.append(q.strip())
            passages.append(p.strip())
            true_labels.append(int(label_str.strip()))

    evaluator = T5RetrievalEvaluator()

    pred_labels = []
    pred_probs = []
    crag_scores = []
    latencies = []

    t_start = time.time()
    for q, p in zip(questions, passages):
        t0 = time.time()
        res = evaluator.evaluate(q, p)
        t1 = time.time()
        
        pred_labels.append(1 if res.is_relevant else 0)
        pred_probs.append(res.relevance_probability)
        crag_scores.append(res.crag_score)
        latencies.append(t1 - t0)

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

    mean_lat = np.mean(latencies) * 1000.0

    print("\n[STANDALONE EVALUATOR CLASSIFICATION RESULTS]")
    print(f"Total Evaluated Pairs: {len(true_labels)}")
    print(f"Accuracy:              {acc * 100.0:.2f}%")
    print(f"Precision:             {prec * 100.0:.2f}%")
    print(f"Recall:                {rec * 100.0:.2f}%")
    print(f"F1-Score:              {f1 * 100.0:.2f}%")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"PR-AUC:                {pr_auc:.4f}")
    print(f"Mean Latency / Pair:   {mean_lat:.2f} ms")
    print("\nConfusion Matrix:")
    print("                Predicted (0)  Predicted (1)")
    print(f"  Actual (0):      {cm[0][0]:<12} {cm[0][1]:<12}")
    print(f"  Actual (1):      {cm[1][0]:<12} {cm[1][1]:<12}")

    print("\n" + "=" * 80)
    print("LEVEL 2 EXPERIMENT 2: CRAG RESEARCH BENCHMARK DISTRIBUTION (20 QUERIES)")
    print("=" * 80)

    # Collect 20 unique queries
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
            if len(query_map) >= 20:
                break

    correct_cnt = 0
    ambiguous_cnt = 0
    incorrect_cnt = 0
    crag_latencies = []

    for q, psgs in query_map.items():
        docs = [RetrievedDocument(doc_id=str(i), text=p) for i, p in enumerate(psgs[:5])]
        retriever = LocalRetriever(corpus=docs)
        pipeline = CRAGPipeline(retriever=retriever, evaluator=evaluator)

        c_res = pipeline.run(query=q, top_k=5, verbose=False)
        crag_latencies.append(c_res.latency_seconds)

        if c_res.decision.value == "CORRECT":
            correct_cnt += 1
        elif c_res.decision.value == "AMBIGUOUS":
            ambiguous_cnt += 1
        else:
            incorrect_cnt += 1

    total_q = len(query_map)
    ext_trigger_rate = ((ambiguous_cnt + incorrect_cnt) / total_q) * 100.0

    print("\n[CRAG RESEARCH BENCHMARK DISTRIBUTION TABLE]")
    print(f"{'System Configuration':<28} | {'CORRECT %':<10} | {'AMBIGUOUS %':<12} | {'INCORRECT %':<12} | {'Ext Search %':<12}")
    print("-" * 80)
    print(f"{'Vanilla RAG (Baseline)':<28} | {'N/A':<10} | {'N/A':<12} | {'N/A':<12} | {'0.0%':<12}")
    print(f"{'CRAG + T5-Small + Groq':<28} | {correct_cnt/total_q*100:<9.1f}% | {ambiguous_cnt/total_q*100:<11.1f}% | {incorrect_cnt/total_q*100:<11.1f}% | {ext_trigger_rate:<11.1f}%")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
