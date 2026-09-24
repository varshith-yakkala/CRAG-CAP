import os
import sys
import io
import json
import time
import re
import string
import urllib.request
from collections import Counter
from typing import List, Dict, Any

# Ensure UTF-8 output formatting for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))

from crag_custom.config.settings import settings
from crag_custom.config.thresholds import thresholds
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.corrective.controller import CRAGDecisionController
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.pipeline.crag_pipeline import CRAGPipeline

# Official reference PopQA match and normalization functions
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def match(prediction: str, ground_truth: List[str]) -> bool:
    norm_pred = normalize_answer(prediction)
    for gt in ground_truth:
        norm_gt = normalize_answer(gt)
        if norm_gt and norm_gt in norm_pred:
            return True
    return False

def get_tokens(s: str):
    return normalize_answer(s).split()

def compute_token_prf1(prediction: str, ground_truth_list: list):
    pred_tokens = get_tokens(prediction)
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    max_p, max_r, max_f1 = 0.0, 0.0, 0.0
    for gt in ground_truth_list:
        gt_tokens = get_tokens(gt)
        if not gt_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_p, max_r, max_f1 = precision, recall, f1
    return max_p, max_r, max_f1

def load_full_popqa_test_data(test_file_path: str):
    query_map = {}
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' [SEP] ' not in line:
                continue
            parts = line.split(' [SEP] ')
            q = parts[0].strip()
            p = parts[1].strip()
            if q not in query_map:
                query_map[q] = []
            query_map[q].append(p)
            
    dataset = []
    for idx, (q, psgs) in enumerate(query_map.items(), start=1):
        dataset.append({
            "question_id": f"popqa_full_{idx:05d}",
            "question": q,
            "passages": psgs[:5]
        })
    return dataset

def safe_str(s: str) -> str:
    return s.encode("ascii", "ignore").decode("ascii")

def main():
    print("==================================================================", flush=True)
    print("STARTING HIGHEST TEST ACC T5 EVALUATOR BENCHMARK (1,385 QUERIES)", flush=True)
    print("==================================================================", flush=True)

    # 1. Enforce Real API Mode
    settings.DEBUG_MODE = False
    
    groq_real = bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY) > 10 and not settings.GROQ_API_KEY.startswith("your_"))
    serper_real = bool(settings.SEARCH_API_KEY and len(settings.SEARCH_API_KEY) > 10 and not settings.SEARCH_API_KEY.startswith("your_"))

    print(f"GROQ_API_REAL   = {'YES' if groq_real else 'NO'}", flush=True)
    print(f"SERPER_API_REAL = {'YES' if serper_real else 'NO'}", flush=True)
    print(f"DEBUG_MODE      = {settings.DEBUG_MODE}", flush=True)

    if not groq_real or not serper_real:
        print("[ERROR] Benchmark execution requires valid GROQ_API_KEY and SEARCH_API_KEY in .env!", flush=True)
        sys.exit(1)

    # Output Directory for Highest Test Acc Benchmark
    out_dir = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\results\full_benchmark_highest_acc"
    os.makedirs(out_dir, exist_ok=True)

    test_file_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\CRAG_repo\data\popqa\test_popqa.txt"
    dataset = load_full_popqa_test_data(test_file_path)
    total_queries = len(dataset)

    jsonl_path = os.path.join(out_dir, "full_popqa_highest_acc_predictions.jsonl")

    # Resume Check: Read existing predictions if any
    completed_qids = set()
    existing_records = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    try:
                        rec = json.loads(line.strip())
                        completed_qids.add(rec["question_id"])
                        existing_records.append(rec)
                    except Exception:
                        pass
        print(f"[RESUME] Found {len(completed_qids)} existing processed queries out of {total_queries}.", flush=True)

    # 2. Instantiate Highest Test Acc T5 Evaluator & Controller
    best_evaluator_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_highest_test_acc_final"
    print(f"\n[INIT] Initializing T5RetrievalEvaluator with Highest Test Acc Checkpoint: {best_evaluator_path}", flush=True)
    evaluator_best = T5RetrievalEvaluator(model_path=best_evaluator_path)
    controller = CRAGDecisionController()

    # 3. Fetch PopQA Official Ground Truths
    print("[INIT] Fetching PopQA official ground truth answers from HuggingFace...", flush=True)
    url = "https://huggingface.co/datasets/awinml/popqa_longtail_w_gs/resolve/main/popqa_longtail_w_gs.jsonl"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    popqa_gold_map = {}
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            for line in content.strip().split('\n'):
                rec = json.loads(line)
                q = rec.get('question', '').strip()
                ans = rec.get('answers', [])
                if q:
                    if isinstance(ans, str):
                        ans = [ans]
                    elif isinstance(ans, list):
                        cleaned_ans = []
                        for a in ans:
                            if isinstance(a, str):
                                if a.startswith('[') and a.endswith(']'):
                                    try:
                                        cleaned_ans.extend(json.loads(a))
                                    except Exception:
                                        cleaned_ans.append(a)
                                else:
                                    cleaned_ans.append(a)
                        ans = cleaned_ans
                    popqa_gold_map[q] = ans
        print(f"[INIT] Loaded ground truth answer map for {len(popqa_gold_map)} queries.", flush=True)
    except Exception as e:
        print(f"[WARNING] Ground truth fetch failed: {e}. Match accuracy evaluation will fall back to local test set strings.", flush=True)

    # 4. Process Benchmark Queries
    start_benchmark_time = time.time()
    processed_count = len(completed_qids)

    with open(jsonl_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(dataset, start=1):
            qid = item["question_id"]
            question = item["question"]
            passages = item["passages"]

            if qid in completed_qids:
                continue

            q_start = time.time()
            
            # Setup in-memory local retriever for this query's top-5 passages
            docs = [RetrievedDocument(doc_id=f"{qid}_p{i+1}", text=p, score=0.0) for i, p in enumerate(passages)]
            query_retriever = LocalRetriever(corpus=docs)
            pipeline = CRAGPipeline(retriever=query_retriever, evaluator=evaluator_best, controller=controller)

            try:
                result = pipeline.run(query=question, verbose=False)
                q_latency = time.time() - q_start
                
                # Fetch gold answers
                ans_text = result.final_answer
                gold_answers = popqa_gold_map.get(question, [])
                is_correct = match(ans_text, gold_answers) if gold_answers else False
                p_tok, r_tok, f1_tok = compute_token_prf1(ans_text, gold_answers) if gold_answers else (0.0, 0.0, 0.0)

                scores_list = [float(er.crag_score) for er in result.eval_results] if result.eval_results else []
                max_score_val = float(max(scores_list)) if scores_list else float(result.confidence_score)

                rec = {
                    "question_id": qid,
                    "question": question,
                    "answer": ans_text,
                    "decision": result.decision.value,
                    "max_score": max_score_val,
                    "crag_scores": scores_list,
                    "search_triggered": bool(result.selected_pathway != "INTERNAL_KNOWLEDGE"),
                    "rewritten_query": result.rewritten_query,
                    "latency_seconds": float(q_latency),
                    "gold_answers": gold_answers,
                    "match_correct": is_correct,
                    "token_precision": p_tok,
                    "token_recall": r_tok,
                    "token_f1": f1_tok
                }

                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()

                completed_qids.add(qid)
                existing_records.append(rec)
                processed_count += 1

                print(f"[{processed_count}/{total_queries}] QID: {qid} | Decision: {result.decision.value:9s} | MaxScore: {max_score_val:+.4f} | Match: {'PASS' if is_correct else 'FAIL'} | Latency: {q_latency:.2f}s", flush=True)

            except Exception as ex:
                print(f"[ERROR] Failed query {qid} '{safe_str(question)}': {safe_str(str(ex))}", flush=True)

    total_time = time.time() - start_benchmark_time
    print(f"\n==================================================================", flush=True)
    print(f"BENCHMARK COMPLETED IN {total_time:.2f} SECONDS", flush=True)
    print(f"Processed: {len(existing_records)} / {total_queries} queries", flush=True)
    print(f"==================================================================", flush=True)

    # 5. Compute Benchmark Metrics
    correct_matches = sum(1 for r in existing_records if r.get("match_correct", False))
    match_acc = (correct_matches / len(existing_records)) * 100.0 if existing_records else 0.0

    decisions = [r.get("decision", "CORRECT") for r in existing_records]
    dec_counts = Counter(decisions)

    correct_path_recs = [r for r in existing_records if r.get("decision") == "CORRECT"]
    ambig_path_recs = [r for r in existing_records if r.get("decision") == "AMBIGUOUS"]
    inc_path_recs = [r for r in existing_records if r.get("decision") == "INCORRECT"]

    acc_correct_path = (sum(1 for r in correct_path_recs if r.get("match_correct")) / len(correct_path_recs) * 100.0) if correct_path_recs else 0.0
    acc_ambig_path = (sum(1 for r in ambig_path_recs if r.get("match_correct")) / len(ambig_path_recs) * 100.0) if ambig_path_recs else 0.0
    acc_inc_path = (sum(1 for r in inc_path_recs if r.get("match_correct")) / len(inc_path_recs) * 100.0) if inc_path_recs else 0.0

    import numpy as np
    avg_p = float(np.mean([r.get("token_precision", 0.0) for r in existing_records])) * 100.0
    avg_r = float(np.mean([r.get("token_recall", 0.0) for r in existing_records])) * 100.0
    avg_f1 = float(np.mean([r.get("token_f1", 0.0) for r in existing_records])) * 100.0

    metrics_summary = {
        "evaluator_version": "Highest Test Acc T5-Small Evaluator (t5_evaluator_highest_test_acc_final)",
        "total_queries": len(existing_records),
        "exact_correct_matches": correct_matches,
        "popqa_answer_match_accuracy_pct": match_acc,
        "token_precision_pct": avg_p,
        "token_recall_pct": avg_r,
        "token_f1_score_pct": avg_f1,
        "pathways": {
            "CORRECT": {
                "count": len(correct_path_recs),
                "pct_of_dataset": len(correct_path_recs) / len(existing_records) * 100.0,
                "accuracy_pct": acc_correct_path
            },
            "AMBIGUOUS": {
                "count": len(ambig_path_recs),
                "pct_of_dataset": len(ambig_path_recs) / len(existing_records) * 100.0,
                "accuracy_pct": acc_ambig_path
            },
            "INCORRECT": {
                "count": len(inc_path_recs),
                "pct_of_dataset": len(inc_path_recs) / len(existing_records) * 100.0,
                "accuracy_pct": acc_inc_path
            }
        }
    }

    metrics_json_path = os.path.join(out_dir, "full_popqa_highest_acc_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f_m:
        json.dump(metrics_summary, f_m, indent=2)

    print(f"\nSaved metrics summary to {metrics_json_path}", flush=True)
    print(f"Final Highest-Acc PopQA Answer Accuracy: {match_acc:.2f}% ({correct_matches}/{len(existing_records)})", flush=True)

if __name__ == "__main__":
    main()
