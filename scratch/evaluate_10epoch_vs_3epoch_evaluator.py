import sys
import os
import time
import json
import numpy as np
import torch

sys.path.append(os.path.abspath("."))

from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator

def main():
    old_model_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final"
    new_model_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_10epochs_final"

    print("==================================================================")
    print("DIRECT EVALUATOR COMPARISON: 3-EPOCH VS 10-EPOCH CHECKPOINT")
    print("==================================================================")

    print("\n[1/2] Loading 3-Epoch Evaluator...")
    eval_3ep = T5RetrievalEvaluator(model_path=old_model_path)

    print("\n[2/2] Loading 10-Epoch Evaluator...")
    eval_10ep = T5RetrievalEvaluator(model_path=new_model_path)

    test_file_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\CRAG_repo\data\popqa\test_popqa.txt"
    
    # Load test dataset (Question, Top-5 Passages)
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

    queries = list(query_map.items())
    print(f"\nLoaded {len(queries)} test queries with candidate passages.")

    # Evaluate max scores across queries for 3-Epoch vs 10-Epoch
    max_scores_3ep = []
    max_scores_10ep = []
    
    probs_3ep = []
    probs_10ep = []

    decisions_3ep = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0}
    decisions_10ep = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0}

    def get_decision(max_s):
        if max_s >= 0.5920:
            return "CORRECT"
        elif max_s < -0.9950:
            return "INCORRECT"
        else:
            return "AMBIGUOUS"

    start_t = time.time()
    # Evaluate first 200 queries (1,000 passage evaluations) for direct statistical analysis
    eval_sample_size = min(200, len(queries))
    print(f"Evaluating direct passage scores on {eval_sample_size} test queries (1,000 passage pairs)...")

    for idx, (q, psgs) in enumerate(queries[:eval_sample_size], start=1):
        # 3-Epoch Scores
        res_3ep = [eval_3ep.evaluate(q, p) for p in psgs[:5]]
        scores_3ep = [r.crag_score for r in res_3ep]
        max_3ep = max(scores_3ep)
        max_scores_3ep.append(max_3ep)
        probs_3ep.extend([r.relevance_probability for r in res_3ep])
        decisions_3ep[get_decision(max_3ep)] += 1

        # 10-Epoch Scores
        res_10ep = [eval_10ep.evaluate(q, p) for p in psgs[:5]]
        scores_10ep = [r.crag_score for r in res_10ep]
        max_10ep = max(scores_10ep)
        max_scores_10ep.append(max_10ep)
        probs_10ep.extend([r.relevance_probability for r in res_10ep])
        decisions_10ep[get_decision(max_10ep)] += 1

        if idx % 50 == 0 or idx == eval_sample_size:
            print(f" Evaluated {idx}/{eval_sample_size} queries...")

    eval_time = time.time() - start_t

    print("\n==================================================================")
    print("DIRECT EVALUATOR COMPARISON RESULTS (200 TEST QUERIES / 1,000 PASSAGES)")
    print("==================================================================")
    print(f"Evaluation Latency: {eval_time:.2f}s (~{eval_time/eval_sample_size/5*1000:.1f}ms per passage)")
    print(f"\nMean Max Score:")
    print(f"  3-Epoch Evaluator:  {np.mean(max_scores_3ep):+.4f}")
    print(f"  10-Epoch Evaluator: {np.mean(max_scores_10ep):+.4f}")

    print(f"\nMean Relevance Probability:")
    print(f"  3-Epoch Evaluator:  {np.mean(probs_3ep)*100:.2f}%")
    print(f"  10-Epoch Evaluator: {np.mean(probs_10ep)*100:.2f}%")

    print(f"\nPathway Distribution (Decision Controller Routing):")
    print(f"  3-Epoch Evaluator:")
    print(f"    - CORRECT   (MaxScore >= +0.5920): {decisions_3ep['CORRECT']} ({decisions_3ep['CORRECT']/eval_sample_size*100:.1f}%)")
    print(f"    - AMBIGUOUS (-0.9950 <= Score < +0.5920): {decisions_3ep['AMBIGUOUS']} ({decisions_3ep['AMBIGUOUS']/eval_sample_size*100:.1f}%)")
    print(f"    - INCORRECT (MaxScore < -0.9950): {decisions_3ep['INCORRECT']} ({decisions_3ep['INCORRECT']/eval_sample_size*100:.1f}%)")
    
    print(f"  10-Epoch Evaluator:")
    print(f"    - CORRECT   (MaxScore >= +0.5920): {decisions_10ep['CORRECT']} ({decisions_10ep['CORRECT']/eval_sample_size*100:.1f}%)")
    print(f"    - AMBIGUOUS (-0.9950 <= Score < +0.5920): {decisions_10ep['AMBIGUOUS']} ({decisions_10ep['AMBIGUOUS']/eval_sample_size*100:.1f}%)")
    print(f"    - INCORRECT (MaxScore < -0.9950): {decisions_10ep['INCORRECT']} ({decisions_10ep['INCORRECT']/eval_sample_size*100:.1f}%)")

if __name__ == "__main__":
    main()
