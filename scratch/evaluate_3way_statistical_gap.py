import sys
import os
import time
import json
import numpy as np

sys.path.append(os.path.abspath("."))

from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator

def main():
    path_3ep = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final"
    path_10ep = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_10epochs_final"
    path_best = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_highest_test_acc_final"

    print("[1/3] Loading 3-Epoch Evaluator...")
    eval_3ep = T5RetrievalEvaluator(model_path=path_3ep)

    print("[2/3] Loading 10-Epoch Evaluator...")
    eval_10ep = T5RetrievalEvaluator(model_path=path_10ep)

    print("[3/3] Loading Highest Test Acc Evaluator...")
    eval_best = T5RetrievalEvaluator(model_path=path_best)

    test_file_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\CRAG_repo\data\popqa\test_popqa.txt"
    
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

    queries = list(query_map.items())[:150] # 150 test queries (750 passage evaluations)
    print(f"\nEvaluating score distributions on {len(queries)} test queries (750 passage pairs)...")

    dec_3ep = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0}
    dec_10ep = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0}
    dec_best = {"CORRECT": 0, "AMBIGUOUS": 0, "INCORRECT": 0}

    def get_decision(max_s):
        if max_s >= 0.5920:
            return "CORRECT"
        elif max_s < -0.9950:
            return "INCORRECT"
        else:
            return "AMBIGUOUS"

    scores_3ep = []
    scores_10ep = []
    scores_best = []

    for q, psgs in queries:
        s3 = [eval_3ep.evaluate(q, p).crag_score for p in psgs[:5]]
        s10 = [eval_10ep.evaluate(q, p).crag_score for p in psgs[:5]]
        sbest = [eval_best.evaluate(q, p).crag_score for p in psgs[:5]]

        max_3 = max(s3)
        max_10 = max(s10)
        max_best = max(sbest)

        scores_3ep.append(max_3)
        scores_10ep.append(max_10)
        scores_best.append(max_best)

        dec_3ep[get_decision(max_3)] += 1
        dec_10ep[get_decision(max_10)] += 1
        dec_best[get_decision(max_best)] += 1

    n = len(queries)
    print("\n==================================================================")
    print("STATISTICAL SCORE & PATHWAY COMPARISON (150 TEST QUERIES / 750 PASSAGES)")
    print("==================================================================")
    print("Mean Max CRAG Score:")
    print(f"  3-Epoch Evaluator:             {np.mean(scores_3ep):+.4f}")
    print(f"  10-Epoch Evaluator:            {np.mean(scores_10ep):+.4f}")
    print(f"  Highest Test Acc Evaluator:    {np.mean(scores_best):+.4f}")

    print("\nPathway Decisions (% Distribution):")
    print(f"  3-Epoch Evaluator:             CORRECT={dec_3ep['CORRECT']/n*100:.1f}%, AMBIGUOUS={dec_3ep['AMBIGUOUS']/n*100:.1f}%, INCORRECT={dec_3ep['INCORRECT']/n*100:.1f}%")
    print(f"  10-Epoch Evaluator:            CORRECT={dec_10ep['CORRECT']/n*100:.1f}%, AMBIGUOUS={dec_10ep['AMBIGUOUS']/n*100:.1f}%, INCORRECT={dec_10ep['INCORRECT']/n*100:.1f}%")
    print(f"  Highest Test Acc Evaluator:    CORRECT={dec_best['CORRECT']/n*100:.1f}%, AMBIGUOUS={dec_best['AMBIGUOUS']/n*100:.1f}%, INCORRECT={dec_best['INCORRECT']/n*100:.1f}%")

if __name__ == "__main__":
    main()
