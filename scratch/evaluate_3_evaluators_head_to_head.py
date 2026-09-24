import os
import json
import re
import string
import urllib.request
import numpy as np

url = 'https://huggingface.co/datasets/awinml/popqa_longtail_w_gs/resolve/main/popqa_longtail_w_gs.jsonl'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
popqa_gold_map = {}
with urllib.request.urlopen(req) as resp:
    for line in resp.read().decode('utf-8').strip().split('\n'):
        rec = json.loads(line)
        q = rec.get('question', '').strip()
        ans = rec.get('answers', [])
        if q:
            if isinstance(ans, str): ans = [ans]
            popqa_gold_map[q] = ans

def norm(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', ''.join(ch for ch in s.lower() if ch not in string.punctuation)).split())

def match_fn(pred, golds):
    np = norm(pred)
    for g in golds:
        ng = norm(g)
        if ng and ng in np: return True
    return False

p_3ep_file = r'c:\Users\varsh\OneDrive\Desktop\2-2\capstone\results\full_benchmark\full_popqa_predictions.jsonl'
p_10ep_file = r'c:\Users\varsh\OneDrive\Desktop\2-2\capstone\results\full_benchmark_10epochs\full_popqa_10epochs_predictions.jsonl'
p_best_file = r'c:\Users\varsh\OneDrive\Desktop\2-2\capstone\results\full_benchmark_highest_acc\full_popqa_highest_acc_predictions.jsonl'

def load_recs(fp):
    recs = {}
    if os.path.exists(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    recs[r['question_id']] = r
    return recs

recs_3ep = load_recs(p_3ep_file)
recs_10ep = load_recs(p_10ep_file)
recs_best = load_recs(p_best_file)

print(f"3-Epoch Evaluator Processed Queries:          {len(recs_3ep)}")
print(f"10-Epoch Evaluator Processed Queries:         {len(recs_10ep)}")
print(f"Highest Test Acc Evaluator Processed Queries: {len(recs_best)}")

# Find mutual overlap queries
overlap_qids = [qid for qid in recs_best if qid in recs_3ep and qid in recs_10ep]
print(f"\nMutual Overlap Queries for Direct 3-Way Comparison: {len(overlap_qids)}")

if overlap_qids:
    c_3ep, c_10ep, c_best = 0, 0, 0
    d_3ep = {'CORRECT': 0, 'AMBIGUOUS': 0, 'INCORRECT': 0}
    d_10ep = {'CORRECT': 0, 'AMBIGUOUS': 0, 'INCORRECT': 0}
    d_best = {'CORRECT': 0, 'AMBIGUOUS': 0, 'INCORRECT': 0}

    for qid in overlap_qids:
        q = recs_best[qid]['question']
        golds = popqa_gold_map.get(q, [])

        a3 = recs_3ep[qid].get('answer', recs_3ep[qid].get('final_answer', ''))
        a10 = recs_10ep[qid].get('answer', recs_10ep[qid].get('final_answer', ''))
        abest = recs_best[qid].get('answer', recs_best[qid].get('final_answer', ''))

        if match_fn(a3, golds): c_3ep += 1
        if match_fn(a10, golds): c_10ep += 1
        if match_fn(abest, golds): c_best += 1

        d_3ep[recs_3ep[qid].get('decision', 'CORRECT')] += 1
        d_10ep[recs_10ep[qid].get('decision', 'CORRECT')] += 1
        d_best[recs_best[qid].get('decision', 'CORRECT')] += 1

    n = len(overlap_qids)
    print("\n==================================================================")
    print(f"HEAD-TO-HEAD 3-WAY ACCURACY & PATHWAY COMPARISON ({n} QUERIES)")
    print("==================================================================")
    print(f"1. 3-Epoch Evaluator (t5_evaluator_final):             {c_3ep/n*100.0:.2f}% ({c_3ep}/{n})")
    print(f"2. 10-Epoch Evaluator (t5_evaluator_10epochs_final):   {c_10ep/n*100.0:.2f}% ({c_10ep}/{n})")
    print(f"3. Highest Test Acc Evaluator (t5_evaluator_highest): {c_best/n*100.0:.2f}% ({c_best}/{n})")

    print("\n--- Pathway Distributions ---")
    print(f"3-Epoch Evaluator:      CORRECT={d_3ep['CORRECT']} ({d_3ep['CORRECT']/n*100:.1f}%), AMBIGUOUS={d_3ep['AMBIGUOUS']} ({d_3ep['AMBIGUOUS']/n*100:.1f}%), INCORRECT={d_3ep['INCORRECT']} ({d_3ep['INCORRECT']/n*100:.1f}%)")
    print(f"10-Epoch Evaluator:     CORRECT={d_10ep['CORRECT']} ({d_10ep['CORRECT']/n*100:.1f}%), AMBIGUOUS={d_10ep['AMBIGUOUS']} ({d_10ep['AMBIGUOUS']/n*100:.1f}%), INCORRECT={d_10ep['INCORRECT']} ({d_10ep['INCORRECT']/n*100:.1f}%)")
    print(f"Highest Test Acc Eval: CORRECT={d_best['CORRECT']} ({d_best['CORRECT']/n*100:.1f}%), AMBIGUOUS={d_best['AMBIGUOUS']} ({d_best['AMBIGUOUS']/n*100:.1f}%), INCORRECT={d_best['INCORRECT']} ({d_best['INCORRECT']/n*100:.1f}%)")
