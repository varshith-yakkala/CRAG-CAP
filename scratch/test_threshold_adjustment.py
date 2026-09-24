import sys
import os
import torch

sys.path.append(os.path.abspath("."))

from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator

def get_decision_custom(max_s, upper_thresh=0.5920, lower_thresh=-0.9950):
    if max_s >= upper_thresh:
        return "CORRECT"
    elif max_s < lower_thresh:
        return "INCORRECT"
    else:
        return "AMBIGUOUS"

def main():
    new_model_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_10epochs_final"
    eval_10ep = T5RetrievalEvaluator(model_path=new_model_path)

    test_q = "What is Henry Feilden's occupation?"
    test_p1 = "Henry Master Feilden (21 February 1818 - 5 September 1875) was an English Conservative Party politician."
    test_p2 = "Feilden was born in Hampstead, London. He was educated at Bedford School."

    res1 = eval_10ep.evaluate(test_q, test_p1)
    res2 = eval_10ep.evaluate(test_q, test_p2)

    print("==================================================================")
    print("THRESHOLD ADJUSTMENT FIX FOR 10-EPOCH EVALUATOR OVER-CONFIDENCE")
    print("==================================================================")
    print(f"Passage 1 (Relevant): Score = {res1.crag_score:+.4f}")
    print(f"Passage 2 (Fluff):    Score = {res2.crag_score:+.4f}\n")

    thresholds_to_test = [
        (+0.5920, -0.9950, "Default Original Paper Thresholds"),
        (+0.9900, -0.9900, "Adjusted Threshold (+0.9900 Upper)"),
        (+0.9985, -0.9900, "Calibrated Threshold (+0.9985 Upper)"),
    ]

    for upper, lower, desc in thresholds_to_test:
        d1 = get_decision_custom(res1.crag_score, upper_thresh=upper, lower_thresh=lower)
        d2 = get_decision_custom(res2.crag_score, upper_thresh=upper, lower_thresh=lower)
        print(f"--- {desc} (Upper={upper:+.4f}, Lower={lower:+.4f}) ---")
        print(f"  Passage 1 Decision: {d1}")
        print(f"  Passage 2 Decision: {d2}")

if __name__ == "__main__":
    main()
