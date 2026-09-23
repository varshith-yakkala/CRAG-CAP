import sys
import os
sys.path.append(os.path.abspath("."))

import torch
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator

old_model_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final"
new_model_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_10epochs_final"

print("Loading 3-epoch evaluator...")
eval_3ep = T5RetrievalEvaluator(model_path=old_model_path)

print("Loading 10-epoch evaluator...")
eval_10ep = T5RetrievalEvaluator(model_path=new_model_path)

test_q = "What is Henry Feilden's occupation?"
test_p1 = "Henry Master Feilden (21 February 1818 - 5 September 1875) was an English Conservative Party politician."
test_p2 = "Feilden was born in Hampstead, London. He was educated at Bedford School."

res_3ep_1 = eval_3ep.evaluate(test_q, test_p1)
res_3ep_2 = eval_3ep.evaluate(test_q, test_p2)

res_10ep_1 = eval_10ep.evaluate(test_q, test_p1)
res_10ep_2 = eval_10ep.evaluate(test_q, test_p2)

print("\n=== COMPARISON ON SAMPLE QUESTION ===")
print(f"Q: {test_q}")
print(f"Passage 1 (Relevant):")
print(f"  3-Epoch:  CRAG Score={res_3ep_1.crag_score:+.4f}, Prob={res_3ep_1.relevance_probability:.4f}")
print(f"  10-Epoch: CRAG Score={res_10ep_1.crag_score:+.4f}, Prob={res_10ep_1.relevance_probability:.4f}")
print(f"Passage 2 (Irrelevant/Fluff):")
print(f"  3-Epoch:  CRAG Score={res_3ep_2.crag_score:+.4f}, Prob={res_3ep_2.relevance_probability:.4f}")
print(f"  10-Epoch: CRAG Score={res_10ep_2.crag_score:+.4f}, Prob={res_10ep_2.relevance_probability:.4f}")
