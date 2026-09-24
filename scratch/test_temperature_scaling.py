import sys
import os
import torch

sys.path.append(os.path.abspath("."))

from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator

def evaluate_with_temperature(evaluator, question: str, passage: str, temperature: float = 1.0):
    inputs = evaluator.tokenizer(
        f"evaluate relevance: question: {question} context: {passage}",
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(evaluator.device)
    
    with torch.no_grad():
        decoder_start_id = evaluator.model.config.decoder_start_token_id or 0
        decoder_input_ids = torch.tensor([[decoder_start_id]], device=evaluator.device)
        outputs = evaluator.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            decoder_input_ids=decoder_input_ids
        )
        logits = outputs.logits[0, 0]
        
        logit_1 = logits[evaluator.id_1].item()
        logit_0 = logits[evaluator.id_0].item()
        
        # Apply Temperature Scaling
        l1_scaled = logit_1 / temperature
        l0_scaled = logit_0 / temperature
        
        denom = torch.exp(torch.tensor(l1_scaled)) + torch.exp(torch.tensor(l0_scaled))
        prob_1 = (torch.exp(torch.tensor(l1_scaled)) / denom).item()
        crag_score = 2.0 * prob_1 - 1.0
        
        return logit_1, logit_0, prob_1, crag_score

def main():
    new_model_path = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_10epochs_final"
    eval_10ep = T5RetrievalEvaluator(model_path=new_model_path)

    test_q = "What is Henry Feilden's occupation?"
    test_p1 = "Henry Master Feilden (21 February 1818 - 5 September 1875) was an English Conservative Party politician."
    test_p2 = "Feilden was born in Hampstead, London. He was educated at Bedford School."

    print("==================================================================")
    print("TEMPERATURE SCALING FIX FOR 10-EPOCH EVALUATOR OVER-CONFIDENCE")
    print("==================================================================")

    for temp in [1.0, 2.0, 3.0, 5.0, 10.0]:
        _, _, p1, s1 = evaluate_with_temperature(eval_10ep, test_q, test_p1, temperature=temp)
        _, _, p2, s2 = evaluate_with_temperature(eval_10ep, test_q, test_p2, temperature=temp)
        print(f"\n--- Temperature T = {temp:.1f} ---")
        print(f"  Passage 1 (Relevant): Score={s1:+.4f}, Prob={p1:.4f} => Decision: {'CORRECT' if s1 >= 0.5920 else ('INCORRECT' if s1 < -0.9950 else 'AMBIGUOUS')}")
        print(f"  Passage 2 (Fluff):    Score={s2:+.4f}, Prob={p2:.4f} => Decision: {'CORRECT' if s2 >= 0.5920 else ('INCORRECT' if s2 < -0.9950 else 'AMBIGUOUS')}")

if __name__ == "__main__":
    main()
