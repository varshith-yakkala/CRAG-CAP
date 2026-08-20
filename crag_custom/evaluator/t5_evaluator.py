import os
import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration
from crag_custom.evaluator.base import BaseRetrievalEvaluator
from crag_custom.evaluator.schemas import EvaluationResult
from crag_custom.config.settings import settings

class T5RetrievalEvaluator(BaseRetrievalEvaluator):
    """
    Custom T5 Retrieval Evaluator Adapter.
    Model: T5ForConditionalGeneration (t5-small, ~60M params) fine-tuned for sequence-to-sequence relevance classification.
    Input Format: "evaluate relevance: question: <QUESTION> context: <PASSAGE>"
    Target Format: "1" = Relevant, "0" = Irrelevant

    SCORE ADAPTATION METHODOLOGY:
      The original CRAG evaluator (T5ForSequenceClassification) produced continuous regression scores.
      Our T5-small evaluator generates target tokens "1" or "0".
      To retain confidence information, we extract decoder logits for target tokens "1" and "0":
        logit_1 = decoder_logit(token_id("1"))
        logit_0 = decoder_logit(token_id("0"))
        P("1") = exp(logit_1) / (exp(logit_1) + exp(logit_0))
        P("0") = 1 - P("1")

      Our score adaptation formula maps P("1") in [0, 1] to CRAG-compatible signed score space:
        CRAG_score = 2 * P("1") - 1.0  in [-1.0, +1.0]
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_path = model_path or settings.T5_MODEL_PATH
        self.tokenizer_path = tokenizer_path or settings.T5_TOKENIZER_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[T5RetrievalEvaluator] Loading T5 model from: {self.model_path}")
        
        # Load Tokenizer with fallback logic
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            print(f"[T5RetrievalEvaluator] Fallback to tokenizer path: {self.tokenizer_path} (Reason: {e})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
        # Load T5 Model
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Resolve target token IDs for '1' and '0' dynamically
        tokens_1 = self.tokenizer.encode("1", add_special_tokens=False)
        tokens_0 = self.tokenizer.encode("0", add_special_tokens=False)
        
        self.id_1 = tokens_1[0]
        self.id_0 = tokens_0[0]
        
        print(f"[T5RetrievalEvaluator] Dynamic Token IDs: '1' -> {self.id_1}, '0' -> {self.id_0}")

    def evaluate(self, question: str, passage: str) -> EvaluationResult:
        """
        Evaluates a single (question, passage) pair.
        Returns logit-based P(1), P(0), CRAG-compatible score, label, and raw text output.
        """
        input_text = f"evaluate relevance: question: {question} context: {passage}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # 1. Generate text output
            gen_ids = self.model.generate(**inputs, max_length=5)
            raw_output = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            
            # 2. Forward pass to extract decoder logits for first output token
            decoder_start_id = (
                self.model.config.decoder_start_token_id
                if self.model.config.decoder_start_token_id is not None
                else 0
            )
            decoder_input_ids = torch.tensor([[decoder_start_id]], device=self.device)
            
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                decoder_input_ids=decoder_input_ids
            )
            
            # Logits for first generated token: shape (1, 1, vocab_size)
            first_token_logits = outputs.logits[0, 0]
            
            logit_1 = first_token_logits[self.id_1].item()
            logit_0 = first_token_logits[self.id_0].item()
            
            # 3. Softmax probability P(1) over tokens '1' vs '0'
            denom = torch.exp(first_token_logits[self.id_1]) + torch.exp(first_token_logits[self.id_0])
            prob_relevant = (torch.exp(first_token_logits[self.id_1]) / denom).item()
            prob_irrelevant = 1.0 - prob_relevant
            
            # 4. Our Score Adaptation: CRAG_score = 2*P(1) - 1 in [-1.0, +1.0]
            crag_score = 2.0 * prob_relevant - 1.0
            
            label = "1" if prob_relevant >= 0.5 or raw_output == "1" else "0"
            is_relevant = (label == "1")
            
        return EvaluationResult(
            label=label,
            is_relevant=is_relevant,
            relevance_probability=prob_relevant,
            crag_score=crag_score,
            raw_output=raw_output,
            logit_1=logit_1,
            logit_0=logit_0,
            metadata={"p_0": prob_irrelevant}
        )

    def evaluate_batch(self, question: str, passages: List[str]) -> List[EvaluationResult]:
        """
        Evaluates each retrieved passage individually.
        """
        results = []
        for p in passages:
            results.append(self.evaluate(question, p))
        return results
