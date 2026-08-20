# Implementation Comparison: Reference CRAG vs. Custom CRAG System

**Document**: `results/final_benchmark/CRAG_IMPLEMENTATION_COMPARISON.md`  
**Date**: August 19, 2026  
**Reference Repository**: [`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG)  
**Custom Codebase**: [`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom)

---

## 1. Executive Summary

This document provides a code-level architectural and behavioral comparison between the original reference implementation of **Corrective Retrieval-Augmented Generation (CRAG)** ([`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG)) and our custom implementation ([`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom)).

The purpose of this analysis is to establish precisely how closely our system reproduces the original CRAG architecture, control flow, decision mechanisms, and knowledge refinement pathways, and to document every component replacement.

---

## 2. Original CRAG Architecture vs. Our CRAG Architecture

### **Original Control Flow (`HuskyInSalt/CRAG`)**
```text
USER QUERY
    ↓
RETRIEVER (Pre-retrieved passages / BM25 / Dense)
    ↓
RETRIEVAL EVALUATOR (T5-large Cross-Encoder, ~770M params)
    ↓
EVALUATOR SCORING (Continuous logit score in [-1.0, +1.0])
    ↓
CRAG 3-WAY DECISION CONTROLLER (process_flag logic: upper=+0.592, lower=-0.995)
    ↓
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│        CORRECT          │        AMBIGUOUS        │        INCORRECT        │
│                         │                         │                         │
│ Internal Refinement     │ Internal + External     │ Discard Internal        │
│ - Decomposition         │ Refinement              │ - Query Rewriting       │
│ - T5 Sub-strip Filter   │ - Sub-strip Filter      │ - Serper Web Search     │
│ - Recomposition         │ - Serper Web Search     │ - External Filtering    │
└───────────┬─────────────┴────────────┬────────────┴────────────┬────────────┘
            │                          │                         │
            └──────────────────────────┼─────────────────────────┘
                                       ↓
                           GENERATIVE LLM (vLLM Self-RAG 7B / LLaMA-2)
                                       ↓
                                  FINAL ANSWER
```

### **Our Custom Control Flow (`crag_custom/`)**
```text
USER QUERY
    ↓
RETRIEVER (LocalRetriever in-memory corpus interface)
    ↓
RETRIEVAL EVALUATOR (Our T5-small Seq2Seq Checkpoint, ~60.5M params)
    ↓
EVALUATOR SCORING (Decoder logit extraction P(1) -> Score = 2P(1) - 1.0)
    ↓
CRAG 3-WAY DECISION CONTROLLER (CRAGDecisionController: upper=+0.592, lower=-0.995)
    ↓
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│        CORRECT          │        AMBIGUOUS        │        INCORRECT        │
│                         │                         │                         │
│ Internal Refinement     │ Internal + External     │ Discard Internal        │
│ - Decomposition         │ Refinement              │ - Query Rewriting (Groq)│
│ - T5 Sub-strip Filter   │ - Sub-strip Filter      │ - Serper Web Search     │
│ - Recomposition         │ - Serper Web Search     │ - External Filtering    │
└───────────┬─────────────┴────────────┬────────────┴────────────┬────────────┘
            │                          │                         │
            └──────────────────────────┼─────────────────────────┘
                                       ↓
                           GENERATIVE LLM (Groq API openai/gpt-oss-120b)
                                       ↓
                                  FINAL ANSWER
```

---

## 3. Component-by-Component Comparison Table

| Component | Original CRAG (`HuskyInSalt/CRAG`) | Our Custom CRAG (`crag_custom/`) | Status |
| :--- | :--- | :--- | :---: |
| **Overall Architecture** | Sequential retrieval evaluation, 3-way routing, knowledge refinement, generation | Sequential retrieval evaluation, 3-way routing, knowledge refinement, generation | **SAME** |
| **Retrieval Backend** | Pre-retrieved text files / BM25 / Dense retriever | `LocalRetriever` in-memory corpus interface | **REPLACED** |
| **Retrieval Evaluator** | Cross-encoder sequence classifier | Fine-tuned seq2seq model with probability adapter | **REPLACED** |
| **Evaluator Model** | `T5ForSequenceClassification` (`t5-large`, ~770M params) | `T5ForConditionalGeneration` (`t5-small`, ~60.5M params) | **REPLACED** |
| **Evaluator Input** | `Question [SEP] Passage` | `evaluate relevance: question: <Q> context: <P>` | **MODIFIED** |
| **Evaluator Output** | Continuous logit score in $[-1.0, +1.0]$ | Logits for `"1"` vs `"0"` mapped to CRAG score $2P(1) - 1.0$ | **MODIFIED** |
| **Score Calculation** | Direct logit score extraction | $P(1) = \frac{\exp(\text{logit}_1)}{\exp(\text{logit}_1) + \exp(\text{logit}_0)} \implies \text{Score} = 2P(1) - 1.0$ | **MODIFIED** |
| **Threshold Logic** | `process_flag` (`upper=+0.5920`, `lower=-0.9950`) | `CRAGDecisionController` (`upper=+0.5920`, `lower=-0.9950`) | **SAME** |
| **CORRECT Path** | Internal knowledge decomposition & T5 filtering | Internal knowledge decomposition & T5 filtering | **SAME** |
| **AMBIGUOUS Path** | Internal refinement + Serper search + external filtering | Internal refinement + Serper search + external filtering | **SAME** |
| **INCORRECT Path** | Discard internal -> Query rewrite -> Serper search | Discard internal -> Query rewrite -> Serper search | **SAME** |
| **Passage Decomposition** | `fixed_num`, `excerption`, `selection` modes | `fixed_num`, `excerption`, `selection` modes | **SAME** |
| **Knowledge Filtering** | T5 evaluator sub-strip relevance filtering | T5 evaluator sub-strip relevance filtering | **SAME** |
| **Knowledge Recomposition** | Concatenation of valid sub-strips | Concatenation of valid sub-strips | **SAME** |
| **Internal Knowledge** | `internal_knowledge_preparation.py` | `crag_custom/knowledge/internal.py` | **SAME** |
| **External Knowledge** | `external_knowledge_preparation.py` | `crag_custom/knowledge/external.py` | **SAME** |
| **Query Rewriting** | Prompted LLM (`gpt-3.5-turbo` / LLaMA-2) | `GroqQuestionRewriter` (`groq/compound-mini`) | **REPLACED** |
| **Web Search Engine** | Serper.dev Google Search API | `SerperSearchProvider` (Serper.dev Google Search API) | **SAME** |
| **Generative LLM** | vLLM local engine (`selfrag_llama2_7b`) | `GroqGenerator` via Groq API (`openai/gpt-oss-120b`) | **REPLACED** |
| **Pipeline Orchestration**| `CRAG_Inference.py` execution script | `CRAGPipeline` class (`crag_pipeline.py`) | **SAME** |

---

## 4. Evaluator Model & Score Mechanism Comparison

* **Original Evaluator (`HuskyInSalt/CRAG`)**:
  * Class: `T5ForSequenceClassification` (`num_labels=1`).
  * Parameter Count: ~770 Million (`t5-large`).
  * Score Output: Single regression logit output representing degree of relevance.
* **Our Evaluator (`crag_custom/evaluator/t5_evaluator.py`)**:
  * Class: `T5ForConditionalGeneration`.
  * Parameter Count: `60,511,616` (~60.5M parameters, `t5-small`).
  * Score Output: Probability adapter extracting decoder logits for token ID `209` (`"1"`) vs ID `3` (`"0"`):
    $$P(1) = \frac{\exp(\text{logit}_1)}{\exp(\text{logit}_1) + \exp(\text{logit}_0)}, \quad \text{CRAG\_score} = 2P(1) - 1.0 \in [-1.0, +1.0]$$
  * **Classification**: An adapter-based replacement that produces a score bounded in $[-1.0, +1.0]$ compatible with the reference CRAG decision thresholds.

---

## 5. Decision Controller Logic Comparison

Side-by-side verification of decision code:

```python
# ORIGINAL CRAG (CRAG_Inference.py: process_flag)
def process_flag(scores, n_docs, threshold1, threshold2):
    # threshold1 = +0.592, threshold2 = -0.995
    flags = []
    for score in scores:
        if score >= threshold1:
            flags.append('2')  # CORRECT
        elif score >= threshold2:
            flags.append('1')  # AMBIGUOUS
        else:
            flags.append('0')  # INCORRECT
    if '2' in flags:
        return 2  # CORRECT
    elif '1' in flags:
        return 1  # AMBIGUOUS
    else:
        return 0  # INCORRECT
```

```python
# OUR CUSTOM CRAG (crag_custom/corrective/controller.py)
class CRAGDecisionController:
    def decide(self, eval_results):
        max_score = max([r.crag_score for r in eval_results])
        if max_score >= self.upper_threshold:        # +0.5920
            return DecisionResult(CRAGDecision.CORRECT, max_score)
        elif max_score >= self.lower_threshold:      # -0.9950
            return DecisionResult(CRAGDecision.AMBIGUOUS, max_score)
        else:
            return DecisionResult(CRAGDecision.INCORRECT, max_score)
```
*Verification*: Both implementations perform **mathematically identical max-score thresholding**.

---

## 6. Real Execution Trace Verification

Verification of real executions from [`results/real_crag_validation/real_validation_trace.txt`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/real_crag_validation/real_validation_trace.txt):

1. **Trace 1 ("What is George Rankin's occupation?")**:
   * Evaluator Output: $P(1) = 0.903745 \implies \text{CRAG Score} = +0.807489$
   * Decision: $+0.807489 \ge +0.5920 \implies \mathbf{CORRECT}$
   * Pathway: Refined internal knowledge, generated final answer via Groq.
2. **Trace 2 ("In what city was Billy Carlson born?")**:
   * Evaluator Output: $P(1) = 0.663334 \implies \text{CRAG Score} = +0.326668$
   * Decision: $-0.9950 \le +0.326668 < +0.5920 \implies \mathbf{AMBIGUOUS}$
   * Pathway: Executed real Serper web search, combined internal + external knowledge, generated answer via Groq.
3. **Trace 3 ("What is the capital of Australia?")**:
   * Evaluator Output: $P(1) = 0.003290 \implies \text{CRAG Score} = -0.993419$
   * Decision: $-0.9950 \le -0.993419 < +0.5920 \implies \mathbf{AMBIGUOUS}$
   * Pathway: Executed real Serper web search, combined internal + external knowledge, generated answer via Groq.

---

## 7. Exact List of Changes (Substitutions)

1. **`T5-large` SequenceClassifier (770M)** $\longrightarrow$ **`T5-small` ConditionalGenerator (60.5M)**
2. **Regression Logit Output** $\longrightarrow$ **Decoder Logit Softmax Adapter ($2P(1)-1$)**
3. **Local vLLM `selfrag_llama2_7b` Generator** $\longrightarrow$ **Groq API (`openai/gpt-oss-120b`)**
4. **Local Query Rewriter LLM** $\longrightarrow$ **Groq API (`groq/compound-mini`)**
5. **File-based Pre-retrieved Dataset Loader** $\longrightarrow$ **`LocalRetriever` In-Memory Interface**

---

## 8. Exact List of Preserved Components

1. **Sequential Control Flow**: Evaluator $\rightarrow$ Decision Controller $\rightarrow$ Corrective Pathway $\rightarrow$ Generator.
2. **3-Way Routing Mechanism**: `CORRECT`, `AMBIGUOUS`, and `INCORRECT` paths.
3. **Threshold Boundaries**: Upper threshold $= +0.5920$, Lower cutoff $= -0.9950$.
4. **Passage Decomposition Modes**: `selection`, `excerption`, and `fixed_num`.
5. **Sub-strip Relevance Filtering**: Using T5 evaluator to strip irrelevant text chunks.
6. **External Search Engine**: Google Serper.dev API (`SerperSearchProvider`).
7. **Knowledge Recomposition**: Concatenation of filtered internal and external web strips.

---

## 9. Final Equivalence Classification

### **Classification**: **`FAITHFUL ARCHITECTURAL REIMPLEMENTATION WITH COMPONENT SUBSTITUTIONS`**

*Rationale*: The core CRAG control flow, 3-way decision controller, corrective routing logic (`CORRECT`, `AMBIGUOUS`, `INCORRECT`), passage decomposition (`selection`, `excerption`, `fixed_num`), T5 sub-strip filtering, and recomposition are 100% preserved. The component implementations for evaluator model size/class (T5-small seq2seq vs T5-large cross-encoder), generator LLM provider (Groq API vs local vLLM), and retrieval backend have been substituted.
