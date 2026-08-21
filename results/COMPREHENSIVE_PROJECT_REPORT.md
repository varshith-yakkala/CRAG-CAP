# Comprehensive Research & Implementation Report: Custom Lightweight Corrective Retrieval-Augmented Generation (CRAG)

**Project Repository**: [`https://github.com/varshith-yakkala/CRAG-CAP.git`](https://github.com/varshith-yakkala/CRAG-CAP.git)  
**Document**: `results/COMPREHENSIVE_PROJECT_REPORT.md`  
**Date**: August 21, 2026  
**Authors**: Capstone Research Team  

---

## 1. Executive Overview

This report documents the step-by-step design, implementation, fine-tuning, integration, and full-dataset benchmark evaluation of our **Custom Corrective Retrieval-Augmented Generation (CRAG)** framework. 

Our system addresses the core limitation of standard RAG—blind reliance on low-quality or irrelevant retrieved documents—by implementing a lightweight retrieval evaluator, a 3-way decision controller, internal passage refinement (decomposition, filtering, recomposition), and external web search query rewriting.

### Key Benchmark Results (Official PopQA Test Dataset, $N = 1,385$ Queries):
* **PopQA Answer Match Accuracy**: **58.63%** (812 / 1,385 exact matches) — outperforming the published 2024 CRAG paper (**53.90%**) by **+4.73%**.
* **Evaluator Parameter Compression**: **12.8x parameter reduction** (**60.5M** T5-small vs **770M** T5-large).
* **Evaluator Inference Speed**: **~209 ms – 239 ms per passage** on standard PyTorch CPU.
* **Full Dataset Coverage**: **100.00%** (1,385 / 1,385 queries completed live with 0 mock calls).

---

## 2. Step-by-Step Implementation Walkthrough

```text
[Step 1: Dataset & Evaluator Training] ──> [Step 2: Score Adapter & 3-Way Controller]
                                                               │
[Step 4: Groq 120B + Serper Integration] <── [Step 3: Knowledge Processing Modules]
                                                               │
[Step 5: Full-Dataset 1,385 Benchmark]   ──> [Step 6: Forensic Audit & Accuracy Validation]
```

### **Step 1: Dataset Preparation & Evaluator Fine-Tuning**
1. We utilized the official PopQA training dataset (`train_popqa.txt`) containing retrieved passages paired with binary relevance labels (`1` for relevant, `0` for irrelevant).
2. We fine-tuned a `google-t5/t5-small` seq2seq model (`crag_model_training.ipynb`) to predict token `'1'` (Token ID `209`) for relevant context vs `'0'` (Token ID `3`) for irrelevant context.
3. Fine-tuning ran for 3 epochs with AdamW ($\text{lr} = 5\times 10^{-4}$), achieving **94.40% relevance classification accuracy** on the validation set.
4. Saved final checkpoint to `t5_evaluator_model/t5_evaluator_final`.

### **Step 2: Score Adapter & 3-Way Decision Controller**
1. **Score Adapter Implementation** (`crag_custom/evaluator/t5_evaluator.py`):
   Extracted raw decoder step-0 logits for token `'1'` ($L_1$) and `'0'` ($L_0$), computing normalized relevance probability $P(1)$:
   $$P(1) = \frac{\exp(L_1)}{\exp(L_1) + \exp(L_0)}$$
   $$\text{CRAG\_score} = 2.0 \times P(1) - 1.0 \quad \in [-1.0, +1.0]$$
2. **Decision Controller Implementation** (`crag_custom/corrective/controller.py`):
   Evaluates $\text{MaxScore} = \max(S_1, S_2, \dots, S_k)$ for top $k=5$ passages:
   * **`CORRECT`** ($\text{MaxScore} \ge +0.5920$): Trigger internal passage refinement; skip web search.
   * **`INCORRECT`** ($\text{MaxScore} < -0.9950$): Discard internal passages; trigger external search.
   * **`AMBIGUOUS`** ($-0.9950 \le \text{MaxScore} < +0.5920$): Combine internal refined context + external search snippets.

### **Step 3: Knowledge Processing Modules**
1. **Passage Decomposition** (`crag_custom/knowledge/decomposition.py`):
   Splits retrieved passages into fine-grained sub-strips using `selection` mode (~1-3 sentences).
2. **Relevance Filtering** (`crag_custom/knowledge/filtering.py`):
   Evaluates each sub-strip using T5 evaluator; discards sub-strips scoring below lower cutoff ($\gamma_2 = -0.9950$).
3. **Knowledge Recomposition** (`crag_custom/knowledge/recomposition.py`):
   Concatenates surviving relevant sub-strips into a clean context string.

### **Step 4: Groq LLM & Google Serper API Integration**
1. **Groq LLM Client** (`crag_custom/llm/groq_client.py`):
   Integrated Groq Cloud API with `openai/gpt-oss-120b` (120B parameters). Configured dynamic multi-key auto-rotation (4 API keys) with automatic 429 rate limit backoff.
2. **Query Rewriter** (`crag_custom/llm/query_rewriter.py`):
   Uses Groq 120B to rewrite raw user questions into concise search engine keywords.
3. **Google Serper Provider** (`crag_custom/search/web_search.py`):
   Queries Google Serper.dev API to fetch live organic Google search titles, snippets, and URLs.

### **Step 5: Full-Dataset 1,385 Benchmark Execution**
Executed [`crag_custom/scripts/run_full_popqa_benchmark.py`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom/scripts/run_full_popqa_benchmark.py) live across **all 1,385 test questions** in `CRAG_repo/data/popqa/test_popqa.txt`. Stored predictions in `full_popqa_predictions.jsonl`.

### **Step 6: Metric Audit & Answer Accuracy Validation**
Audited all 1,385 generated predictions against official PopQA ground-truth answer lists using reference CRAG `match()` normalization function from `CRAG_repo/scripts/metrics.py`.

---

## 3. Why We Chose These Specific Models & Infrastructure

### **A. Evaluator Model: Why `google-t5/t5-small` (60.5M Parameters)?**
1. **Parameter Efficiency**: At **60.5M parameters**, T5-small is **12.8x smaller** than T5-large (770M).
2. **CPU Edge Deployability**: Evaluates candidate passages in **~209ms – 239ms per passage** on standard CPU without requiring expensive GPU infrastructure.
3. **High Discriminative Accuracy**: Achieved **94.40% relevance classification accuracy** on PopQA passage pairs, proving that a 60.5M seq2seq model is sufficient for binary relevance scoring when properly fine-tuned.

### **B. Generator Model: Why Groq API (`openai/gpt-oss-120b`)?**
1. **Model Capacity (120B Parameters)**: Provides **17x model capacity scaling** over 7B models (LLaMA-2 7B).
2. **Superior Entity Extraction**: In factual QA datasets (PopQA), a 120B model excels at parsing dense context and extracting exact entity answers (occupations, birthplaces, dates) while ignoring distractor text.
3. **LPU Speed & Zero Infrastructure Cost**: Groq's LPU architecture generates completions in ~3 seconds per query, eliminating local GPU hosting requirements.

### **C. Search Provider: Why Google Serper API (`Serper.dev`)?**
1. **Live Google Search Indexing**: Retrieves live, high-ranking Google Knowledge Graph and organic search snippets.
2. **Structured JSON Output**: Delivers clean snippet arrays easily parsed by the knowledge recomposition module.

---

## 4. What We Improved (Comparison with Original CRAG Paper)

| Metric / Dimension | Published CRAG Paper *(Yan et al., 2024)* | Our Custom CRAG Implementation | Improvement / Benefit |
| :--- | :---: | :---: | :--- |
| **PopQA Answer Match Accuracy** | **53.90%** | **58.63%** | **+4.73% Higher Accuracy** across 1,385 queries |
| **Evaluator Model Size** | T5-large (770M params) | **T5-small (60.5M params)** | **12.8x Parameter Reduction** |
| **Evaluator Deployment** | GPU Server Cluster | **PyTorch CPU (~209ms/pass)** | **Zero High-End GPU Requirement** |
| **Generative LLM Capacity** | Self-RAG LLaMA-2 7B | **Groq `openai/gpt-oss-120b`** | **17x Model Capacity Scaling** |
| **Query Rewriting Quality** | LLaMA-2 7B | **Groq 120B** | **Cleaner Google Search Keywords** |
| **INCORRECT Pathway Recovery** | Baseline | **57.45% Match Accuracy** | Recovers answers when local retrieval fails completely |

---

## 5. Architectural Comparison: Original CRAG vs. Our Custom CRAG

### **What Was Preserved (100% Structural Fidelity)**
1. **3-Way Control Flow**: Exact `CORRECT`, `AMBIGUOUS`, and `INCORRECT` pathway logic.
2. **Threshold Boundaries**: Upper threshold $\gamma_1 = +0.5920$, lower cutoff $\gamma_2 = -0.9950$.
3. **Passage Processing Pipeline**: Sub-strip passage decomposition, T5 relevance filtering, and recomposition.

### **What Was Changed & Why**

| Component | Original CRAG | Our Custom CRAG | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Evaluator Architecture** | `T5ForSequenceClassification` (770M) | **`T5ForConditionalGeneration` (60.5M) + Logit Adapter** | Replaced heavy sequence classifier with lightweight seq2seq model + logit probability adapter $2P(1) - 1.0$. |
| **Generator Architecture** | Local Self-RAG 7B | **Groq Cloud API (`openai/gpt-oss-120b`)** | Replaced local 7B model with 120B frontier model for superior answer synthesis. |
| **Search Engine** | Bing / DuckDuckGo (2023) | **Google Serper API (`Serper.dev`)** | Replaced legacy wrappers with real-time Google Search indexing. |

---

## 6. Final 1,385-Query Benchmark Results

```text
===================================================================================
100% FULL POPQA BENCHMARK EVALUATION RESULTS (1,385 / 1,385 QUERIES)
===================================================================================
Total Test Dataset Queries:            1,385  (PopQA test_popqa.txt)
Successfully Processed Queries:        1,385  (100.00% Full Dataset Coverage)
Failed / Unprocessed Queries:          0      (0.00% Failure Rate)

Final PopQA Answer Match Accuracy:     58.63% (812 / 1,385 Exact Correct Matches)
Original Published CRAG Paper:         53.90% (Yan et al., 2024)
Standard RAG Baseline:                 44.80% (Yan et al., 2024)
Accuracy Improvement Over Paper:       +4.73% (Driven by Groq 120B LLM Capacity)

Pathway Distribution & Accuracy (N = 1,385):
- CORRECT Pathway (Internal Only):     583 Queries (42.09%) | Accuracy: 70.50%
- AMBIGUOUS Pathway (Internal+Web):    661 Queries (47.73%) | Accuracy: 48.41%
- INCORRECT Pathway (Web Correction):  141 Queries (10.18%) | Accuracy: 57.45%

Latency & Performance:
- Mean Pipeline Latency:               23.770 seconds
- Median Pipeline Latency:             9.897 seconds
- Real Groq LLM API Calls:             1,385 Successful Calls (openai/gpt-oss-120b)
- Real Serper Search API Calls:        802 Successful Requests (Google Serper.dev)
- Mock Provider Calls:                 0 (100% Real Execution)
===================================================================================
```
