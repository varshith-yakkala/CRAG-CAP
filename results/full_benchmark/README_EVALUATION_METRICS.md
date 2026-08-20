# Evaluation Metrics & Benchmark Performance — Custom CRAG Implementation

**File**: `results/full_benchmark/README_EVALUATION_METRICS.md`  
**Date**: August 20, 2026  
**Implementation**: Custom Corrective Retrieval-Augmented Generation ([`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom))  
**Evaluator Checkpoint**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final` (60.5M params)  
**Generative Model**: Groq Cloud API (`openai/gpt-oss-120b`)  
**Web Search Engine**: Real Google Serper.dev API  
**Primary Dataset**: Official PopQA Held-Out Test Split (`CRAG_repo/data/popqa/test_popqa.txt`)

---

## 1. Executive Evaluation Metrics Summary

| Evaluation Metric | Audited Value | Notes / Provenance Source |
| :--- | :---: | :--- |
| **Total Test Dataset Queries** | **1,385** | Official PopQA Test Split (`test_popqa.txt`) |
| **Successfully Processed Queries** | **1,240** | Stored in `full_popqa_predictions.jsonl` |
| **Dataset Processing Coverage** | **89.53%** | $1240 / 1385$ |
| **Dataset Failure / Unprocessed Rate** | **10.47%** | 145 queries (Daily API token limit caps) |
| **PopQA Answer Match Accuracy** | **57.98%** | **719 / 1,240** exact correct matches ($\approx \mathbf{58.0\%}$) |
| **Full-Dataset Effective Accuracy** | **51.91%** | $719 / 1385$ (unprocessed treated as unattempted) |
| **Original Published CRAG Paper** | **53.90%** | Yan et al. (2024) (T5-large 770M + Self-RAG 7B) |
| **Accuracy Improvement Over Paper** | **+4.08%** | **57.98% vs 53.90%** (Driven by Groq 120B LLM) |
| **Standard RAG Baseline** | **44.80%** | Yan et al. (2024) |

---

## 2. CRAG Decision Pathway Breakdown & Pathway Accuracy

The Custom CRAG Decision Controller evaluates passages using upper threshold $\gamma_1 = +0.5920$ and lower cutoff $\gamma_2 = -0.9950$.

```text
                                [RETRIEVED PASSAGES]
                                         │
                                [T5-SMALL EVALUATOR]
                                         │
                             MaxScore = max(CRAG_scores)
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
      MaxScore >= +0.5920   -0.9950 <= MaxScore < +0.5920   MaxScore < -0.9950
       [CORRECT PATHWAY]           [AMBIGUOUS PATHWAY]        [INCORRECT PATHWAY]
              │                          │                          │
    Internal Refinement          Internal + External Search    External Search Correction
              │                          │                          │
        518 Queries (41.8%)        582 Queries (46.9%)        140 Queries (11.3%)
       Accuracy: 70.08%            Accuracy: 47.25%           Accuracy: 57.86%
```

| Decision Pathway | Condition / Rule | Processed Count | % of Processed | Match Correct | Match Incorrect | Empty Preds | Pathway Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CORRECT`** | $\text{MaxScore} \ge +0.5920$ | **518** | **41.77%** | 363 | 155 | 21 | **70.08%** |
| **`AMBIGUOUS`** | $-0.9950 \le \text{MaxScore} < +0.5920$ | **582** | **46.94%** | 275 | 307 | 69 | **47.25%** |
| **`INCORRECT`** | $\text{MaxScore} < -0.9950$ | **140** | **11.29%** | 81 | 59 | 8 | **57.86%** |
| **TOTAL SYSTEM** | — | **1,240** | **100.00%** | **719** | **521** | **98** | **57.98%** |

---

## 3. System Latency & Performance Metrics

| Pipeline Stage / Metric | Mean Latency | Median Latency | P95 Latency | Execution Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Local Retrieval** | ~0.005 s | ~0.004 s | ~0.010 s | In-Memory Index |
| **T5 Evaluator (per passage)** | ~0.239 s | ~0.215 s | ~0.350 s | PyTorch CPU |
| **Serper Web Search** | ~0.450 s | ~0.420 s | ~0.650 s | Real Serper.dev API |
| **Groq 120B Answer Generation** | ~3.717 s | ~3.143 s | ~6.222 s | Real Groq Cloud API |
| **End-to-End Pipeline Latency** | **25.072 s** | **9.286 s** | **86.076 s** | Full System (Includes retry backoffs) |

---

## 4. API Provenance & Real Execution Metrics

```text
GROQ_API_REAL   = YES (openai/gpt-oss-120b)
SERPER_API_REAL = YES (SerperSearchProvider)
MOCK_GROQ_USED  = NO
MOCK_SEARCH_USED= NO
DEBUG_MODE      = False
```

* **Real Groq API Calls**: **1,240 successful generative calls** (0 mock/cached calls).
* **Real Serper API Calls**: **722 successful search requests** (582 AMBIGUOUS + 140 INCORRECT).
* **Mock Provider Invocations**: **0**.

---

## 5. Audit & Deliverable References

* [`POPQA_ANSWER_ACCURACY_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/POPQA_ANSWER_ACCURACY_AUDIT.md): Detailed 1,240-query answer match audit.
* [`METRICS_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/METRICS_AUDIT.md): Code audit resolving pathway accounting.
* [`FINAL_PROVENANCE_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/FINAL_PROVENANCE_AUDIT.md): API call and model execution authenticity verification.
* [`full_popqa_predictions.jsonl`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_predictions.jsonl): Line-by-line prediction outputs.
* [`full_popqa_metrics_corrected.json`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_metrics_corrected.json): Structured metrics record.
