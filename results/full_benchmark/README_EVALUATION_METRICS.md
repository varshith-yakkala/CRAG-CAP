# Evaluation Metrics & Benchmark Performance — Custom CRAG Implementation

**File**: `results/full_benchmark/README_EVALUATION_METRICS.md`  
**Date**: August 20, 2026  
**Implementation**: Custom Corrective Retrieval-Augmented Generation ([`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom))  
**Evaluator Checkpoint**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final` (60.5M params)  
**Generative Model**: Groq Cloud API (`openai/gpt-oss-120b`)  
**Web Search Engine**: Real Google Serper.dev API  
**Primary Dataset**: Official PopQA Held-Out Test Split (`CRAG_repo/data/popqa/test_popqa.txt`)

---

## 1. Executive Evaluation Metrics Summary (100% Complete)

| Evaluation Metric | Audited Value | Notes / Provenance Source |
| :--- | :---: | :--- |
| **Total Test Dataset Queries** | **1,385** | Official PopQA Test Split (`test_popqa.txt`) |
| **Successfully Processed Queries** | **1,385** | Stored in `full_popqa_predictions.jsonl` |
| **Dataset Processing Coverage** | **100.00%** | $1385 / 1385$ |
| **Dataset Failure Rate** | **0.00%** | 0 failed queries |
| **PopQA Answer Match Accuracy** | **58.63%** | **812 / 1,385** exact correct matches ($\approx \mathbf{58.6\%}$) |
| **Original Published CRAG Paper** | **53.90%** | Yan et al. (2024) (T5-large 770M + Self-RAG 7B) |
| **Accuracy Improvement Over Paper** | **+4.73%** | **58.63% vs 53.90%** (Driven by Groq 120B LLM) |
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
        583 Queries (42.1%)        661 Queries (47.7%)        141 Queries (10.2%)
       Accuracy: 70.50%            Accuracy: 48.41%           Accuracy: 57.45%
```

| Decision Pathway | Condition / Rule | Processed Count | % of Dataset | Match Correct | Match Incorrect | Empty Preds | Pathway Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CORRECT`** | $\text{MaxScore} \ge +0.5920$ | **583** | **42.09%** | 411 | 172 | 24 | **70.50%** |
| **`AMBIGUOUS`** | $-0.9950 \le \text{MaxScore} < +0.5920$ | **661** | **47.73%** | 320 | 341 | 80 | **48.41%** |
| **`INCORRECT`** | $\text{MaxScore} < -0.9950$ | **141** | **10.18%** | 81 | 60 | 8 | **57.45%** |
| **TOTAL SYSTEM** | — | **1,385** | **100.00%** | **812** | **573** | **112** | **58.63%** |

---

## 3. System Latency & Performance Metrics

| Pipeline Stage / Metric | Mean Latency | Median Latency | P95 Latency | Execution Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Local Retrieval** | ~0.005 s | ~0.004 s | ~0.010 s | In-Memory Index |
| **T5 Evaluator (per passage)** | ~0.239 s | ~0.215 s | ~0.350 s | PyTorch CPU |
| **Serper Web Search** | ~0.450 s | ~0.420 s | ~0.650 s | Real Serper.dev API |
| **Groq 120B Answer Generation** | ~3.717 s | ~3.143 s | ~6.222 s | Real Groq Cloud API |
| **End-to-End Pipeline Latency** | **23.770 s** | **9.897 s** | **81.671 s** | Full System (Includes retry backoffs) |

---

## 4. API Provenance & Real Execution Metrics

```text
GROQ_API_REAL   = YES (openai/gpt-oss-120b)
SERPER_API_REAL = YES (SerperSearchProvider)
MOCK_GROQ_USED  = NO
MOCK_SEARCH_USED= NO
DEBUG_MODE      = False
```

* **Real Groq API Calls**: **1,385 successful generative calls** (0 mock/cached calls).
* **Real Serper API Calls**: **802 successful search requests** (661 AMBIGUOUS + 141 INCORRECT).
* **Mock Provider Invocations**: **0**.

---

## 5. Audit & Deliverable References

* [`POPQA_ANSWER_ACCURACY_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/POPQA_ANSWER_ACCURACY_AUDIT.md): Detailed 1,385-query answer match audit.
* [`FULL_POPQA_BENCHMARK_REPORT_CORRECTED.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/FULL_POPQA_BENCHMARK_REPORT_CORRECTED.md): Complete full-dataset report.
* [`METRICS_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/METRICS_AUDIT.md): Code audit resolving pathway accounting.
* [`FINAL_PROVENANCE_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/FINAL_PROVENANCE_AUDIT.md): API call and model execution authenticity verification.
* [`full_popqa_predictions.jsonl`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_predictions.jsonl): Line-by-line prediction outputs.
* [`full_popqa_metrics_corrected.json`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_metrics_corrected.json): Structured metrics record.
