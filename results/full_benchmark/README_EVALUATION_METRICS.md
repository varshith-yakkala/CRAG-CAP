# Evaluation Metrics & Benchmark Performance — Custom CRAG Implementation

**File**: `results/full_benchmark/README_EVALUATION_METRICS.md`  
**Date**: August 22, 2026  
**Implementation**: Custom Corrective Retrieval-Augmented Generation ([`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom))  
**Evaluator Checkpoint**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final` (60.5M params)  
**Generative Model**: Groq Cloud API (`openai/gpt-oss-120b`)  
**Web Search Engine**: Real Google Serper.dev API  
**Primary Dataset**: Official PopQA Held-Out Test Split (`CRAG_repo/data/popqa/test_popqa.txt`)

---

## 1. Executive Evaluation Metrics Summary (100% Complete)

| Evaluation Metric Category | Specific Metric Name | Tested Value | Benchmark Notes |
| :--- | :--- | :---: | :--- |
| **Dataset Coverage** | **Total Test Queries Processed** | **1,385 / 1,385** | **100.00% Dataset Coverage** (0.00% failure rate) |
| **Exact Match (EM)** | **PopQA Answer Match Accuracy** | **58.63%** | **812 / 1,385** exact correct matches ($\approx \mathbf{58.6\%}$) |
| **QA Token Overlap** | **Token Recall** | **59.25%** | Percentage of target gold entity tokens in answer |
| **QA Token Overlap** | **Token Precision** | **11.65%** | Ratio of gold entity tokens vs total answer words |
| **QA Token Overlap** | **Token F1-Score** | **18.87%** | Harmonic mean of token precision and recall |
| **Text Similarity** | **ROUGE-L F1-Score** | **18.87%** | Longest Common Subsequence similarity |
| **Evaluator Model** | **Validation Classification Accuracy**| **94.40%** | Binary relevance score accuracy (T5-small 60.5M) |
| **Evaluator Model** | **Relevance F1-Score** | **94.18%** | Precision: 92.80%, Recall: 95.60% |
| **Paper Baseline** | **Original CRAG Paper (Yan 2024)** | **53.90%** | T5-large 770M + Self-RAG LLaMA-2 7B |
| **Improvement** | **Accuracy Advantage Over Paper** | **+4.73%** | **58.63% vs 53.90%** (Driven by Groq 120B LLM) |

---

## 2. CRAG Decision Pathway Breakdown & Pathway Metrics

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
       Token Recall: 71.90%        Token Recall: 48.55%       Token Recall: 57.09%
```

| Decision Pathway | Condition / Rule | Count | % of Dataset | Match Accuracy | Token Recall | Token Precision | Token F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CORRECT`** | $\text{MaxScore} \ge +0.5920$ | **583** | **42.09%** | **70.50%** | **71.90%** | 14.02% | **22.66%** |
| **`AMBIGUOUS`** | $-0.9950 \le \text{MaxScore} < +0.5920$ | **661** | **47.73%** | **48.41%** | **48.55%** | 9.46% | **15.38%** |
| **`INCORRECT`** | $\text{MaxScore} < -0.9950$ | **141** | **10.18%** | **57.45%** | **57.09%** | 12.07% | **19.59%** |
| **TOTAL SYSTEM** | — | **1,385** | **100.00%** | **58.63%** | **59.25%** | **11.65%** | **18.87%** |

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
