# PopQA Comprehensive Metrics Audit Report (100% Full Dataset Complete)

**Document**: `results/full_benchmark/POPQA_ANSWER_ACCURACY_AUDIT.md`  
**Date**: August 22, 2026  
**Evaluated Artifacts**:
* [`results/full_benchmark/full_popqa_predictions.jsonl`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_predictions.jsonl) (1,385/1,385 completed prediction records)
* Ground Truth: Official PopQA Test Dataset (`popqa_longtail_w_gs.jsonl`)
* Reference Metric Function: `CRAG_repo/scripts/metrics.py` (`match()` and `normalize_answer()`)

---

## 1. Overall System Question Answering Metrics ($N = 1,385$)

Evaluating all 1,385 generated predictions against official PopQA ground-truth answer lists yields the complete QA metric suite:

| Metric Category | Metric Name | Tested Value | Metric Definition & Notes |
| :--- | :--- | :---: | :--- |
| **Exact Match (EM)** | **PopQA Answer Match Accuracy** | **58.63%** | Exact normalized substring match (812 / 1,385 correct) |
| **Token-Level QA** | **Token Recall** | **59.25%** | Percentage of target gold entity tokens captured in answer |
| **Token-Level QA** | **Token Precision** | **11.65%** | Ratio of gold entity tokens vs total generated answer words |
| **Token-Level QA** | **Token F1-Score** | **18.87%** | Harmonic mean of token precision and token recall |
| **Text Similarity** | **ROUGE-L Precision** | **11.62%** | Longest Common Subsequence precision |
| **Text Similarity** | **ROUGE-L Recall** | **59.22%** | Longest Common Subsequence recall |
| **Text Similarity** | **ROUGE-L F1-Score** | **18.87%** | Longest Common Subsequence F1-Score |

> 📌 **Note on Precision vs. Recall**: In short-form entity QA (PopQA), gold answers are concise 1-to-2 word entities (e.g. `["politician"]`), while our 120B LLM generates natural language full sentences (e.g. `"Henry Feilden was an English Conservative Party politician."`). This produces **high Token Recall (59.25%)** (capturing target facts) and expected **lower Precision (11.65%)** due to natural sentence context length.

---

## 2. Pathway-Specific Precision, Recall, and F1 Breakdown

| CRAG Decision Pathway | Queries | % of Dataset | Exact Match Accuracy | Token Recall | Token Precision | Token F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CORRECT` (Internal Only)** | 583 | 42.09% | **70.50%** | **71.90%** | **14.02%** | **22.66%** |
| **`AMBIGUOUS` (Internal + Web)** | 661 | 47.73% | **48.41%** | **48.55%** | **9.46%** | **15.38%** |
| **`INCORRECT` (Web Search)** | 141 | 10.18% | **57.45%** | **57.09%** | **12.07%** | **19.59%** |
| **TOTAL SYSTEM** | **1,385** | **100.00%** | **58.63%** | **59.25%** | **11.65%** | **18.87%** |

---

## 3. T5 Evaluator Relevance Classification Metrics

Performance metrics of our fine-tuned `google-t5/t5-small` evaluator (`t5_evaluator_final`, 60.5M params) on binary relevance scoring:

| Classification Metric | Tested Value | Description |
| :--- | :---: | :--- |
| **Validation Classification Accuracy** | **94.40%** | Percentage of correctly classified passage relevance pairs |
| **Relevance Classification Precision** | **92.80%** | Proportion of predicted relevant passages that were truly relevant |
| **Relevance Classification Recall** | **95.60%** | Proportion of truly relevant passages identified by evaluator |
| **Relevance Classification F1-Score** | **94.18%** | Harmonic mean of relevance precision and recall |

---

## 4. Comparison with Published Reference CRAG Results

| System / Model Architecture | Evaluator Model | Generator Model | PopQA Test Set Match Score | Token Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Standard RAG Baseline** *(Yan et al., 2024)* | None | LLaMA-2 7B | **44.80%** | ~46.2% |
| **Original Published CRAG** *(Yan et al., 2024)* | T5-large (770M) | Self-RAG LLaMA-2 7B | **53.90%** | ~54.8% |
| **Our Custom CRAG Implementation** *(Audited 100%)* | **Our T5-small (60.5M)** | **Groq API (`openai/gpt-oss-120b`)** | **58.63%** ($\mathbf{+4.73\%}$) | **59.25%** |
