# PopQA Answer Match Accuracy Audit Report

**Document**: `results/full_benchmark/POPQA_ANSWER_ACCURACY_AUDIT.md`  
**Date**: August 20, 2026  
**Evaluated Artifacts**:
* [`results/full_benchmark/full_popqa_predictions.jsonl`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_predictions.jsonl)
* Ground Truth: Official PopQA Test Dataset (`popqa_longtail_w_gs.jsonl`)
* Reference Metric Function: `CRAG_repo/scripts/metrics.py` (`match()` and `normalize_answer()`)

---

## 1. Executive Answer Match Audit Results

Using the official reference CRAG `match()` normalization function, each of the 1,240 generated predictions in `full_popqa_predictions.jsonl` was evaluated against its corresponding ground-truth answer list from the official PopQA test benchmark.

### Overall PopQA Answer Accuracy (Denominator = 1,240 Processed Queries):

* **Total Processed Queries Evaluated**: **1,240**
* **Exact Correct Count (`match() == True`)**: **719**
* **Exact Incorrect Count (`match() == False`)**: **521**
* **Unscorable Queries (Missing Gold Answer)**: **0**
* **Audited PopQA Answer Match Accuracy**: **57.98%** ($\approx \mathbf{58.0\%}$)

---

## 2. Accuracy Breakdown by CRAG Decision Pathway

| CRAG Decision Pathway | Total Queries | Exact Correct (`match=True`) | Exact Incorrect (`match=False`) | Empty Predictions | Pathway Match Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CORRECT (Internal Knowledge)** | 518 | 363 | 155 | 21 | **70.08%** |
| **AMBIGUOUS (Combined Knowledge)** | 582 | 275 | 307 | 69 | **47.25%** |
| **INCORRECT (External Search)** | 140 | 81 | 59 | 8 | **57.86%** |
| **TOTAL SYSTEM** | **1,240** | **719** | **521** | **98** | **57.98%** |

---

## 3. Comparison with Published Reference CRAG Results

| System / Model Architecture | Evaluator Model | Generator Model | PopQA Test Set Match Score |
| :--- | :---: | :---: | :---: |
| **Standard RAG Baseline** *(Yan et al., 2024)* | None | LLaMA-2 7B | **44.80%** |
| **Original Published CRAG** *(Yan et al., 2024)* | T5-large (770M) | Self-RAG LLaMA-2 7B | **53.90%** |
| **Our Custom CRAG Implementation** *(Audited)* | **Our T5-small (60.5M)** | **Groq API (`openai/gpt-oss-120b`)** | **57.98%** ($\mathbf{+4.08\%}$ over paper) |

---

## 4. Full Dataset Effective Accuracy (Denominator = 1,385 Total Benchmark Queries)

If the 145 unprocessed queries (due to daily API token caps) are treated as unattempted/incorrect across the full 1,385-query dataset:

* **Full Dataset Effective Accuracy**: $719 / 1385 = \mathbf{51.91\%}$
* **Processing Coverage**: $1240 / 1385 = \mathbf{89.53\%}$
* **Failure / Unprocessed Rate**: $145 / 1385 = \mathbf{10.47\%}$
