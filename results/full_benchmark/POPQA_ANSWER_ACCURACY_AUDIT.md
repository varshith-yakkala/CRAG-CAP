# PopQA Answer Match Accuracy Audit Report (100% Full Dataset Complete)

**Document**: `results/full_benchmark/POPQA_ANSWER_ACCURACY_AUDIT.md`  
**Date**: August 20, 2026  
**Evaluated Artifacts**:
* [`results/full_benchmark/full_popqa_predictions.jsonl`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_predictions.jsonl) (1,385/1,385 completed prediction records)
* Ground Truth: Official PopQA Test Dataset (`popqa_longtail_w_gs.jsonl`)
* Reference Metric Function: `CRAG_repo/scripts/metrics.py` (`match()` and `normalize_answer()`)

---

## 1. Executive Answer Match Audit Results (100% Dataset Complete)

Using the official reference CRAG `match()` normalization function, all 1,385 generated predictions in `full_popqa_predictions.jsonl` were evaluated against their corresponding ground-truth answer lists from the official PopQA test benchmark split.

### Overall PopQA Answer Accuracy (100% Full Dataset Coverage):

* **Total Test Dataset Queries Evaluated**: **1,385 / 1,385 (100.00% Coverage)**
* **Exact Correct Match Count (`match() == True`)**: **812**
* **Exact Incorrect Match Count (`match() == False`)**: **573**
* **Unscorable Queries (Missing Gold Answer)**: **0**
* **Audited 100% PopQA Answer Match Accuracy**: **58.63%** ($\approx \mathbf{58.6\%}$)

---

## 2. Accuracy Breakdown by CRAG Decision Pathway (N = 1,385)

| CRAG Decision Pathway | Total Queries | % of Dataset | Exact Correct (`match=True`) | Exact Incorrect (`match=False`) | Empty Predictions | Pathway Match Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CORRECT (Internal Knowledge)** | 583 | 42.09% | 411 | 172 | 24 | **70.50%** |
| **AMBIGUOUS (Combined Knowledge)** | 661 | 47.73% | 320 | 341 | 80 | **48.41%** |
| **INCORRECT (External Search)** | 141 | 10.18% | 81 | 60 | 8 | **57.45%** |
| **TOTAL SYSTEM** | **1,385** | **100.00%** | **812** | **573** | **112** | **58.63%** |

---

## 3. Comparison with Published Reference CRAG Results

| System / Model Architecture | Evaluator Model | Generator Model | PopQA Test Set Match Score |
| :--- | :---: | :---: | :---: |
| **Standard RAG Baseline** *(Yan et al., 2024)* | None | LLaMA-2 7B | **44.80%** |
| **Original Published CRAG** *(Yan et al., 2024)* | T5-large (770M) | Self-RAG LLaMA-2 7B | **53.90%** |
| **Our Custom CRAG Implementation** *(Audited 100%)* | **Our T5-small (60.5M)** | **Groq API (`openai/gpt-oss-120b`)** | **58.63%** ($\mathbf{+4.73\%}$ over paper) |

---

## 4. Latency & External Search Trigger Summary

* **External Search Trigger Rate**: **57.91%** (802 / 1,385 queries)
* **Mean Pipeline Latency**: **23.770 seconds** (Includes retry backoffs)
* **Median Pipeline Latency**: **9.897 seconds**
* **P95 Pipeline Latency**: **81.671 seconds**
* **Real Groq API Calls**: **1,385 successful calls** (`openai/gpt-oss-120b`)
* **Real Serper API Calls**: **802 successful search requests** (Google Serper.dev)
* **Mock Calls**: **0** (`DEBUG_MODE = False`)
