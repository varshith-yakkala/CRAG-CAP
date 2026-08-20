# Full-Dataset PopQA CRAG Benchmark Report (100% Complete)

**Date**: August 20, 2026  
**Dataset File**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\CRAG_repo\data\popqa\test_popqa.txt`  
**Evaluator Checkpoint**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final` (60.5M params)  
**Generative Model**: Groq API (`openai/gpt-oss-120b`)  
**Search Engine**: Real Google Serper.dev API  
**Audit Document**: [`POPQA_ANSWER_ACCURACY_AUDIT.md`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/POPQA_ANSWER_ACCURACY_AUDIT.md)

---

## 1. Executive Metrics Summary (100% Full Dataset Complete)

- **Dataset**: Official PopQA Held-Out Test Split (`test_popqa.txt`)
- **Total Test Queries**: **1,385 Queries**
- **Queries Successfully Processed**: **1,385 Queries (100.00% Coverage)**
- **Unprocessed / Failed Queries**: **0 Queries (0.00% Failure Rate)**
- **PopQA Answer Match Accuracy**: **58.63%** (812 / 1,385 exact correct matches)

### Pathway Distribution & Accuracy (N = 1,385):
- **CORRECT Pathway Count (%)**: **583 (42.09%)** | Accuracy: **70.50%**
- **AMBIGUOUS Pathway Count (%)**: **661 (47.73%)** | Accuracy: **48.41%**
- **INCORRECT Pathway Count (%)**: **141 (10.18%)** | Accuracy: **57.45%**
- **External Search Trigger Count (%)**: **802 (57.91%)**
- **Sum of Processed Percentages**: **100.00%** ($583 + 661 + 141 = 1,385$)

### Performance & Latency Metrics:
- **Mean Latency**: **23.770 seconds**
- **Median Latency**: **9.897 seconds**
- **P95 Latency**: **81.671 seconds**
- **Groq Real**: **YES** (1,385 successful calls)
- **Serper Real**: **YES** (802 successful search calls)
- **Mocks Used**: **NO** (`DEBUG_MODE = False`)

---

## 2. Comparison with Published Original CRAG Paper (Yan et al., 2024)

| Metric | Published CRAG Paper *(Yan et al., 2024)* | Our Full-Dataset Custom CRAG *(100% Complete)* |
| :--- | :---: | :---: |
| **Dataset** | PopQA Test Set | **PopQA Held-Out Test Split (`test_popqa.txt`)** |
| **Total Test Queries** | 1,385 Queries | **1,385 Benchmark Queries (100.0% Processed)** |
| **Processing Coverage** | 100.0% | **100.0%** (1,385 / 1,385) |
| **Evaluator Model** | T5-large (770M) | **Our Custom T5-small (60.5M)** |
| **Generator Model** | Self-RAG LLaMA-2 7B | **Groq API (`openai/gpt-oss-120b`)** |
| **PopQA Match Score** | **53.90%** | **58.63%** ($\mathbf{+4.73\%}$ higher accuracy) |
| **Median Pipeline Latency** | *N/A* | **9.897 s** |
| **External Search Trigger Rate** | *N/A* | **57.91% (802 / 1,385)** |

---

## 3. Provenance & Component Summary

1. **Evaluator Model**: `google-t5/t5-small` fine-tuned to 60.5M parameters (**12.8x parameter reduction** compared to 770M T5-large). Evaluates passages in ~209ms on CPU.
2. **Generative LLM**: Groq Cloud API (`openai/gpt-oss-120b`) providing 120B parameter reasoning capacity (**17x parameter scaling** compared to 7B LLaMA-2).
3. **External Web Search**: Google Serper.dev API providing live organic Google search snippets.
4. **All 1,385 prediction records** are saved line-by-line in [`results/full_benchmark/full_popqa_predictions.jsonl`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/results/full_benchmark/full_popqa_predictions.jsonl).
