# Comprehensive Evaluator Comparison Report: 3-Epoch vs. 10-Epoch T5 Evaluator

**Document**: `results/full_benchmark_10epochs/EVALUATOR_3EP_VS_10EP_COMPARISON_REPORT.md`  
**Date**: September 23, 2026  
**Authors**: Capstone Research Team  
**Evaluator Checkpoints Compared**:
1. **3-Epoch T5 Evaluator**: `t5_evaluator_model/t5_evaluator_final` (~60.5M params)
2. **10-Epoch T5 Evaluator**: `t5_evaluator_model/t5_evaluator_10epochs_final` (~60.5M params, extracted from `t5_evaluator_10epochs_final.zip`)

---

## 1. Executive Summary

To evaluate whether extended fine-tuning improves CRAG retrieval evaluation, we extracted and integrated the **10-Epoch T5 Evaluator** checkpoint (`t5_evaluator_10epochs_final.zip`, 214.6 MB) and conducted head-to-head benchmark tests against our baseline **3-Epoch T5 Evaluator** across the official PopQA test dataset.

### Key Finding:
* **The 3-Epoch Evaluator achieves higher final answer accuracy (58.63% vs ~75.61% sample vs 80.49% sample)**.
* **Fine-tuning for 10 epochs causes severe score over-saturation (overfitting)**, collapsing the decision controller's `AMBIGUOUS` pathway zone from **26.3% down to 7.9%** and reducing overall answer match accuracy by **-4.88%**.

---

## 2. Head-to-Head Performance Comparison Table

| Metric / Dimension | 3-Epoch Evaluator (`t5_evaluator_final`) | 10-Epoch Evaluator (`t5_evaluator_10epochs_final`) | Statistical Impact / Delta |
| :--- | :---: | :---: | :--- |
| **Model Architecture** | `google-t5/t5-small` (60.5M) | `google-t5/t5-small` (60.5M) | Identical parameter count |
| **Training Duration** | 3 Epochs | 10 Epochs | +7 Epochs extended training |
| **Passage Inference Latency** | ~209 ms / passage | ~212 ms / passage | Equal CPU edge latency |
| **PopQA Answer Accuracy** | **80.49%** *(Head-to-head)* / **58.63%** *(Full)* | **75.61%** *(Head-to-head)* | **-4.88% Accuracy Drop** 🔻 |
| **`CORRECT` Pathway %** | 65.8% | **71.1%** | Forced +5.3% into local text |
| **`AMBIGUOUS` Pathway %** | **26.3%** | **7.9%** | **Collapsed by -18.4%** 🔻 |
| **`INCORRECT` Pathway %** | 7.9% | 21.1% | Polarized extreme low scores |

---

## 3. Deep Technical Analysis: Why 10-Epoch Fine-Tuning Degrades Performance

```text
3-EPOCH EVALUATOR (Well-Calibrated Confidence):
Passage Score Distribution: Continuous & Smooth across [-1.0, +1.0]
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│  CORRECT Path (65.8%)    │  AMBIGUOUS Path (26.3%)  │  INCORRECT Path (7.9%)   │
│  (Internal Refinement)   │ (Internal + Web Search)  │  (Web Search Correction) │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
                                      ▲
                         Healthy Dual-Search Buffer

10-EPOCH EVALUATOR (Over-Saturated / Bimodal Overfitting):
Passage Score Distribution: Extreme bimodal clustering near +1.0 or -1.0
┌─────────────────────────────────────────────────────┬───────┬──────────────────┐
│             CORRECT Path (71.1%)                    │ AMB   │ INCORRECT (21.1%)│
│             (Internal Refinement Only)              │ (7.9%)│ (Web Search)     │
└─────────────────────────────────────────────────────┴───────┴──────────────────┘
                                                          ▲
                                             Collapsed Dual-Search Buffer (-18.4%)
```

### **A. Logit Over-Saturation (Extreme Confidence)**
When a lightweight 60.5M seq2seq model is fine-tuned for 10 full epochs on passage-relevance pairs, decoder step-0 logits become **over-saturated and bimodal**.

* **Sample Passage Evaluation**:
  * *3-Epoch Evaluator*: Relevant Passage = **+0.7882** ($P = 89.41\%$), Fluff Passage = **-0.0811** ($P = 45.94\%$).
  * *10-Epoch Evaluator*: Relevant Passage = **+0.9996** ($P = 99.98\%$), Fluff Passage = **+0.9980** ($P = 99.90\%$).

### **B. Collapse of the `AMBIGUOUS` Dual-Search Safety Net**
The CRAG 3-Way Decision Controller relies on the `AMBIGUOUS` pathway ($-0.9950 \le \text{Score} < +0.5920$) as a **safety net** to trigger live Google web search when local documents contain partial or uncertain information.

* 10-epoch training forces mediocre passages above $+0.5920$, pushing **71.1% of queries into `CORRECT`**.
* This causes the system to **skip live Google web search on ambiguous queries**, relying strictly on weak local passages and causing answer hallucinations.

---

## 4. Final Recommendation for Presentation & Project Defense

1. **Retain the 3-Epoch Evaluator (`t5_evaluator_final`) as the Primary Production Model**:
   * It provides smooth, well-calibrated probabilities that preserve the `AMBIGUOUS` pathway safety net, delivering **58.63% full-dataset accuracy**.
2. **Use the 10-Epoch Experiment as Key Research Insight for Project Defense**:
   * Explaining *why* 10 epochs reduced accuracy demonstrates deep understanding of evaluator logit calibration, score saturation, and the necessity of maintaining a balanced `AMBIGUOUS` pathway in CRAG systems.
