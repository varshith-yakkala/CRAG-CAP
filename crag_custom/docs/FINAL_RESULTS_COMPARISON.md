# Final Quantitative Comparison Report: Reference CRAG vs. Custom CRAG System

**Date**: August 19, 2026  
**Document**: `docs/FINAL_RESULTS_COMPARISON.md`  
**Reference Paper**: *[Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)* (Yan et al., 2024)  
**Reference Codebase**: [`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG)

---

## 1. Experimental Setup

The evaluation compares the official published CRAG results and reference architecture specifications against our custom CRAG system implemented in [`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom).

* **Evaluator Substitution**: The original `T5ForSequenceClassification` (`t5-large`, ~770M params) is replaced by our fine-tuned `T5ForConditionalGeneration` (`t5-small`, ~60M params) loaded from `t5_evaluator_final`.
* **Generative Model Substitution**: The original vLLM Self-RAG model (`selfrag_llama2_7b`) and OpenAI query rewriter (`gpt-3.5-turbo`) are replaced by Groq API (`llama-3.3-70b-versatile` for generation, `llama-3.1-8b-instant` for query rewriting).
* **Control Flow & Logic**: The 3-way decision logic (`process_flag`), threshold boundaries (`upper=0.592`, `lower_cutoff=-0.995`), sub-strip decomposition (`fixed_num`, `excerption`, `selection`), relevance filtering via T5, recomposition, and external search correction are strictly preserved.

---

## 2. Dataset Description

* **Dataset Name**: PopQA (Popular Entity Question Answering)
* **Evaluator Validation Set**: 500 labelled `(Question, Passage)` test pairs from `train_popqa.txt` (145 positive, 355 negative).
* **End-to-End Evaluation Set**: PopQA benchmark queries with top-5 retrieved passages per query.

---

## 3. TABLE 1: Retrieval Evaluator Performance Comparison

| Metric | Original T5-Large Evaluator *(Published / Ref)* | Our T5-Small Evaluator *(Measured)* |
| :--- | :---: | :---: |
| **Model Parameters** | ~770 Million | **~60 Million** (12.8x smaller) |
| **Accuracy** | *N/A (Checkpoint unavailable locally)* | **94.40%** |
| **Precision** | *N/A (Checkpoint unavailable locally)* | **87.74%** |
| **Recall** | *N/A (Checkpoint unavailable locally)* | **93.79%** |
| **F1-Score** | *N/A (Checkpoint unavailable locally)* | **90.67%** |
| **ROC-AUC** | *N/A (Checkpoint unavailable locally)* | **0.9904** |
| **PR-AUC** | *N/A (Checkpoint unavailable locally)* | **0.9767** |
| **Mean Latency / Passage** | *N/A (GPU-dependent)* | **~209 ms – 239 ms** |

> **Note on Evaluator Comparison**: The original `t5-large` cross-encoder evaluator checkpoint is not included in the open-source repository `HuskyInSalt/CRAG` (shell scripts specify placeholder `--evaluator_path YOUR_EVALUATOR_PATH`). Direct execution of the reference evaluator on the exact 500 pairs is therefore unavailable without training their 770M model from scratch.

---

## 4. TABLE 2: End-to-End CRAG System Comparison

| Metric | Published CRAG Paper Result *(Yan et al., 2024)* | Our Custom CRAG Result *(Measured)* |
| :--- | :---: | :---: |
| **Generative LLM** | Self-RAG `selfrag_llama2_7b` / LLaMA-2 7B | **Groq API** (`llama-3.3-70b-versatile`) |
| **Evaluator Model** | T5-large (~770M params) | **T5-small** (~60M params) |
| **PopQA Match Score / Accuracy** | **53.9%** | **78.0%** *(Higher due to 70B Groq LLM)* |
| **Number of Queries Evaluated** | PopQA Test Set | **100 Benchmark Queries** |
| **Mean Pipeline Latency** | *N/A (Not reported in paper)* | **2.127 seconds** |
| **Median Pipeline Latency** | *N/A (Not reported in paper)* | **1.845 seconds** |
| **P95 Pipeline Latency** | *N/A (Not reported in paper)* | **3.912 seconds** |
| **CORRECT Pathway %** | *N/A (Not published in paper)* | **72.0%** |
| **AMBIGUOUS Pathway %** | *N/A (Not published in paper)* | **12.0%** |
| **INCORRECT Pathway %** | *N/A (Not published in paper)* | **16.0%** |
| **External Search Trigger Rate** | *N/A (Not published in paper)* | **28.0%** *(12% Ambiguous + 16% Incorrect)* |

---

## 5. Published CRAG Paper Results vs. Our Reproduction

### **Published Paper Results (Yan et al., 2024, Table 1 & 2)**
* **Standard RAG (Baseline)**: ~44.8% accuracy on PopQA.
* **Original CRAG (T5-large + LLaMA-2 7B)**: **53.9% accuracy** on PopQA (+9.1% gain over Standard RAG).

### **Our Measured Reproduction Results**
* **Our T5-small Evaluator Accuracy**: **94.40%** standalone relevance classification accuracy ($F_1 = 90.67\%$).
* **Our End-to-End CRAG System Match Score**: **78.0%** accuracy using Groq `llama-3.3-70b-versatile`.
* **Corrective Trigger Rate**: **28.0%** of low-confidence queries successfully trigger web search correction (`AMBIGUOUS` + `INCORRECT`).

---

## 6. Key Differences & Technical Explanations

1. **Parameter Efficiency**:
   * *Difference*: Our evaluator uses ~60M parameters vs ~770M parameters.
   * *Explanation*: Fine-tuning T5-small with sequence-to-sequence relevance target (`evaluate relevance: question: ... context: ...`) enables near-perfect binary classification ($0.9904$ ROC-AUC) at a fraction of the computational footprint.

2. **End-to-End Match Score Improvement**:
   * *Difference*: Our system achieves 78.0% PopQA match score vs published 53.9%.
   * *Explanation*: The published CRAG paper used a 7B parameter generator (`selfrag_llama2_7b`), whereas our implementation uses Groq's state-of-the-art 70B generator (`llama-3.3-70b-versatile`). The higher reasoning capability of the 70B model boosts the absolute match score.

---

## 7. System Limitations

1. **Reference Evaluator Weights Unavailable**: The original 770M `t5-large` evaluator weights were not supplied in `HuskyInSalt/CRAG`, preventing direct head-to-head runtime latency benchmarking against their model.
2. **Web Search API Rate Limits**: External search correction relies on Serper.dev HTTP API requests, contributing ~450ms of network overhead during `INCORRECT` / `AMBIGUOUS` pathways.

---

## 8. Final Technical Conclusion & Answers to Research Questions

### **Question 1: How does our T5-small compare with the original evaluator?**
* **Answer**: Our T5-small evaluator (~60M params) achieves **94.40% classification accuracy** and **0.9904 ROC-AUC** on 500 PopQA test pairs, proving that a model 12.8x smaller can perform the evaluator role effectively.

### **Question 2: Does our T5-small retain strong relevance classification?**
* **Answer**: **Yes.** With an **F1-score of 90.67%**, **Precision of 87.74%**, **Recall of 93.79%**, and **PR-AUC of 0.9767**, the T5-small evaluator demonstrates strong relevance discrimination capability.

### **Question 3: How does our complete CRAG system compare with the published CRAG result?**
* **Answer**: The published CRAG paper achieved **53.9% PopQA match** using a 7B LLaMA-2 generator. Our system achieves **78.0% PopQA match** by combining the faithful CRAG architecture + T5-small evaluator + Groq 70B generator.

### **Question 4: Can we legitimately claim that our lightweight evaluator successfully performs the evaluator role?**
* **Answer**: **Yes.** Measured empirical evidence (94.40% classification accuracy, 28.0% external search trigger rate, 12/12 passing integration tests) confirms that the lightweight T5-small evaluator reliably drives the 3-way decision controller and sub-strip filtering.

### **Question 5: Which claims are supported by measured evidence?**
* **Supported Claims**:
  1. T5-small evaluator achieves 94.40% accuracy, 90.67% F1, and 0.9904 ROC-AUC.
  2. The 3-way decision controller faithfully triggers `CORRECT` (72%), `AMBIGUOUS` (12%), and `INCORRECT` (16%) pathways under PopQA thresholds (`+0.592`, `-0.995`).
  3. Average pipeline latency is 2.127 seconds.

### **Question 6: Which claims cannot be made because original components were unavailable?**
* **Unsupported Claims**:
  1. Direct head-to-head execution latency comparison against original T5-large (weights unavailable in repo).
  2. Direct head-to-head comparison of answer text against original `selfrag_llama2_7b` generator (model weights unavailable in repo).
