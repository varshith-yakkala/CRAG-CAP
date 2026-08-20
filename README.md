# Custom Corrective Retrieval-Augmented Generation (CRAG-CAP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end reimplementation and evaluation of **Corrective Retrieval-Augmented Generation (CRAG)** (*Yan et al., 2024*), replacing the original 770M T5-large cross-encoder with a fine-tuned **60.5M parameter T5-small seq2seq evaluator** and scaling generation with **Groq Cloud API (`openai/gpt-oss-120b`)** and real-time **Google Serper API**.

---

## 📊 Key Evaluation Performance Metrics

Audited against **1,240 queries** from the official PopQA test dataset:

* **PopQA Answer Match Accuracy**: **57.98%** ($\mathbf{+4.08\%}$ over published 2024 CRAG paper's 53.90%).
* **Evaluator Parameter Compression**: **12.8x parameter reduction** (60.5M vs 770M T5-large).
* **Evaluator Inference Speed**: **~209 ms – 239 ms per passage** on PyTorch CPU.
* **Dataset Processing Coverage**: **89.53%** (1,240 / 1,385 queries).
* **External Search Trigger Rate**: **58.23%** (722 / 1,240 queries).

### Pathway Breakdown & Pathway Accuracy

| Decision Pathway | Condition / Rule | Queries | % of Processed | Pathway Answer Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **`CORRECT`** | $\text{MaxScore} \ge +0.5920$ | 518 | 41.77% | **70.08%** |
| **`AMBIGUOUS`** | $-0.9950 \le \text{MaxScore} < +0.5920$ | 582 | 46.94% | **47.25%** |
| **`INCORRECT`** | $\text{MaxScore} < -0.9950$ | 140 | 11.29% | **57.86%** |
| **TOTAL SYSTEM** | — | **1,240** | **100.00%** | **57.98%** |

---

## 🏗️ Architectural Modifications (Original CRAG vs. Custom CRAG)

| Component | Original CRAG Paper *(Yan et al., 2024)* | Our Custom CRAG Implementation | Impact |
| :--- | :---: | :---: | :--- |
| **Evaluator Model** | `t5-large` (770M params) | **`t5-small` (60.5M params)** | **12.8x Parameter Reduction** |
| **Generative LLM** | Self-RAG LLaMA-2 7B | **Groq API (`openai/gpt-oss-120b`)** | **+4.08% Accuracy Boost** |
| **Search Engine** | Bing / DuckDuckGo (2023) | **Google Serper API (Live 2026)** | **Cleaner Factual Search Snippets** |
| **3-Way Controller** | $\gamma_1 = +0.5920, \gamma_2 = -0.9950$ | $\gamma_1 = +0.5920, \gamma_2 = -0.9950$ | **100% Preserved** |
| **Knowledge Processing** | Decompose $\rightarrow$ Filter $\rightarrow$ Recompose | Decompose $\rightarrow$ Filter $\rightarrow$ Recompose | **100% Preserved** |

---

## 📁 Repository Structure

```text
CRAG-CAP/
├── crag_custom/                          # Complete Custom CRAG Implementation
│   ├── config/                           # Configuration & threshold settings
│   ├── corrective/                       # 3-Way decision controller (process_flag)
│   ├── docs/                             # Architecture & audit docs
│   ├── evaluator/                        # Fine-tuned T5-small seq2seq evaluator & adapter
│   ├── knowledge/                        # Passage decomposition, sub-strip filtering & recomposition
│   ├── llm/                              # Groq LLM generator & query rewriter
│   ├── pipeline/                         # End-to-end CRAG Pipeline orchestrator
│   ├── retrieval/                        # Local retriever & document schema
│   ├── scripts/                          # Execution scripts & benchmark runners
│   └── search/                           # Google Serper web search provider
├── results/                              # Research Benchmark Deliverables
│   ├── final_benchmark/                  # 100-Query benchmark & architectural comparison
│   └── full_benchmark/                   # Full-dataset PopQA benchmark deliverables
│       ├── full_popqa_predictions.jsonl  # 1,240 live prediction records
│       ├── full_popqa_metrics_corrected.json
│       ├── METRICS_AUDIT.md              # Metrics accounting audit report
│       ├── POPQA_ANSWER_ACCURACY_AUDIT.md# Audited 57.98% answer accuracy report
│       ├── FINAL_PROVENANCE_AUDIT.md     # Provenance & API authenticity verification
│       └── README_EVALUATION_METRICS.md  # Detailed evaluation metrics documentation
├── CRAG_repo/                            # Official Reference CRAG Implementation & Data
├── t5_evaluator_model/                   # Evaluator model configuration & tokenizer files
├── crag_model_training.ipynb             # Google Colab fine-tuning notebook for T5-small
└── crag_custom/.env.example              # Clean environment template (without API keys)
```

---

## 🚀 Quick Start & Installation

```bash
# Clone the repository
git clone https://github.com/varshith-yakkala/CRAG-CAP.git
cd CRAG-CAP

# Install dependencies
pip install -r crag_custom/requirements.txt

# Set environment variables in crag_custom/.env
cp crag_custom/.env.example crag_custom/.env
```

### Run Pipeline Demo
```bash
python -m crag_custom.scripts.demo_all_paths
```

### Run Benchmark Audit Verification
```bash
python -m crag_custom.scripts.run_full_popqa_benchmark
```

---

## 📖 Citation & References

* **Yan et al., 2024**: *Corrective Retrieval Augmented Generation*. arXiv:2401.15884.
* **Reference Repository**: [`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG)
