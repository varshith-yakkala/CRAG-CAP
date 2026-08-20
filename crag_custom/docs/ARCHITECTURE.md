# Technical Architecture Document: Custom CRAG System

## 1. Executive Summary & Research Statement

This system implements a faithful reproduction of the **Corrective Retrieval-Augmented Generation (CRAG)** architecture from the paper *[Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884.pdf)* and reference repository [`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG).

> **Scientific Definition**:
> "We retain the original CRAG architecture, corrective pathways, knowledge refinement, and three-way decision mechanism while replacing the original T5-large retrieval evaluator (~770M params) with a lightweight T5-small sequence-to-sequence relevance evaluator (~60M params), and replacing generative model dependencies with Groq API."

---

## 2. System Architecture & Control Flow

```mermaid
flowchart TD
    UserQuery["User Question"] --> Retriever["Retriever"]
    Retriever --> Passages["Retrieved Passages [d1, d2, ..., dn]"]
    
    Passages --> T5Eval["Our T5-Small Evaluator"]
    T5Eval -->|Individual Passage Evaluation| EvalScores["Per-Doc Scores: P(1), P(0) & CRAG_score = 2P(1) - 1"]
    
    EvalScores --> Controller["CRAG Decision Controller"]
    
    Controller -->|max(scores) >= upper_threshold| CorrectPath["CORRECT Path"]
    Controller -->|lower_cutoff <= max(scores) < upper_threshold| AmbiguousPath["AMBIGUOUS Path"]
    Controller -->|all scores < lower_cutoff| IncorrectPath["INCORRECT Path"]

    CorrectPath --> InternalDecompose["Passage Decomposition (fixed_num / excerption / selection)"]
    InternalDecompose --> InternalFilter["Sub-strip Relevance Filtering (via Our T5 Evaluator)"]
    InternalFilter --> InternalRecompose["Knowledge Recomposition"]

    IncorrectPath --> Rewriter["Groq Question Rewriter"]
    Rewriter --> ExtSearch["External Search Provider (Serper / Mock)"]
    ExtSearch --> ExtProcessing["External Knowledge Scraping & Filtering"]

    AmbiguousPath --> Merge["Combine Refined Internal + External Knowledge"]

    InternalRecompose --> GroqGen["Groq Answer Generator"]
    ExtProcessing --> GroqGen
    Merge --> GroqGen

    GroqGen --> FinalAnswer["Final Answer"]
```

---

## 3. Our T5 Evaluator & Score Adaptation Methodology

### Model Specifications
* **Base Model**: `T5ForConditionalGeneration` (`t5-small`, ~60M parameters).
* **Location**: `c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final`.
* **Prompt Format**:
  ```text
  evaluate relevance: question: <QUESTION> context: <PASSAGE>
  ```
* **Target Output**: `"1"` = Relevant, `"0"` = Irrelevant.

### Logit Extraction & Score Adaptation Formula
The original CRAG evaluator (`T5ForSequenceClassification`) produced continuous regression scores. Our evaluator generates text tokens `"1"` or `"0"`. To preserve confidence information:
1. Token IDs for `"1"` and `"0"` are dynamically resolved (`id_1` and `id_0`).
2. Decoder logits for the first generated token are extracted from forward pass over `decoder_input_ids = [[decoder_start_token_id]]`:
   $$\text{logit}_1 = \text{decoder\_logit}(\text{id\_1})$$
   $$\text{logit}_0 = \text{decoder\_logit}(\text{id\_0})$$
3. Softmax probability $P(1)$ is calculated:
   $$P(1) = \frac{e^{\text{logit}_1}}{e^{\text{logit}_1} + e^{\text{logit}_0}} \in [0.0, 1.0], \quad P(0) = 1.0 - P(1)$$
4. **Our Score Adaptation**: Maps $P(1) \in [0.0, 1.0]$ into CRAG-compatible signed score space $[-1.0, +1.0]$:
   $$\text{CRAG\_score} = 2 \cdot P(1) - 1.0 \in [-1.0, +1.0]$$

---

## 4. Reference CRAG 3-Way Decision Logic & Threshold Mapping

Passages are evaluated individually: $E(q, d_i) \rightarrow \text{score}_i$.

Extracted directly from reference `CRAG_Inference.py` (`process_flag`) and `run_crag_inference.sh`:
* **Reference Thresholds**:
  * PopQA Default: `upper_threshold = 0.592`, `lower_threshold = 0.995` $\rightarrow$ `lower_cutoff = -0.995`
  * PubQA Default: `upper_threshold = 0.500`, `lower_threshold = 0.915` $\rightarrow$ `lower_cutoff = -0.915`

* **Exact Decision Rules**:
  * **`CORRECT`**: $\max_i(\text{score}_i) \ge \text{upper\_threshold}$ (e.g. $\ge +0.592$)
  * **`INCORRECT`**: $\forall i, \text{score}_i < -\text{lower\_threshold}$ (e.g. $< -0.995$)
  * **`AMBIGUOUS`**: Otherwise ($-\text{lower\_threshold} \le \max_i(\text{score}_i) < \text{upper\_threshold}$)

---

## 5. Groq Generative Integration

Groq API is used exclusively for generative components:
1. **Groq Question Rewriter**: Converts queries into search engine keywords (`llama-3.1-8b-instant`).
2. **Groq Answer Generator**: Generates grounded responses (`llama-3.3-70b-versatile`).
