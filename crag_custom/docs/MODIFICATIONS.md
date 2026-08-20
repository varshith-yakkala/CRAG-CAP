# System Modification Matrix: Reference CRAG vs. Our Implementation

> **Important Research Clarification**:
> The architectural contribution is **not** a redesigned CRAG controller. The modification is the replacement of the original retrieval evaluator with a lightweight sequence-to-sequence T5 evaluator and replacing generative model dependencies with Groq API while retaining the full CRAG corrective decision mechanism.

| Component | Original CRAG Reference | Our Implementation | Reason / Rationale |
| :--- | :--- | :--- | :--- |
| **Evaluator Model** | `T5ForSequenceClassification` (`t5-large`) | `T5ForConditionalGeneration` (`t5-small`) | Lightweight, fast inference on consumer GPUs/CPUs (~60M vs 770M params). |
| **Evaluator Parameters** | ~770M parameters | ~60M parameters | Reduced memory footprint & faster evaluation latency. |
| **Evaluator Input Format** | `"Question [SEP] Passage"` | `"evaluate relevance: question: ... context: ..."` | Matches fine-tuning prompt template of our supplied checkpoint. |
| **Evaluator Output** | Linear regression scalar logit | Text Token `"1"`/`"0"` + Softmax Probability $P(1)$ | Logit extraction over target tokens `1` & `0` enables probability computation. |
| **Score Conversion** | Original scalar classification score | Our Score Adaptation: $\text{CRAG\_score} = 2P(1) - 1$ | Maps $P(1) \in [0, 1]$ directly to paper's signed $[-1, +1]$ score range. |
| **Decision Mechanism** | CRAG 3-Way Threshold Controller | **UNCHANGED** | Preserves scientific 3-way decision boundaries (`CORRECT`, `AMBIGUOUS`, `INCORRECT`). |
| **Lower Cutoff Logic** | `score < -lower_threshold` (e.g. `< -0.995`) | **UNCHANGED** | Faithfully reproduces reference `process_flag` cutoff condition. |
| **Correct Path** | Internal Refinement (Decompose $\rightarrow$ Filter $\rightarrow$ Recompose) | **UNCHANGED** | Preserves knowledge refinement pipeline. |
| **Ambiguous Path** | Internal + External Combination | **UNCHANGED** | Combines internal & external knowledge streams. |
| **Incorrect Path** | Query Rewrite $\rightarrow$ Web Search $\rightarrow$ External Processing | **UNCHANGED** | Triggers web search correction mechanism. |
| **Knowledge Filtering** | Evaluator Cross-Encoder | **UNCHANGED** | Sub-strips filtered using our T5 Evaluator. |
| **Generator LLM** | Self-RAG `selfrag_llama2_7b` / LLaMA-2 | **Groq API** (`llama-3.3-70b-versatile`) | Ultra-fast token generation (~500 t/s) and state-of-the-art 70B reasoning. |
| **Question Rewriter** | OpenAI `gpt-3.5-turbo-16k` | **Groq API** (`llama-3.1-8b-instant`) | Free, low-latency search keyword extraction. |
