# CRAG Reference Repository vs. Custom Implementation Comparison Table

This document presents a component-by-component comparison between the reference implementation [`HuskyInSalt/CRAG`](https://github.com/HuskyInSalt/CRAG) and our system [`crag_custom/`](file:///c:/Users/varsh/OneDrive/Desktop/2-2/capstone/crag_custom).

---

## 1. System Component Comparison Matrix

| Component | Reference CRAG (`HuskyInSalt/CRAG`) | Our Custom Implementation (`crag_custom/`) | Status | Technical Difference Details |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval System** | Pre-retrieved Top-10 documents from Contriever / DPR | In-Memory `LocalRetriever` / Custom Retriever Interface | **SAME CONCEPT / MODIFIED INTERFACE** | Reference uses static document files; our retriever is modular and accepts dynamic corpus objects or local text files. |
| **Retrieval Top-K** | Default `ndocs = 10` | Default `top_k = 5` (Configurable via `TOP_K`) | **SAME CONFIGURABLE PARAMETER** | Reference defaults to top 10; ours allows top-10 or top-5 via CLI/settings. |
| **Evaluator Model** | `T5ForSequenceClassification` (`t5-large`, ~770M params) | `T5ForConditionalGeneration` (`t5-small`, ~60M params) | **INTENTIONAL SUBSTITUTION** | Substituted fine-tuned sequence classification model with lightweight sequence-to-sequence conditional generation model (`t5_evaluator_final`). |
| **Evaluator Input Format** | `question [SEP] passage` | `evaluate relevance: question: <QUESTION> context: <PASSAGE>` | **INTENTIONAL SUBSTITUTION** | Formatted specifically to match our supplied T5 checkpoint's training prompt template. |
| **Evaluator Output** | Linear regression scalar logit | Text Token `"1"` / `"0"` + Decoder Softmax Probability $P(1)$ | **INTENTIONAL SUBSTITUTION** | Output string decoded as `"1"`/`"0"`; $P(1)$ extracted from decoder logits of token `'1'` vs `'0'`. |
| **Score Adaptation** | Regression logit directly used | $\text{CRAG\_score} = 2 \cdot P(1) - 1.0 \in [-1.0, +1.0]$ | **INTENTIONAL ADAPTATION** | Maps continuous probability $P(1) \in [0, 1]$ into signed $[-1, +1]$ score space expected by reference threshold bounds. |
| **Decision Mechanism** | `process_flag()` score max thresholding | `CRAGDecisionController` max score thresholding | **EXACT SAME LOGIC** | Evaluates passages individually; checks $\max(\text{scores}) \ge \text{upper\_threshold} \implies \text{CORRECT}$, $\max(\text{scores}) < -\text{lower\_threshold} \implies \text{INCORRECT}$, else $\text{AMBIGUOUS}$. |
| **Decision Thresholds** | PopQA: `upper=0.592`, `lower=0.995` (`cutoff=-0.995`) | PopQA: `upper=0.592`, `lower=0.995` (`cutoff=-0.995`) | **EXACT SAME THRESHOLDS** | Identical threshold parameters loaded from reference configuration. |
| **Correct Pathway** | Internal Knowledge Refinement (Decomposition $\rightarrow$ T5 Filter $\rightarrow$ Recomposition) | Internal Knowledge Refinement (Decomposition $\rightarrow$ T5 Filter $\rightarrow$ Recomposition) | **EXACT SAME CONTROL FLOW** | Internal passages split into sub-strips, evaluated individually via T5 Evaluator, and recomposed into refined context. |
| **Ambiguous Pathway** | Combined Internal + External Knowledge Processing | Combined Internal + External Knowledge Processing | **EXACT SAME CONTROL FLOW** | Internal refined strips combined with external search results using `[sep]` delimiters. |
| **Incorrect Pathway** | Reject Internal $\rightarrow$ Rewrite Query $\rightarrow$ Search Web $\rightarrow$ Ext Filter | Reject Internal $\rightarrow$ Rewrite Query $\rightarrow$ Search Web $\rightarrow$ Ext Filter | **EXACT SAME CONTROL FLOW** | Internal context completely discarded; external search evidence scraped, filtered, and passed to generator. |
| **Passage Decomposition** | Split into sub-strips (`fixed_num`, `excerption`, `selection`) | Split into sub-strips (`fixed_num`, `excerption`, `selection`) | **EXACT SAME LOGIC** | Identical passage decomposition algorithms (`fixed_num` chunking, `excerption` sentence splitting). |
| **Sub-strip Filtering** | Cross-encoder T5 relevance filtering on sub-strips | Our T5 Evaluator relevance filtering on sub-strips | **SAME PIPELINE / SUBSTITUTED MODEL** | Sub-strips evaluated using the same T5 Evaluator model instance. |
| **Knowledge Recomposition** | Concatenation of relevant sub-strips | Concatenation of relevant sub-strips | **EXACT SAME LOGIC** | Relevant sub-strips stitched together to form refined prompt context. |
| **Query Rewriter** | OpenAI `gpt-3.5-turbo-16k` | Groq API (`llama-3.1-8b-instant`) | **INTENTIONAL SUBSTITUTION** | OpenAI query rewriter replaced with fast Groq API keyword extractor. |
| **External Search Provider** | Google Serper API (`https://google.serper.dev/search`) | `SerperSearchProvider` + `MockSearchProvider` | **SAME PROVIDER + MOCK FALLBACK** | Primary search uses Serper API; offline mock provider available for `DEBUG_MODE`. |
| **Generative LLM** | Self-RAG `selfrag_llama2_7b` / LLaMA-2 via vLLM | Groq API (`llama-3.3-70b-versatile`) | **INTENTIONAL SUBSTITUTION** | Self-hosted LLaMA-2 local generator replaced with Groq API endpoint. |

---

## 2. Summary of Intentional Modifications

1. **Retrieval Evaluator Substitution**:
   * Reference: `T5ForSequenceClassification` (`t5-large`, ~770M params)
   * Ours: `T5ForConditionalGeneration` (`t5-small`, ~60M params) loaded from `t5_evaluator_final`.

2. **Generative LLM & Query Rewriter Substitution**:
   * Reference: Local vLLM `selfrag_llama2_7b` / OpenAI `gpt-3.5-turbo`
   * Ours: Groq API (`llama-3.3-70b-versatile` for answer generation, `llama-3.1-8b-instant` for query rewriting).

---

## 3. Unintentional Architectural Differences

* **None Identified**. All pipeline control flow logic, decision boundaries, 3-way branching, sub-strip filtering, and recomposition mechanisms are faithfully preserved.
