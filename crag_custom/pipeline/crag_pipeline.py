import time
from typing import List, Optional
from crag_custom.retrieval.base import BaseRetriever
from crag_custom.evaluator.base import BaseRetrievalEvaluator
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.corrective.controller import CRAGDecisionController
from crag_custom.corrective.decision import CRAGDecision
from crag_custom.knowledge.internal import process_internal_knowledge
from crag_custom.knowledge.external import process_external_knowledge
from crag_custom.search.base import BaseSearchProvider
from crag_custom.search.web_search import get_search_provider
from crag_custom.llm.query_rewriter import GroqQuestionRewriter
from crag_custom.llm.generator import GroqGenerator
from crag_custom.pipeline.pipeline_result import PipelineResult
from crag_custom.config.settings import settings
from crag_custom.config.thresholds import thresholds

class CRAGPipeline:
    """
    Faithful implementation of the Corrective Retrieval-Augmented Generation (CRAG) Pipeline.
    Architecture:
    Retriever -> Our T5 Evaluator -> CRAG 3-Way Controller -> Corrective Pathway -> Groq Generator
    """
    def __init__(
        self,
        retriever: BaseRetriever,
        evaluator: Optional[BaseRetrievalEvaluator] = None,
        controller: Optional[CRAGDecisionController] = None,
        search_provider: Optional[BaseSearchProvider] = None,
        rewriter: Optional[GroqQuestionRewriter] = None,
        generator: Optional[GroqGenerator] = None,
        decompose_mode: Optional[str] = None
    ):
        self.retriever = retriever
        self.evaluator = evaluator or T5RetrievalEvaluator()
        self.controller = controller or CRAGDecisionController(
            upper_threshold=thresholds.DEFAULT_UPPER,
            lower_threshold=thresholds.DEFAULT_LOWER
        )
        self.search_provider = search_provider or get_search_provider()
        self.rewriter = rewriter or GroqQuestionRewriter()
        self.generator = generator or GroqGenerator()
        self.decompose_mode = decompose_mode or settings.DECOMPOSE_MODE

    def run(self, query: str, top_k: int = 5, verbose: bool = True) -> PipelineResult:
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 80)
            print(f"[QUERY] {query}")
            print("=" * 80)
            
        # 1. Retrieval
        docs = self.retriever.retrieve(query, top_k=top_k)
        if verbose:
            print(f"[RETRIEVAL] Retrieved {len(docs)} documents.")
            for idx, d in enumerate(docs):
                print(f"  Doc {idx+1}: {d.text[:80]}...")
                
        # 2. Evaluation via Our T5 Evaluator (each passage evaluated individually)
        passages = [d.text for d in docs]
        eval_results = self.evaluator.evaluate_batch(query, passages)
        
        if verbose:
            print("\n[EVALUATOR] Individual Passage Evaluations:")
            for idx, res in enumerate(eval_results):
                print(
                    f"  Doc {idx+1}: P(relevant)={res.relevance_probability:.4f} | "
                    f"CRAG score={res.crag_score:+.4f} | Label={res.label} | Output='{res.raw_output}'"
                )
                
        # 3. CRAG 3-Way Decision Controller
        decision_res = self.controller.decide(eval_results)
        decision = decision_res.decision
        
        if verbose:
            print(f"\n[CRAG DECISION] Score Max={decision_res.max_score:+.4f} => Decision: {decision.value}")
            
        # 4. Corrective Pathway Execution
        processed_knowledge = ""
        rewritten_query = None
        web_snippets = []
        pathway_name = ""

        if decision == CRAGDecision.CORRECT:
            pathway_name = "INTERNAL_KNOWLEDGE"
            if verbose:
                print("[ACTION] Executing CORRECT Path: Refining internal retrieved knowledge.")
            processed_knowledge = process_internal_knowledge(
                question=query,
                passages=passages,
                evaluator=self.evaluator,
                decompose_mode=self.decompose_mode,
                top_n=top_k
            )

        elif decision == CRAGDecision.INCORRECT:
            pathway_name = "EXTERNAL_SEARCH"
            if verbose:
                print("[ACTION] Executing INCORRECT Path: Discarding internal knowledge -> Question Rewriting -> Web Search.")
            rewritten_query = self.rewriter.rewrite(query)
            if verbose:
                print(f"  Rewritten Search Query: '{rewritten_query}'")
            web_snippets = self.search_provider.search(rewritten_query, top_k=top_k)
            if verbose:
                print(f"  Fetched {len(web_snippets)} external web snippets.")
            processed_knowledge = process_external_knowledge(
                question=query,
                web_snippets=web_snippets,
                evaluator=self.evaluator,
                top_n=top_k
            )

        else: # AMBIGUOUS
            pathway_name = "COMBINED_KNOWLEDGE"
            if verbose:
                print("[ACTION] Executing AMBIGUOUS Path: Combining Internal + External Web Knowledge.")
            internal_k = process_internal_knowledge(
                question=query,
                passages=passages,
                evaluator=self.evaluator,
                decompose_mode=self.decompose_mode,
                top_n=top_k
            )
            rewritten_query = self.rewriter.rewrite(query)
            web_snippets = self.search_provider.search(rewritten_query, top_k=top_k)
            external_k = process_external_knowledge(
                question=query,
                web_snippets=web_snippets,
                evaluator=self.evaluator,
                top_n=top_k
            )
            processed_knowledge = f"Knowledge1: {internal_k} [sep] Knowledge2: {external_k}"

        if verbose:
            print(f"\n[PROCESSED KNOWLEDGE] Length: {len(processed_knowledge)} chars")
            print(f"  Content snippet: '{processed_knowledge[:120]}...'")

        # 5. Final Answer Generation via Groq
        if verbose:
            print(f"\n[GENERATION] Querying Groq Generator ({settings.GROQ_MODEL})...")
        final_answer = self.generator.generate(query=query, context=processed_knowledge)

        if verbose:
            print(f"\n[FINAL ANSWER]\n{final_answer}")
            print("=" * 80 + "\n")

        latency = time.time() - start_time
        return PipelineResult(
            query=query,
            retrieved_documents=docs,
            eval_results=eval_results,
            decision=decision,
            confidence_score=decision_res.max_score,
            selected_pathway=pathway_name,
            processed_knowledge=processed_knowledge,
            rewritten_query=rewritten_query,
            web_snippets=web_snippets,
            final_answer=final_answer,
            latency_seconds=latency
        )
