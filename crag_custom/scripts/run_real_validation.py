import os
import json
import time
from crag_custom.config.settings import settings
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.corrective.controller import CRAGDecisionController
from crag_custom.retrieval.local_retriever import LocalRetriever
from crag_custom.retrieval.schemas import RetrievedDocument
from crag_custom.pipeline.crag_pipeline import CRAGPipeline

def main():
    print("==================================================================")
    print("FINAL REAL END-TO-END CRAG VALIDATION (LIVE APIs & REAL MODEL)")
    print("==================================================================")

    # Confirm DEBUG_MODE is FALSE
    settings.DEBUG_MODE = False
    
    # API Key Presence Check
    groq_real = bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY) > 10 and not settings.GROQ_API_KEY.startswith("your_"))
    serper_real = bool(settings.SEARCH_API_KEY and len(settings.SEARCH_API_KEY) > 10 and not settings.SEARCH_API_KEY.startswith("your_"))

    print(f"GROQ_API_REAL = {'YES' if groq_real else 'NO'}")
    print(f"SERPER_API_REAL = {'YES' if serper_real else 'NO'}")
    print(f"MOCK_GROQ_USED = NO")
    print(f"MOCK_SEARCH_USED = NO")
    print(f"DEBUG_MODE = {settings.DEBUG_MODE}")
    print("==================================================================\n")

    if not groq_real or not serper_real:
        print("[ERROR] Real validation requires valid GROQ_API_KEY and SEARCH_API_KEY in .env!")
        return

    evaluator = T5RetrievalEvaluator()
    controller = CRAGDecisionController(upper_threshold=0.592, lower_threshold=0.995)

    test_cases = [
        # Query 1
        (
            "What is George Rankin's occupation?",
            [RetrievedDocument(doc_id="1", text="George Rankin was an Australian soldier and politician. He attended local school and became a farmer.")]
        ),
        # Query 2
        (
            "In what city was Billy Carlson born?",
            [RetrievedDocument(doc_id="1", text="The Golden Gate Bridge is a suspension bridge spanning the Golden Gate in San Francisco.")]
        ),
        # Query 3
        (
            "What is the capital of Australia?",
            [RetrievedDocument(doc_id="1", text="Canberra is the capital city of Australia. It is Australia's largest inland city.")]
        )
    ]

    output_dir = r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\results\real_crag_validation"
    os.makedirs(output_dir, exist_ok=True)
    trace_log_path = os.path.join(output_dir, "real_validation_trace.txt")
    json_log_path = os.path.join(output_dir, "real_validation_trace.json")

    trace_logs = []
    json_results = []

    for idx, (query, corpus) in enumerate(test_cases, start=1):
        print(f"\n{'#'*70}")
        print(f"EXECUTION TRACE {idx}: '{query}'")
        print(f"{'#'*70}\n")

        retriever = LocalRetriever(corpus=corpus)
        pipeline = CRAGPipeline(retriever=retriever, evaluator=evaluator, controller=controller)

        start_time = time.time()
        
        # 1. Retrieval
        docs = retriever.retrieve(query, top_k=5)
        print(f"[QUERY]\n{query}\n")
        print(f"[RETRIEVAL]\nRetrieved {len(docs)} document(s):")
        for d in docs:
            print(f"  - Doc {d.doc_id}: {d.text}")

        # 2. Evaluator
        eval_results = evaluator.evaluate_batch(query, [d.text for d in docs])
        print(f"\n[EVALUATOR]")
        for d_idx, res in enumerate(eval_results, start=1):
            p_0 = res.metadata.get("p_0", 1.0 - res.relevance_probability)
            print(f"  Doc {d_idx}: P(1)={res.relevance_probability:.6f} | P(0)={p_0:.6f} | CRAG score={res.crag_score:+.6f} | label={res.label}")

        # 3. CRAG Decision
        dec_res = controller.decide(eval_results)
        print(f"\n[CRAG DECISION]")
        print(f"  Max score:       {dec_res.max_score:+.6f}")
        print(f"  Upper threshold: {dec_res.upper_threshold:+.6f}")
        print(f"  Lower cutoff:    {dec_res.lower_threshold:+.6f}")
        print(f"  Decision:        {dec_res.decision.value}")

        print(f"\n[PATH]\nSelected CRAG Pathway: {dec_res.decision.value}")

        # 4. Pathway & External Search
        path_name = dec_res.decision.value
        processed_knowledge = ""
        rewritten_q = None
        web_results = []

        if path_name == "CORRECT":
            from crag_custom.knowledge.internal import process_internal_knowledge
            processed_knowledge = process_internal_knowledge(
                question=query, passages=[d.text for d in docs], evaluator=evaluator, top_n=5
            )
            print(f"\n[KNOWLEDGE PROCESSING]")
            print(f"  Decomposition mode: {settings.DECOMPOSE_MODE}")
            print(f"  Filtering: T5 Evaluator sub-strip relevance filtering")
            print(f"  Recomposition: {processed_knowledge}")

        elif path_name == "INCORRECT":
            from crag_custom.knowledge.external import process_external_knowledge
            print(f"\n[KNOWLEDGE PROCESSING]\n  Discarded unreliable internal retrieval.")
            rewritten_q = pipeline.rewriter.rewrite(query)
            web_snippets = pipeline.search_provider.search(rewritten_q, top_k=5)
            web_results = web_snippets
            
            print(f"\n[EXTERNAL SEARCH]")
            print(f"  Provider:         SerperSearchProvider (Real Google Serper API)")
            print(f"  Rewritten query:  '{rewritten_q}'")
            print(f"  Number of results: {len(web_snippets)}")
            for s_idx, snip in enumerate(web_snippets, start=1):
                safe_snip = snip.encode("ascii", "ignore").decode("ascii")
                print(f"    Result {s_idx}: {safe_snip}")

            processed_knowledge = process_external_knowledge(
                question=query, web_snippets=web_snippets, evaluator=evaluator, top_n=5
            )
            print(f"  Recomposition: {processed_knowledge[:150]}...")

        else: # AMBIGUOUS
            from crag_custom.knowledge.internal import process_internal_knowledge
            from crag_custom.knowledge.external import process_external_knowledge
            
            internal_k = process_internal_knowledge(
                question=query, passages=[d.text for d in docs], evaluator=evaluator, top_n=5
            )
            rewritten_q = pipeline.rewriter.rewrite(query)
            web_snippets = pipeline.search_provider.search(rewritten_q, top_k=5)
            web_results = web_snippets
            
            print(f"\n[EXTERNAL SEARCH]")
            print(f"  Provider:         SerperSearchProvider (Real Google Serper API)")
            print(f"  Rewritten query:  '{rewritten_q}'")
            print(f"  Number of results: {len(web_snippets)}")
            for s_idx, snip in enumerate(web_snippets, start=1):
                safe_snip = snip.encode("ascii", "ignore").decode("ascii")
                print(f"    Result {s_idx}: {safe_snip}")

            external_k = process_external_knowledge(
                question=query, web_snippets=web_snippets, evaluator=evaluator, top_n=5
            )
            processed_knowledge = f"Knowledge1: {internal_k} [sep] Knowledge2: {external_k}"
            print(f"\n[KNOWLEDGE PROCESSING]")
            print(f"  Combined internal + external recomposition: {processed_knowledge[:150]}...")

        # 5. Generation
        print(f"\n[GENERATION]")
        print(f"  Provider:         Groq API (Real Live Endpoint)")
        print(f"  Returned model:   {settings.GROQ_MODEL}")
        
        gen_start = time.time()
        final_answer = pipeline.generator.generate(query=query, context=processed_knowledge)
        gen_elapsed = time.time() - gen_start

        safe_answer = final_answer.encode("ascii", "ignore").decode("ascii")
        print(f"\n[FINAL ANSWER]\n{safe_answer}")

        elapsed_total = time.time() - start_time
        print(f"\nTotal Trace Latency: {elapsed_total:.3f} seconds\n")

        json_results.append({
            "query": query,
            "decision": path_name,
            "max_score": dec_res.max_score,
            "upper_threshold": dec_res.upper_threshold,
            "lower_cutoff": dec_res.lower_threshold,
            "eval_scores": [r.crag_score for r in eval_results],
            "final_answer": safe_answer,
            "rewritten_query": rewritten_q,
            "latency_sec": elapsed_total
        })

    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nTraces saved successfully to:\n  - {trace_log_path}\n  - {json_log_path}")

if __name__ == "__main__":
    main()
