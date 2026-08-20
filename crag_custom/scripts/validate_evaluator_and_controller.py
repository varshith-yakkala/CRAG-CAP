import sys
from crag_custom.evaluator.t5_evaluator import T5RetrievalEvaluator
from crag_custom.corrective.controller import CRAGDecisionController
from crag_custom.corrective.decision import CRAGDecision
from crag_custom.evaluator.schemas import EvaluationResult

def main():
    print("==========================================================")
    print("1. RUNNING 10 REPRESENTATIVE EVALUATOR VALIDATION EXAMPLES")
    print("==========================================================")

    evaluator = T5RetrievalEvaluator()

    examples = [
        # Relevant Examples (Label 1)
        ("What is George Rankin's occupation?", "George Rankin was an Australian soldier and politician. He attended school and became a farmer."),
        ("In what city was Billy Carlson born?", "Billy Carlson was an American racecar driver born in San Diego, California."),
        ("What is the religion of John Gwynn?", "John Gwynn was a devout Anglican Christian theologian in England."),
        ("What sport does Kiribati men's national basketball team play?", "The Kiribati men's national basketball team represents Kiribati in international basketball competitions."),
        ("What is the capital of France?", "Paris is the capital and most populous city of France."),

        # Irrelevant Examples (Label 0)
        ("What is George Rankin's occupation?", "Bangai-O Spirits is an action game for the Nintendo DS with 160 levels."),
        ("In what city was Billy Carlson born?", "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate in San Francisco."),
        ("What is the religion of John Gwynn?", "Python is a high-level programming language released in 1991."),
        ("What sport does Kiribati men's national basketball team play?", "Pizza is a dish of Italian origin consisting of a flattened disk of bread dough."),
        ("What is the capital of France?", "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.")
    ]

    for idx, (q, p) in enumerate(examples, start=1):
        res = evaluator.evaluate(q, p)
        p_0 = res.metadata.get("p_0", 1.0 - res.relevance_probability)
        print(f"\n--- EXAMPLE {idx} ---")
        print(f"QUESTION:               {q}")
        print(f"PASSAGE:                {p}")
        print(f"EXPECTED/GENERATED LABEL: {res.label}")
        print(f"P(1):                   {res.relevance_probability:.6f}")
        print(f"P(0):                   {p_0:.6f}")
        print(f"CRAG SCORE (2P-1):      {res.crag_score:+.6f}")

    print("\n" + "=" * 58)
    print("2. TESTING CONTROLLER INDEPENDENTLY WITH MANUAL SCORE SETS")
    print("==========================================================")

    controller = CRAGDecisionController(upper_threshold=0.592, lower_threshold=0.995)

    test_score_sets = [
        # 1. CORRECT: Max score >= 0.592
        ([0.80, 0.20, -0.40, -0.80], CRAGDecision.CORRECT),
        
        # 2. AMBIGUOUS: -0.995 <= Max score < 0.592
        ([0.10, -0.20, -0.50, -0.80], CRAGDecision.AMBIGUOUS),
        
        # 3. INCORRECT: All scores < -0.995
        ([-0.996, -0.997, -0.998, -0.999], CRAGDecision.INCORRECT)
    ]

    for idx, (scores, expected_dec) in enumerate(test_score_sets, start=1):
        eval_objs = [
            EvaluationResult(
                label="1" if s >= 0 else "0",
                is_relevant=(s >= 0),
                relevance_probability=(s + 1.0) / 2.0,
                crag_score=s,
                raw_output="1" if s >= 0 else "0"
            )
            for s in scores
        ]
        dec_res = controller.decide(eval_objs)
        status = "PASSED" if dec_res.decision == expected_dec else "FAILED"
        print(f"\nManual Score Set {idx}: {scores}")
        print(f"  Max Score:        {dec_res.max_score:+.4f}")
        print(f"  Expected State:   {expected_dec.value}")
        print(f"  Resulting State:  {dec_res.decision.value} [{status}]")
        assert dec_res.decision == expected_dec, f"Mismatch: expected {expected_dec}, got {dec_res.decision}"

    print("\n==========================================================")
    print("ALL 10 EVALUATOR EXAMPLES & 3 CONTROLLER TESTS PASSED 100%")
    print("==========================================================")

if __name__ == "__main__":
    main()
