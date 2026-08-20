from dataclasses import dataclass

@dataclass
class CRAGThresholds:
    """
    Exact threshold values extracted from reference CRAG repository scripts (run_crag_inference.sh).
    Scores are on the signed [-1.0, +1.0] scale.
    """
    # PopQA defaults
    POPQA_UPPER: float = 0.592
    POPQA_LOWER: float = -0.995

    # PubQA defaults
    PUBQA_UPPER: float = 0.500
    PUBQA_LOWER: float = -0.915

    # General default thresholds for CRAG decision routing
    DEFAULT_UPPER: float = 0.500
    DEFAULT_LOWER: float = -0.900

thresholds = CRAGThresholds()
