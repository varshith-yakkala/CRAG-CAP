from typing import List

def recompose_knowledge(strips: List[str]) -> str:
    """
    Recomposes knowledge strips into structured context string.
    """
    if not strips:
        return ""
    return " ; ".join(strips)
