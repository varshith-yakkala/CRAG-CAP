from typing import Optional
from crag_custom.llm.groq_client import GroqClient
from crag_custom.config.settings import settings

class GroqGenerator:
    """
    Groq Answer Generator for CRAG.
    Generates grounded answers based on refined knowledge context.
    """
    def __init__(self, client: Optional[GroqClient] = None, model: Optional[str] = None):
        self.client = client or GroqClient()
        self.model = model or settings.GROQ_MODEL

    def generate(self, query: str, context: Optional[str] = None) -> str:
        if context:
            prompt = (
                "Refer to the following documents, follow the instruction and answer the question.\n\n"
                f"Documents: {context}\n\n"
                f"Instruction: Answer the question concisely and accurately based on the documents.\n"
                f"Question: {query}"
            )
        else:
            prompt = (
                "Answer the following question accurately and concisely.\n"
                f"Question: {query}"
            )
            
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, factual AI assistant. Answer using the supplied knowledge context. "
                    "Do not hallucinate unsupported facts. Prioritize relevant evidence."
                )
            },
            {"role": "user", "content": prompt}
        ]
        
        return self.client.generate(
            messages=messages,
            model=self.model,
            temperature=0.0,
            max_tokens=250
        )
