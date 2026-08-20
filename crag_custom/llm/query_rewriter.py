from typing import Optional
from crag_custom.llm.groq_client import GroqClient
from crag_custom.config.settings import settings

class GroqQuestionRewriter:
    """
    Groq Question Rewriter for CRAG INCORRECT and AMBIGUOUS pathways.
    Converts question into search engine keywords using main Groq model.
    """
    def __init__(self, client: Optional[GroqClient] = None, model: Optional[str] = None):
        self.client = client or GroqClient()
        self.model = model or settings.GROQ_MODEL

    def rewrite(self, query: str, task: str = "popqa") -> str:
        prompt = (
            "Extract search keywords for a web search engine from the following question.\n"
            "Respond ONLY with the search keywords. Do NOT include conversational text.\n\n"
            f"Question: {query}\n"
            "Keywords:"
        )
        messages = [
            {"role": "system", "content": "You are a concise search query optimization system."},
            {"role": "user", "content": prompt}
        ]
        
        rewritten = self.client.generate(
            messages=messages,
            model=self.model,
            temperature=0.1,
            max_tokens=50
        )
        if not rewritten:
            return query
        return rewritten
