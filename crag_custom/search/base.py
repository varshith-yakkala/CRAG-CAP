from abc import ABC, abstractmethod
from typing import List

class BaseSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Execute web search for rewritten query. Returns web result snippets.
        """
        pass
