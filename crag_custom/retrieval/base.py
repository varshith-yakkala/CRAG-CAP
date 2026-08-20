from abc import ABC, abstractmethod
from typing import List
from crag_custom.retrieval.schemas import RetrievedDocument

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        """
        Retrieve top_k documents for a given query.
        """
        pass
