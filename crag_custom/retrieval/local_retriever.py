from typing import List, Optional
from crag_custom.retrieval.base import BaseRetriever
from crag_custom.retrieval.schemas import RetrievedDocument

class LocalRetriever(BaseRetriever):
    """
    Local retriever implementation that searches over a corpus of documents or pre-loaded benchmark passages.
    """
    def __init__(self, corpus: Optional[List[RetrievedDocument]] = None):
        self.corpus = corpus or []

    def add_documents(self, documents: List[RetrievedDocument]):
        self.corpus.extend(documents)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        if not self.corpus:
            return []
        
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.corpus:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words.intersection(doc_words))
            score = float(overlap)
            scored_docs.append((score, doc))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
