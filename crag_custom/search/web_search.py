import os
import json
import requests
from typing import List, Optional
from crag_custom.search.base import BaseSearchProvider
from crag_custom.config.settings import settings

class SerperSearchProvider(BaseSearchProvider):
    """
    Search provider using Serper.dev Google Search API (exact search engine from reference paper).
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SEARCH_API_KEY
        self.url = "https://google.serper.dev/search"

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if not self.api_key:
            raise ValueError("[SerperSearchProvider] ERROR: SEARCH_API_KEY is not set. Real mode requires a valid API key.")

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        payload = json.dumps({"q": query})

        response = requests.post(self.url, headers=headers, data=payload, timeout=10)
        if response.status_code != 200:
            raise RuntimeError(f"[SerperSearchProvider] HTTP {response.status_code}: {response.text}")
            
        result = response.json()
        snippets = []
        if "organic" in result:
            for item in result["organic"][:top_k]:
                title = item.get("title", "")
                snippet = item.get("snippet", title)
                snippets.append(f"{title}: {snippet}")
        return snippets if snippets else [query]

class MockSearchProvider(BaseSearchProvider):
    """
    Mock search provider for offline testing / debug mode.
    """
    def search(self, query: str, top_k: int = 5) -> List[str]:
        return [
            f"External Web Result 1 for '{query}': Detailed web search evidence and factual information.",
            f"External Web Result 2 for '{query}': Additional context gathered from public web pages."
        ][:top_k]

def get_search_provider() -> BaseSearchProvider:
    if settings.DEBUG_MODE or not settings.SEARCH_API_KEY:
        return MockSearchProvider()
    return SerperSearchProvider()
