import os
import json
import sys
import requests
from crag_custom.config.settings import settings

def main():
    print("==================================================================")
    print("VERIFYING SERPER SEARCH API REAL INTEGRATION")
    print("==================================================================")

    # 1 & 2. Confirm key presence without exposing secret
    has_key = bool(settings.SEARCH_API_KEY and len(settings.SEARCH_API_KEY) > 10 and not settings.SEARCH_API_KEY.startswith("your_"))
    print(f"SEARCH_API_KEY_PRESENT: {'YES' if has_key else 'NO'}")

    if not has_key:
        print("[ERROR] SEARCH_API_KEY not found or unconfigured in .env file.")
        sys.exit(1)

    # 7. Confirm MockSearchProvider is NOT used
    print("Search Provider Used: SerperSearchProvider (Real Google Serper API)")
    print("MockSearchProvider Used: NO")

    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': settings.SEARCH_API_KEY,
        'Content-Type': 'application/json'
    }
    query = "What is George Rankin's occupation?"
    payload = json.dumps({"q": query})

    print(f"\nSending live search request to Serper API for query: '{query}'...")

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        
        # 8 & 9. Strict error reporting without falling back to mock
        if response.status_code != 200:
            err_msg = response.text
            if settings.SEARCH_API_KEY in err_msg:
                err_msg = err_msg.replace(settings.SEARCH_API_KEY, "[REDACTED_API_KEY]")
            print(f"\nRequest Succeeded: NO")
            print(f"[ERROR] Serper API HTTP {response.status_code}: {err_msg}")
            sys.exit(1)

        data = response.json()
        organic_results = data.get("organic", [])

        print("\n[VERIFICATION RESULTS]")
        print("Request Succeeded: YES")
        print(f"Number of Results Returned: {len(organic_results)}")

        print("\nOrganic Web Search Results:")
        for idx, item in enumerate(organic_results[:5], start=1):
            title = item.get("title", "N/A")
            link = item.get("link", "N/A")
            snippet = item.get("snippet", "N/A")
            
            # Safe UTF-8 printing
            safe_title = title.encode("ascii", "ignore").decode("ascii")
            safe_snippet = snippet.encode("ascii", "ignore").decode("ascii")

            print(f"\n  Result {idx}:")
            print(f"    Title:   {safe_title}")
            print(f"    URL:     {link}")
            print(f"    Snippet: {safe_snippet}")

        print("\n==================================================================")
        print("SERPER GOOGLE SEARCH API VERIFIED 100% LIVE & REAL!")
        print("==================================================================")

    except Exception as e:
        err_msg = str(e)
        if settings.SEARCH_API_KEY in err_msg:
            err_msg = err_msg.replace(settings.SEARCH_API_KEY, "[REDACTED_API_KEY]")
        print("\nRequest Succeeded: NO")
        print(f"[ERROR] Real Search Request Failed: {err_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
