import os
import sys
from crag_custom.config.settings import settings
from crag_custom.llm.groq_client import GroqClient

def main():
    print("==================================================================")
    print("GROQ API REAL INTEGRATION VERIFICATION")
    print("==================================================================")

    # Confirm key presence without printing the key
    has_key = bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY) > 10 and not settings.GROQ_API_KEY.startswith("your_"))
    print(f"GROQ_API_KEY_PRESENT: {'YES' if has_key else 'NO'}")

    if not has_key:
        print("[ERROR] Valid GROQ_API_KEY not found in .env.")
        sys.exit(1)

    try:
        # Instantiate GroqClient with real API Key and active model
        client = GroqClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)
        client.is_debug = False

        prompt = "Explain in two sentences what Corrective RAG (CRAG) is."
        
        # Real API request execution
        raw_response = client.client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100
        )

        print("\nAPI request succeeded: YES")
        print(f"Model returned by API: {raw_response.model}")
        
        res_text = raw_response.choices[0].message.content.strip()
        safe_text = res_text.encode("ascii", "ignore").decode("ascii")
        print(f"Response text:\n{safe_text}")
        
        if hasattr(raw_response, 'usage') and raw_response.usage:
            print("\nToken usage:")
            print(f"  Prompt Tokens:     {raw_response.usage.prompt_tokens}")
            print(f"  Completion Tokens: {raw_response.usage.completion_tokens}")
            print(f"  Total Tokens:      {raw_response.usage.total_tokens}")

        print("\n==================================================================")
        print("REAL GROQ API INTEGRATION VERIFIED WITHOUT FALLBACK!")
        print("==================================================================")

    except Exception as e:
        err_msg = str(e)
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY in err_msg:
            err_msg = err_msg.replace(settings.GROQ_API_KEY, "[REDACTED_API_KEY]")
        print(f"\nAPI request succeeded: NO")
        print(f"[ERROR] Request failed: {err_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
