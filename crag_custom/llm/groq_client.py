import os
import re
import time
from typing import List, Dict, Any, Optional
from crag_custom.config.settings import settings

class GroqClient:
    """
    Centralized Groq API Client abstraction with Automatic Multi-Key Rotation.
    Only used for generative components: Question Rewriting & Final Answer Generation.
    NEVER used for retrieval evaluation or CRAG decision making.
    """
    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: Optional[str] = None,
        debug_mode: Optional[bool] = None
    ):
        self.api_keys = api_keys or settings.GROQ_API_KEYS
        if not self.api_keys and settings.GROQ_API_KEY:
            self.api_keys = [settings.GROQ_API_KEY]
            
        self.key_index = 0
        self.model = model or settings.GROQ_MODEL
        self.debug_mode = debug_mode if debug_mode is not None else settings.DEBUG_MODE
        
        self.client = None
        if not self.debug_mode:
            self._init_client()

    def _init_client(self):
        from groq import Groq
        current_key = self.api_keys[self.key_index]
        if not current_key or current_key.startswith("your_"):
            raise ValueError("[GroqClient] ERROR: GROQ_API_KEY is missing or invalid in real mode.")
        self.client = Groq(api_key=current_key)

    def _rotate_key(self):
        if len(self.api_keys) > 1:
            prev_idx = self.key_index
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            print(f"[GroqClient] Rotating API Key from Key #{prev_idx+1} -> Key #{self.key_index+1}...")
            self._init_client()
            return True
        return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> str:
        target_model = model or self.model

        if self.debug_mode:
            user_msg = messages[-1]["content"] if messages else ""
            return f"[MOCK GROQ GENERATION ({target_model})]: Answer to query '{user_msg[:60]}...' based on provided context."

        max_attempts = 12
        reconnect = 0
        last_error = None
        
        while reconnect < max_attempts:
            try:
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                reconnect += 1
                last_error = e
                err_str = str(e)
                
                # Check for 429 Rate Limits
                if "429" in err_str or "rate_limit" in err_str:
                    # Attempt key rotation if multiple keys configured
                    if len(self.api_keys) > 1:
                        self._rotate_key()
                        time.sleep(1.0)
                        continue

                    # Fallback to retry wait if only 1 key
                    match_min = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', err_str)
                    if match_min:
                        mins = float(match_min.group(1)) if match_min.group(1) else 0.0
                        secs = float(match_min.group(2)) if match_min.group(2) else 5.0
                        total_wait = (mins * 60.0) + secs + 2.0
                        print(f"[GroqClient] Rate limit hit. Waiting {total_wait:.1f}s (attempt {reconnect}/{max_attempts})...")
                        time.sleep(total_wait)
                    else:
                        sleep_time = 5.0 * reconnect
                        print(f"[GroqClient] Rate limit hit. Sleeping {sleep_time:.1f}s (attempt {reconnect}/{max_attempts})...")
                        time.sleep(sleep_time)
                else:
                    print(f"[GroqClient] Exception on attempt {reconnect}/{max_attempts}: {e}")
                    time.sleep(2.0)

        raise RuntimeError(f"[GroqClient] ERROR: Real Groq API request failed after {max_attempts} attempts: {last_error}")
