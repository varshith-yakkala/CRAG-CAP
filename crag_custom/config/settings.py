import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    T5_MODEL_PATH: str = field(
        default_factory=lambda: os.getenv(
            "T5_MODEL_PATH",
            r"c:\Users\varsh\OneDrive\Desktop\2-2\capstone\t5_evaluator_model\t5_evaluator_final"
        )
    )
    T5_TOKENIZER_PATH: str = field(
        default_factory=lambda: os.getenv("T5_TOKENIZER_PATH", "google-t5/t5-small")
    )
    DEVICE: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GROQ_API_KEYS: List[str] = field(
        default_factory=lambda: [
            k.strip() for k in os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", "")).split(",") if k.strip()
        ]
    )
    GROQ_MODEL: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    )
    GROQ_REWRITER_MODEL: str = field(
        default_factory=lambda: os.getenv("GROQ_REWRITER_MODEL", "groq/compound-mini")
    )
    
    SEARCH_API_KEY: str = field(default_factory=lambda: os.getenv("SEARCH_API_KEY", ""))
    SEARCH_PROVIDER: str = field(default_factory=lambda: os.getenv("SEARCH_PROVIDER", "serper"))
    
    TOP_K: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    DECOMPOSE_MODE: str = field(default_factory=lambda: os.getenv("DECOMPOSE_MODE", "selection"))
    
    DEBUG_MODE: bool = field(
        default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")
    )

settings = Settings()
