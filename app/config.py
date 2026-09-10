import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Application Metadata
APP_TITLE = "ContentForge AI Backend"
APP_DESCRIPTION = "Agentic AI backend for intelligent multi-format content transformation."
APP_VERSION = "0.1.0"

# Groq Model & Generation Configuration
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "2500"))

def get_groq_api_keys() -> List[str]:
    """Reads configured keys dynamically from environment or .env."""
    load_dotenv(override=True)
    raw_keys = [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY"),
    ]
    seen = set()
    deduped: List[str] = []
    for key in raw_keys:
        if key and key.strip():
            k_val = key.strip()
            if k_val not in seen:
                seen.add(k_val)
                deduped.append(k_val)
    return deduped


GROQ_API_KEYS: List[str] = get_groq_api_keys()


def get_configured_keys_count() -> int:
    """Returns the number of configured Groq API keys without exposing secrets."""
    return len(get_groq_api_keys())


def get_groq_status_summary() -> str:
    """Safe diagnostic summary showing only the count of configured keys and model."""
    return f"Groq keys loaded: {get_configured_keys_count()} | Model: {GROQ_MODEL}"