import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized Pydantic Settings model for ContentForge backend configuration.
    Reads environment variables from system environment or .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Metadata
    app_title: str = Field(default="ContentForge API", alias="APP_TITLE")
    app_description: str = Field(
        default="AI-Powered Multi-Format Content Transformation Platform",
        alias="APP_DESCRIPTION",
    )
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")

    # Groq Model & Generation Configuration
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.2, alias="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=2500, alias="GROQ_MAX_TOKENS")

    # Groq API Keys for Fallback & Rotation
    groq_api_key_1: Optional[str] = Field(default=None, alias="GROQ_API_KEY_1")
    groq_api_key_2: Optional[str] = Field(default=None, alias="GROQ_API_KEY_2")
    groq_api_key_3: Optional[str] = Field(default=None, alias="GROQ_API_KEY_3")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")

    def get_groq_api_keys(self) -> List[str]:
        """Reads configured Groq keys in priority order (key 1 -> 2 -> 3 -> default)."""
        raw_keys = [
            self.groq_api_key_1,
            self.groq_api_key_2,
            self.groq_api_key_3,
            self.groq_api_key,
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


# Global Settings instance
settings = Settings()

# Backwards-compatible Module-level Exports & helper functions
APP_TITLE: str = settings.app_title
APP_DESCRIPTION: str = settings.app_description
APP_VERSION: str = settings.app_version

GROQ_MODEL: str = settings.groq_model
DEFAULT_TEMPERATURE: float = settings.groq_temperature
DEFAULT_MAX_TOKENS: int = settings.groq_max_tokens


def get_groq_api_keys() -> List[str]:
    """Reads configured keys dynamically from settings or environment."""
    return settings.get_groq_api_keys()


GROQ_API_KEYS: List[str] = get_groq_api_keys()


def get_configured_keys_count() -> int:
    """Returns the number of configured Groq API keys without exposing secrets."""
    return len(get_groq_api_keys())


def get_groq_status_summary() -> str:
    """Safe diagnostic summary showing only the count of configured keys and model."""
    return f"Groq keys loaded: {get_configured_keys_count()} | Model: {GROQ_MODEL}"