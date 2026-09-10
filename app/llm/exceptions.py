import re
from typing import List, Optional


def sanitize_error_message(message: str, keys: Optional[List[str]] = None) -> str:
    """
    Sanitizes sensitive information (Groq API keys, Bearer tokens, secrets)
    from error strings and exception messages.
    """
    if not message:
        return ""

    sanitized = str(message)

    # Redact common Groq key patterns: gsk_...
    sanitized = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED_GROQ_KEY]", sanitized)

    # Redact Authorization / Bearer tokens
    sanitized = re.sub(r"(?i)bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", sanitized)

    # Redact any explicitly passed keys if present
    if keys:
        for k in keys:
            if k and len(k) > 4:
                sanitized = sanitized.replace(k, "[REDACTED_GROQ_KEY]")

    return sanitized


class GroqConfigurationError(Exception):
    """
    Raised when Groq API keys or required configuration values are missing or invalid.
    Never exposes secrets in the error representation.
    """

    def __init__(self, message: str = "No Groq API keys are configured."):
        super().__init__(sanitize_error_message(message))


class GroqRequestError(Exception):
    """
    Raised when a Groq request fails after attempting all configured keys or
    when encountering a non-retryable provider error.
    Never exposes secrets in the error representation.
    """

    def __init__(
        self,
        message: str = "Groq request failed after trying configured keys.",
        original_error: Optional[Exception] = None,
    ):
        sanitized_msg = sanitize_error_message(message)
        super().__init__(sanitized_msg)
        self.original_error = original_error


class LLMError(Exception):
    """
    Base application-level exception for LLM operational errors.
    Ensures sanitized messages and maintains original error references.
    """

    def __init__(
        self,
        message: str = "An LLM error occurred.",
        original_error: Optional[Exception] = None,
    ):
        sanitized_msg = sanitize_error_message(message)
        super().__init__(sanitized_msg)
        self.original_error = original_error


class LLMConfigurationError(LLMError, GroqConfigurationError):
    """
    Raised when Groq API keys or required configuration settings are invalid or missing.
    Inherits from GroqConfigurationError for backwards compatibility.
    """

    def __init__(self, message: str = "No Groq API keys are configured."):
        LLMError.__init__(self, message)
        GroqConfigurationError.__init__(self, message)


class LLMRateLimitError(LLMError):
    """
    Raised when provider rate limits or quotas are exceeded.
    """

    def __init__(
        self,
        message: str = "Groq rate limit or quota exceeded.",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, original_error=original_error)


class LLMAllKeysExhaustedError(LLMError, GroqRequestError):
    """
    Raised when all configured Groq API keys fail after key rotation.
    Inherits from GroqRequestError for backwards compatibility.
    """

    def __init__(
        self,
        message: str = "All configured Groq API keys are currently unavailable.",
        original_error: Optional[Exception] = None,
    ):
        LLMError.__init__(self, message, original_error=original_error)
        GroqRequestError.__init__(self, message, original_error=original_error)


class LLMResponseError(LLMError):
    """
    Raised when an LLM response cannot be parsed or validated as valid structured JSON.
    """

    def __init__(
        self,
        message: str = "Failed to parse or validate structured JSON response from LLM.",
        original_error: Optional[Exception] = None,
        raw_response: Optional[str] = None,
    ):
        super().__init__(message, original_error=original_error)
        self.raw_response = raw_response
