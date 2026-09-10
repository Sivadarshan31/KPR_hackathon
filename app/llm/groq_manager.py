import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union
import httpx
import groq
from pydantic import BaseModel, ValidationError
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_groq import ChatGroq

from app.config import DEFAULT_TEMPERATURE, GROQ_API_KEYS, GROQ_MODEL, DEFAULT_MAX_TOKENS
from app.llm.exceptions import (
    GroqConfigurationError,
    GroqRequestError,
    LLMAllKeysExhaustedError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    sanitize_error_message,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Exceptions that trigger key fallback in LangChain RunnableWithFallbacks
RETRYABLE_EXCEPTIONS = (
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.AuthenticationError,
    groq.PermissionDeniedError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


def is_retryable_error(exc: Exception) -> bool:
    """
    Determines whether a provider exception is retryable (rate limits, timeouts,
    transient server errors, key invalidation) or non-retryable (bad requests,
    schema validation errors, programming errors).
    """
    # Explicit non-retryable exceptions
    if isinstance(exc, (
        groq.BadRequestError,
        groq.UnprocessableEntityError,
        groq.NotFoundError,
        ValueError,
        TypeError,
        KeyError,
        GroqConfigurationError,
        LLMConfigurationError,
        LLMResponseError,
        ValidationError,
    )):
        return False

    # Explicit retryable exceptions
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        return True

    # Check status_code attribute on APIStatusError or HTTP exceptions
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        if status_code in (401, 403, 429) or status_code >= 500:
            return True
        return False

    # Fallback inspection of error message text
    err_str = str(exc).lower()
    retryable_markers = (
        "rate limit",
        "429",
        "quota",
        "timeout",
        "temporarily unavailable",
        "500",
        "502",
        "503",
        "504",
        "connection error",
        "unauthorized",
        "invalid api key",
    )
    return any(marker in err_str for marker in retryable_markers)


class GroqManager:
    """
    Centralized LLM manager for Groq API integration across ContentForge agents.
    Provides key rotation and fallback across configured Groq API keys.
    Maintains zero-secret-leakage guarantees, structured JSON parsing, and full LangChain compatibility.
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: Optional[str] = None,
        default_temperature: float = DEFAULT_TEMPERATURE,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        raw_keys = api_keys if api_keys is not None else GROQ_API_KEYS
        self._api_keys: List[str] = [k.strip() for k in raw_keys if k and k.strip()]
        self.model: str = model or GROQ_MODEL
        self.default_temperature: float = default_temperature
        self.default_max_tokens: int = default_max_tokens

        logger.info(
            "Groq manager initialized with %d configured keys.",
            len(self._api_keys),
        )

    @property
    def configured_keys_count(self) -> int:
        """Returns the count of loaded, non-empty Groq API keys."""
        return len(self._api_keys)

    def is_configured(self) -> bool:
        """Checks if at least one Groq API key is configured."""
        return len(self._api_keys) > 0

    def get_client(
        self,
        key_slot: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatGroq:
        """
        Constructs and returns a ChatGroq client for a specific key slot (1-indexed).
        Never exposes the API key in logs, representations, or returns.
        """
        if not self.is_configured():
            raise LLMConfigurationError("No Groq API keys are configured.")

        total_keys = len(self._api_keys)
        if key_slot < 1 or key_slot > total_keys:
            raise ValueError(
                f"Invalid key slot: {key_slot}. Configured slots: 1 to {total_keys}."
            )

        key = self._api_keys[key_slot - 1]
        temp = self.default_temperature if temperature is None else temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        return ChatGroq(
            groq_api_key=key,
            model_name=self.model,
            temperature=temp,
            max_tokens=tokens,
        )

    def get_llm(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        with_fallback: bool = True,
    ) -> Union[BaseChatModel, RunnableWithFallbacks]:
        """
        Returns a LangChain-compatible LLM instance for agent usage.
        If multiple keys are configured and with_fallback=True, returns a
        RunnableWithFallbacks chaining primary (slot 1) -> fallback (slot 2) -> fallback (slot 3).
        Supports .invoke(), .ainvoke(), .stream(), and .with_structured_output().
        """
        if not self.is_configured():
            raise LLMConfigurationError("No Groq API keys are configured.")

        temp = self.default_temperature if temperature is None else temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        primary_client = self.get_client(key_slot=1, temperature=temp, max_tokens=tokens)

        if not with_fallback or self.configured_keys_count <= 1:
            return primary_client

        fallback_clients = [
            self.get_client(key_slot=i, temperature=temp, max_tokens=tokens)
            for i in range(2, self.configured_keys_count + 1)
        ]

        return primary_client.with_fallbacks(
            fallbacks=fallback_clients,
            exceptions_to_handle=RETRYABLE_EXCEPTIONS,
        )

    def invoke(
        self,
        input_data: Union[str, Sequence[BaseMessage], Any],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Executes a synchronous request with explicit key fallback across slots (1 -> 2 -> 3).
        Rotates only on retryable errors and raises sanitized LLMAllKeysExhaustedError on failure.
        """
        if not self.is_configured():
            raise LLMConfigurationError("No Groq API keys are configured.")

        messages = (
            [HumanMessage(content=input_data)]
            if isinstance(input_data, str)
            else input_data
        )

        total_keys = len(self._api_keys)
        last_exception: Optional[Exception] = None

        logger.info("Groq request started.")

        for slot in range(1, total_keys + 1):
            try:
                client = self.get_client(key_slot=slot, temperature=temperature)
                response = client.invoke(messages, **kwargs)
                logger.info("Groq request succeeded using configured key slot %d.", slot)
                return response
            except Exception as exc:
                last_exception = exc
                sanitized_err = sanitize_error_message(str(exc), self._api_keys)

                if not is_retryable_error(exc):
                    logger.warning(
                        "Groq request encountered non-retryable error on key slot %d; not rotating.",
                        slot,
                    )
                    raise GroqRequestError(
                        f"Non-retryable Groq request error: {sanitized_err}",
                        original_error=exc,
                    ) from exc

                if slot < total_keys:
                    logger.warning(
                        "Groq request received a retryable provider error on key slot %d; trying next key.",
                        slot,
                    )
                else:
                    logger.error(
                        "Groq request failed after all %d configured keys were attempted.",
                        total_keys,
                    )

        raise LLMAllKeysExhaustedError(
            f"Groq request failed after attempting all {total_keys} configured keys. All configured Groq API keys are currently unavailable. Last error: {sanitize_error_message(str(last_exception), self._api_keys)}",
            original_error=last_exception,
        )

    async def ainvoke(
        self,
        input_data: Union[str, Sequence[BaseMessage], Any],
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Executes an asynchronous request with explicit key fallback across slots (1 -> 2 -> 3).
        Rotates only on retryable errors and raises sanitized LLMAllKeysExhaustedError on failure.
        """
        if not self.is_configured():
            raise LLMConfigurationError("No Groq API keys are configured.")

        messages = (
            [HumanMessage(content=input_data)]
            if isinstance(input_data, str)
            else input_data
        )

        total_keys = len(self._api_keys)
        last_exception: Optional[Exception] = None

        logger.info("Groq request started.")

        for slot in range(1, total_keys + 1):
            try:
                client = self.get_client(key_slot=slot, temperature=temperature)
                response = await client.ainvoke(messages, **kwargs)
                logger.info("Groq request succeeded using configured key slot %d.", slot)
                return response
            except Exception as exc:
                last_exception = exc
                sanitized_err = sanitize_error_message(str(exc), self._api_keys)

                if not is_retryable_error(exc):
                    logger.warning(
                        "Groq request encountered non-retryable error on key slot %d; not rotating.",
                        slot,
                    )
                    raise GroqRequestError(
                        f"Non-retryable Groq request error: {sanitized_err}",
                        original_error=exc,
                    ) from exc

                if slot < total_keys:
                    logger.warning(
                        "Groq request received a retryable provider error on key slot %d; trying next key.",
                        slot,
                    )
                else:
                    logger.error(
                        "Groq request failed after all %d configured keys were attempted.",
                        total_keys,
                    )

        raise LLMAllKeysExhaustedError(
            f"Groq request failed after attempting all {total_keys} configured keys. All configured Groq API keys are currently unavailable. Last error: {sanitize_error_message(str(last_exception), self._api_keys)}",
            original_error=last_exception,
        )

    async def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Convenience async helper sending a prompt and returning the response text.
        Maintains backwards-compatibility with existing API endpoints.
        """
        response = await self.ainvoke(
            [HumanMessage(content=prompt)],
            temperature=temperature,
        )
        return str(response.content)

    def parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON content from an LLM response string.
        Strips markdown code blocks (```json ... ```) if present.
        Raises LLMResponseError on invalid JSON.
        """
        if not text or not text.strip():
            raise LLMResponseError("LLM response text is empty.", raw_response=text)

        cleaned = text.strip()
        # Regex to strip ```json ... ``` code fence if present
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise LLMResponseError(
                    f"Expected JSON object (dict), got {type(parsed).__name__}.",
                    raw_response=text,
                )
            return parsed
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Failed to decode valid JSON from LLM response: {exc}",
                original_error=exc,
                raw_response=text,
            ) from exc

    def parse_structured_output(
        self,
        text_or_dict: Union[str, Dict[str, Any]],
        schema_cls: Type[T],
    ) -> T:
        """
        Parses and validates JSON data into a Pydantic model.
        Raises LLMResponseError on validation failure.
        """
        if isinstance(text_or_dict, str):
            data = self.parse_json_response(text_or_dict)
        else:
            data = text_or_dict

        try:
            return schema_cls.model_validate(data)
        except ValidationError as exc:
            raise LLMResponseError(
                f"LLM output failed Pydantic validation for {schema_cls.__name__}: {exc}",
                original_error=exc,
                raw_response=str(text_or_dict),
            ) from exc


# Reusable default instance
groq_manager = GroqManager()
