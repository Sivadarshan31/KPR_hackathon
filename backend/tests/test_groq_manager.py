import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
import groq
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

from app.llm.exceptions import (
    GroqConfigurationError,
    GroqRequestError,
    sanitize_error_message,
)
from app.llm.groq_manager import GroqManager, is_retryable_error


class SampleStructuredOutput(BaseModel):
    summary: str = Field(description="Sample summary field")
    confidence: float = Field(description="Confidence score")


class TestGroqManager(unittest.IsolatedAsyncioTestCase):
    """
    Comprehensive test suite for GroqManager:
    - Key configuration handling
    - Controlled fallback on retryable errors
    - Immediate failure on non-retryable errors
    - Secret leakage prevention
    - LangChain and structured output compatibility
    """

    def setUp(self):
        self.mock_keys = [
            "mock_key_alpha_1111111111",
            "mock_key_beta_2222222222",
            "mock_key_gamma_3333333333",
        ]

    def test_01_three_keys_configured(self):
        """Test 1: Three keys configured -> configured_keys_count is 3."""
        manager = GroqManager(api_keys=self.mock_keys)
        self.assertEqual(manager.configured_keys_count, 3)
        self.assertTrue(manager.is_configured())

    def test_02_one_key_configured(self):
        """Test 2: One key configured -> configured_keys_count is 1."""
        manager = GroqManager(api_keys=["mock_single_key"])
        self.assertEqual(manager.configured_keys_count, 1)
        self.assertTrue(manager.is_configured())

    def test_03_empty_keys_ignored(self):
        """Test 3: Empty and whitespace keys are ignored."""
        keys = ["mock_key_1", "   ", "", "mock_key_3", None]
        manager = GroqManager(api_keys=keys)
        self.assertEqual(manager.configured_keys_count, 2)
        # Verify valid slots are 1 and 2
        client1 = manager.get_client(1)
        client2 = manager.get_client(2)
        self.assertIsNotNone(client1)
        self.assertIsNotNone(client2)

    def test_04_zero_keys_raises_configuration_error(self):
        """Test 4: Zero configured keys raises GroqConfigurationError on call."""
        manager = GroqManager(api_keys=[])
        self.assertEqual(manager.configured_keys_count, 0)
        self.assertFalse(manager.is_configured())

        with self.assertRaises(GroqConfigurationError) as ctx:
            manager.get_client(1)
        self.assertIn("No Groq API keys are configured", str(ctx.exception))

        with self.assertRaises(GroqConfigurationError):
            manager.get_llm()

        with self.assertRaises(GroqConfigurationError):
            manager.invoke("Hello")

    @patch("app.llm.groq_manager.ChatGroq")
    def test_05_first_key_succeeds_no_fallback(self, mock_chat_groq):
        """Test 5: First key succeeds -> only Key slot 1 is used, no fallback."""
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = AIMessage(content="SUCCESS_RESPONSE")
        mock_chat_groq.return_value = mock_instance

        manager = GroqManager(api_keys=self.mock_keys)
        response = manager.invoke("Test prompt")

        self.assertEqual(response.content, "SUCCESS_RESPONSE")
        self.assertEqual(mock_chat_groq.call_count, 1)
        # Verify key used was slot 1
        _, kwargs = mock_chat_groq.call_args
        self.assertEqual(kwargs.get("groq_api_key"), self.mock_keys[0])

    @patch("app.llm.groq_manager.ChatGroq")
    def test_06_first_key_retryable_error_falls_back_to_key_2(self, mock_chat_groq):
        """Test 6: First key fails with retryable error (RateLimit) -> Key slot 2 succeeds."""
        mock_client_1 = MagicMock()
        rate_limit_resp = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        mock_client_1.invoke.side_effect = groq.RateLimitError(
            message="Rate limit reached",
            response=rate_limit_resp,
            body={"error": {"message": "Rate limit reached"}},
        )

        mock_client_2 = MagicMock()
        mock_client_2.invoke.return_value = AIMessage(content="RECOVERED_ON_KEY_2")

        mock_chat_groq.side_effect = [mock_client_1, mock_client_2]

        manager = GroqManager(api_keys=self.mock_keys)
        response = manager.invoke("Test prompt")

        self.assertEqual(response.content, "RECOVERED_ON_KEY_2")
        self.assertEqual(mock_chat_groq.call_count, 2)
        mock_client_1.invoke.assert_called_once()
        mock_client_2.invoke.assert_called_once()

    @patch("app.llm.groq_manager.ChatGroq")
    def test_07_first_and_second_fail_retryably_key_3_succeeds(self, mock_chat_groq):
        """Test 7: Key 1 and Key 2 fail retryably -> Key 3 is attempted and succeeds."""
        resp_500 = httpx.Response(
            status_code=500,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        mock_client_1 = MagicMock()
        mock_client_1.invoke.side_effect = groq.InternalServerError(
            message="Internal Server Error",
            response=resp_500,
            body={"error": {"message": "Server error"}},
        )

        mock_client_2 = MagicMock()
        mock_client_2.invoke.side_effect = groq.APIConnectionError(
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
            message="Connection timeout",
        )

        mock_client_3 = MagicMock()
        mock_client_3.invoke.return_value = AIMessage(content="RECOVERED_ON_KEY_3")

        mock_chat_groq.side_effect = [mock_client_1, mock_client_2, mock_client_3]

        manager = GroqManager(api_keys=self.mock_keys)
        response = manager.invoke("Test prompt")

        self.assertEqual(response.content, "RECOVERED_ON_KEY_3")
        self.assertEqual(mock_chat_groq.call_count, 3)
        mock_client_1.invoke.assert_called_once()
        mock_client_2.invoke.assert_called_once()
        mock_client_3.invoke.assert_called_once()

    @patch("app.llm.groq_manager.ChatGroq")
    def test_08_all_configured_keys_fail_raises_safe_error(self, mock_chat_groq):
        """Test 8: All configured keys fail -> raises GroqRequestError, no endless loop."""
        resp_429 = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )

        def make_failing_client():
            client = MagicMock()
            client.invoke.side_effect = groq.RateLimitError(
                message="Quota exhausted",
                response=resp_429,
                body={"error": {"message": "Quota exhausted"}},
            )
            return client

        mock_chat_groq.side_effect = [
            make_failing_client(),
            make_failing_client(),
            make_failing_client(),
        ]

        manager = GroqManager(api_keys=self.mock_keys)
        with self.assertRaises(GroqRequestError) as ctx:
            manager.invoke("Test prompt")

        self.assertEqual(mock_chat_groq.call_count, 3)
        self.assertIn("failed after attempting all 3 configured keys", str(ctx.exception))
        # Ensure no secrets in exception message
        for key in self.mock_keys:
            self.assertNotIn(key, str(ctx.exception))

    @patch("app.llm.groq_manager.ChatGroq")
    def test_09_non_retryable_error_does_not_rotate(self, mock_chat_groq):
        """Test 9: Non-retryable error (e.g. 400 BadRequest) fails immediately without rotating."""
        resp_400 = httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        mock_client_1 = MagicMock()
        mock_client_1.invoke.side_effect = groq.BadRequestError(
            message="Invalid prompt structure",
            response=resp_400,
            body={"error": {"message": "Invalid prompt structure"}},
        )
        mock_chat_groq.return_value = mock_client_1

        manager = GroqManager(api_keys=self.mock_keys)
        with self.assertRaises(GroqRequestError) as ctx:
            manager.invoke("Bad request prompt")

        # Slot 2 and 3 should NOT have been attempted
        self.assertEqual(mock_chat_groq.call_count, 1)
        self.assertIn("Non-retryable", str(ctx.exception))

    def test_10_secret_leakage_prevention(self):
        """Test 10: Verify secret strings, Bearer tokens, and gsk_ keys are sanitized."""
        secret_key = "gsk_super_secret_production_key_12345"
        raw_msg = f"Error communicating with provider with key {secret_key} and Bearer token123456"

        sanitized = sanitize_error_message(raw_msg, [secret_key])
        self.assertNotIn(secret_key, sanitized)
        self.assertNotIn("Bearer token123456", sanitized)
        self.assertIn("[REDACTED", sanitized)

        # Verify GroqConfigurationError sanitizes message
        cfg_err = GroqConfigurationError(f"Failed with key {secret_key}")
        self.assertNotIn(secret_key, str(cfg_err))

        # Verify GroqRequestError sanitizes message
        req_err = GroqRequestError(f"Provider failed with {secret_key}")
        self.assertNotIn(secret_key, str(req_err))

    def test_11_structured_output_compatibility(self):
        """Test 11: get_llm() returns a model compatible with .with_structured_output()."""
        manager = GroqManager(api_keys=self.mock_keys)
        llm = manager.get_llm()

        # Check that with_structured_output exists and produces a runnable
        self.assertTrue(hasattr(llm, "with_structured_output"))
        structured_llm = llm.with_structured_output(SampleStructuredOutput)
        self.assertTrue(hasattr(structured_llm, "invoke"))
        self.assertTrue(hasattr(structured_llm, "ainvoke"))

    @patch("app.llm.groq_manager.ChatGroq")
    async def test_12_async_ainvoke_fallback(self, mock_chat_groq):
        """Test 12: Async ainvoke falls back from key 1 to key 2 on retryable error."""
        resp_429 = httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
        mock_client_1 = MagicMock()
        mock_client_1.ainvoke = AsyncMock(
            side_effect=groq.RateLimitError(
                message="Rate limit",
                response=resp_429,
                body={"error": {"message": "Rate limit"}},
            )
        )

        mock_client_2 = MagicMock()
        mock_client_2.ainvoke = AsyncMock(
            return_value=AIMessage(content="ASYNC_SUCCESS_ON_KEY_2")
        )

        mock_chat_groq.side_effect = [mock_client_1, mock_client_2]

        manager = GroqManager(api_keys=self.mock_keys)
        response = await manager.ainvoke("Async test prompt")

        self.assertEqual(response.content, "ASYNC_SUCCESS_ON_KEY_2")
        self.assertEqual(mock_chat_groq.call_count, 2)


if __name__ == "__main__":
    unittest.main()
