import unittest
from unittest.mock import MagicMock, patch
import httpx
import groq
from pydantic import ValidationError

from app.config import settings
from app.llm.exceptions import (
    LLMAllKeysExhaustedError,
    LLMConfigurationError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    GroqRequestError,
    sanitize_error_message,
)
from app.llm.groq_manager import GroqManager
from app.schemas import (
    UserRequest,
    SourceUnderstanding,
    ContentStrategy,
    LinkedInStrategy,
    InstagramStrategy,
    AdvisoryStrategy,
    LinkedInContent,
    InstagramContent,
    AdvisoryContent,
    GeneratedContent,
    ValidationResult,
    ValidationStatus,
    ActionResult,
)


class TestModule3LLMResilience(unittest.TestCase):
    """Tests for Module 3: LLM Resilience, Error Handling, and Structured Output Parsing."""

    def test_01_single_and_multiple_keys_detection(self):
        gm1 = GroqManager(api_keys=["gsk_testkey1"])
        self.assertEqual(gm1.configured_keys_count, 1)
        self.assertTrue(gm1.is_configured())

        gm3 = GroqManager(api_keys=["gsk_k1", "gsk_k2", "gsk_k3"])
        self.assertEqual(gm3.configured_keys_count, 3)

    def test_02_unconfigured_manager_raises_llm_configuration_error(self):
        gm = GroqManager(api_keys=[])
        self.assertFalse(gm.is_configured())
        with self.assertRaises(LLMConfigurationError):
            gm.get_client()

    def test_03_key_rotation_on_rate_limit(self):
        gm = GroqManager(api_keys=["gsk_key1", "gsk_key2"])
        mock_client1 = MagicMock()
        mock_client1.invoke.side_effect = groq.RateLimitError(
            message="Rate limit reached",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
            body=None,
        )
        mock_client2 = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Success from key 2"
        mock_client2.invoke.return_value = mock_response

        def mock_get_client(key_slot=1, **kwargs):
            if key_slot == 1:
                return mock_client1
            return mock_client2

        with patch.object(gm, "get_client", side_effect=mock_get_client):
            res = gm.invoke("Hello")
            self.assertEqual(res.content, "Success from key 2")

    def test_04_all_keys_exhausted_raises_controlled_exception(self):
        gm = GroqManager(api_keys=["gsk_key1", "gsk_key2"])
        mock_client = MagicMock()
        mock_client.invoke.side_effect = groq.RateLimitError(
            message="Quota exceeded",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
            body=None,
        )

        with patch.object(gm, "get_client", return_value=mock_client):
            with self.assertRaises(LLMAllKeysExhaustedError) as cm:
                gm.invoke("Test prompt")
            self.assertIn("unavailable", str(cm.exception).lower())
            self.assertIsInstance(cm.exception, GroqRequestError)
            self.assertIsInstance(cm.exception, LLMError)

    def test_05_non_retryable_error_prevents_pointless_rotation(self):
        gm = GroqManager(api_keys=["gsk_key1", "gsk_key2"])
        mock_client1 = MagicMock()
        mock_client1.invoke.side_effect = groq.BadRequestError(
            message="Invalid prompt length",
            response=httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com")),
            body=None,
        )
        mock_client2 = MagicMock()

        with patch.object(gm, "get_client", side_effect=[mock_client1, mock_client2]):
            with self.assertRaises(GroqRequestError) as cm:
                gm.invoke("Test")
            self.assertIn("Non-retryable", str(cm.exception))
            mock_client2.invoke.assert_not_called()

    def test_06_invalid_json_produces_llm_response_error(self):
        gm = GroqManager(api_keys=["gsk_key1"])
        invalid_raw_text = "Here is the response: { invalid_json: "
        with self.assertRaises(LLMResponseError) as cm:
            gm.parse_json_response(invalid_raw_text)
        self.assertEqual(cm.exception.raw_response, invalid_raw_text)

    def test_07_valid_json_extraction_from_markdown_code_fence(self):
        gm = GroqManager(api_keys=["gsk_key1"])
        raw_text = "```json\n{\"status\": \"ok\", \"count\": 42}\n```"
        parsed = gm.parse_json_response(raw_text)
        self.assertEqual(parsed, {"status": "ok", "count": 42})

    def test_08_secret_sanitization_in_errors(self):
        secret_key = "gsk_1234567890abcdef1234567890"
        err_msg = f"Failed with key {secret_key} and Bearer token secret_token_123"
        sanitized = sanitize_error_message(err_msg, keys=[secret_key])
        self.assertNotIn(secret_key, sanitized)
        self.assertIn("[REDACTED_GROQ_KEY]", sanitized)


class TestModule4Schemas(unittest.TestCase):
    """Tests for Module 4: Request & Response Pydantic Schemas."""

    def test_01_user_request_valid_and_invalid(self):
        req = UserRequest(source_text="Valid source text content.")
        self.assertEqual(req.source_text, "Valid source text content.")
        self.assertEqual(req.requested_platform, "all")

        with self.assertRaises(ValidationError):
            UserRequest(source_text="   ")

        with self.assertRaises(ValidationError):
            UserRequest(source_text="Valid", requested_platform="unsupported_platform")

    def test_02_source_understanding_valid(self):
        su = SourceUnderstanding(
            title="Sample Title",
            summary="Sample summary",
            main_topic="Topic",
            key_points=["Point 1"],
            facts=["Fact 1"],
            entities=["Entity A"],
            important_numbers=["100%"],
            dates=["2026"],
            claims=["Claim A"],
            terminology=["Term A"],
        )
        self.assertEqual(su.title, "Sample Title")
        self.assertEqual(su.important_facts, ["Fact 1"])
        self.assertEqual(su.numbers, ["100%"])
        self.assertEqual(su.important_terminology, ["Term A"])

    def test_03_source_understanding_missing_required_fields(self):
        with self.assertRaises(ValidationError):
            SourceUnderstanding(title="Title only")

    def test_04_content_strategy_valid(self):
        cs = ContentStrategy(
            summary="Strategy summary",
            overall_angle="Tech Innovation",
            target_audience="Engineers",
            key_takeaway="Takeaway",
            linkedin=LinkedInStrategy(
                objective="Thought Leadership",
                audience="Tech Leads",
                angle="Industry Shift",
                key_message="Insight",
                tone="Professional",
                cta="Comment below",
                recommended_structure=["Hook", "Body", "CTA"],
            ),
        )
        self.assertEqual(cs.content_angle, "Tech Innovation")
        self.assertEqual(cs.core_message, "Takeaway")
        self.assertIsNotNone(cs.linkedin)

    def test_05_generated_content_linkedin_instagram_advisory(self):
        # Structured LinkedIn & Advisory
        gc = GeneratedContent(
            linkedin=LinkedInContent(
                hook="Did you know?",
                body="AI is transforming workflows.",
                cta="Share your thoughts.",
                hashtags=["#AI", "#Tech"],
            ),
            instagram=InstagramContent(
                caption="Visual summary caption",
                slides=["Slide 1", "Slide 2"],
                hashtags=["#InstaTech"],
                call_to_action="Save for later",
            ),
            advisory=AdvisoryContent(
                title="Executive Brief",
                summary="Advisory summary",
                important_information=["Key info 1"],
                recommended_actions=["Action 1"],
            ),
        )
        self.assertEqual(gc.linkedin.hook, "Did you know?")
        self.assertEqual(gc.instagram.slides, ["Slide 1", "Slide 2"])
        self.assertEqual(gc.advisory.title, "Executive Brief")

        # String format compatibility
        gc_string = GeneratedContent(
            linkedin="Simple string post",
            advisory="Simple advisory text",
        )
        self.assertEqual(gc_string.linkedin, "Simple string post")

    def test_06_validation_result_valid_pass_and_fail(self):
        vr_pass = ValidationResult(
            status=ValidationStatus.PASS,
            passed=True,
            score=95.0,
            issues=[],
            suggestions=[],
        )
        self.assertEqual(vr_pass.status, ValidationStatus.PASS)

        vr_fail = ValidationResult(
            status="FAIL",
            passed=False,
            score=45.0,
            issues=["Fact mismatch"],
            suggestions=["Fix numbers"],
        )
        self.assertEqual(vr_fail.status, ValidationStatus.FAIL)

    def test_07_validation_result_score_out_of_range(self):
        with self.assertRaises(ValidationError):
            ValidationResult(score=-10.0, passed=False)

        with self.assertRaises(ValidationError):
            ValidationResult(score=105.0, passed=True)

    def test_08_action_result_valid_and_invalid_action(self):
        ar = ActionResult(
            success=True,
            action="export",
            platform="all",
            action_allowed=True,
            status="completed",
            message="Export payload ready",
            validation_status="PASS",
        )
        self.assertTrue(ar.success)
        self.assertEqual(ar.action, "export")


if __name__ == "__main__":
    unittest.main()
