import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.schemas.source_understanding import (
    SourceUnderstandingRequest,
    SourceUnderstanding,
)
from app.agents.source_understanding import (
    SourceUnderstandingAgent,
    SYSTEM_PROMPT,
)
from app.llm.exceptions import GroqConfigurationError
from app.main import app


class TestSourceUnderstanding(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for Agent 1 (Source Understanding Agent):
    - Input schema validation (rejection of empty/whitespace input)
    - Output schema structure and defaults
    - Anti-hallucination prompt verification
    - Agent execution (sync and async with mocked LLM)
    - FastAPI endpoint integration via TestClient
    """

    def setUp(self):
        self.sample_source = (
            "The National Institute of Ocean Technology developed an autonomous underwater "
            "vehicle to monitor marine pollution. The vehicle uses sensors to collect water-quality "
            "measurements and can operate for up to 18 hours. The project began in 2026."
        )
        self.expected_data = {
            "title": "Autonomous Underwater Vehicle for Marine Pollution Monitoring",
            "summary": "NIOT developed an autonomous underwater vehicle to monitor marine pollution.",
            "main_topic": "Autonomous underwater vehicle for marine pollution monitoring",
            "key_points": [
                "Autonomous underwater vehicle was developed",
                "Designed to monitor marine pollution",
                "Operates for up to 18 hours",
                "Project began in 2026",
            ],
            "facts": [
                "The National Institute of Ocean Technology developed the vehicle",
                "The vehicle operates for up to 18 hours",
                "The project began in 2026",
            ],
            "entities": ["National Institute of Ocean Technology"],
            "important_numbers": ["18 hours"],
            "dates": ["2026"],
            "claims": [],
            "terminology": ["autonomous underwater vehicle", "marine pollution", "sensors"],
            "target_audience": "Not specified",
            "tone": "Technical",
            "source_type": "Technical document",
        }

    def test_01_request_schema_validation(self):
        """Test 1: Input schema validation accepts valid text, rejects empty and whitespace."""
        # Valid text
        req = SourceUnderstandingRequest(source_text="  Some valid source text.  ")
        self.assertEqual(req.source_text, "Some valid source text.")

        # Empty string
        with self.assertRaises(ValidationError):
            SourceUnderstandingRequest(source_text="")

        # Whitespace-only string
        with self.assertRaises(ValidationError):
            SourceUnderstandingRequest(source_text="   \n\t   ")

    def test_02_output_schema_structure_and_defaults(self):
        """Test 2: SourceUnderstanding defaults are correctly assigned."""
        obj = SourceUnderstanding(
            title="Test Title",
            summary="Test Summary",
            main_topic="Test Topic",
        )
        self.assertEqual(obj.title, "Test Title")
        self.assertEqual(obj.key_points, [])
        self.assertEqual(obj.facts, [])
        self.assertEqual(obj.entities, [])
        self.assertEqual(obj.important_numbers, [])
        self.assertEqual(obj.dates, [])
        self.assertEqual(obj.claims, [])
        self.assertEqual(obj.terminology, [])
        self.assertEqual(obj.target_audience, "Not specified")
        self.assertEqual(obj.tone, "Neutral")
        self.assertEqual(obj.source_type, "Unknown")

    def test_03_anti_hallucination_prompt_structure(self):
        """Test 3: Verify strict anti-hallucination and source-fidelity directives in system prompt."""
        self.assertIn("ONLY the supplied source text", SYSTEM_PROMPT)
        self.assertIn("Anti-Hallucination", SYSTEM_PROMPT)
        self.assertIn("Do NOT fabricate", SYSTEM_PROMPT)
        self.assertIn("Preserve numbers, statistics, percentages", SYSTEM_PROMPT)
        self.assertIn("separate verified facts from subjective claims", SYSTEM_PROMPT)
        self.assertIn("Do NOT generate social-media posts", SYSTEM_PROMPT)

    def test_04_agent_synchronous_understand_with_mock_llm(self):
        """Test 4: SourceUnderstandingAgent.understand() works with structured output."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        mock_structured_llm.invoke.return_value = SourceUnderstanding(**self.expected_data)
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_manager.get_llm.return_value = mock_llm

        agent = SourceUnderstandingAgent(manager=mock_manager, temperature=0.1)
        result = agent.understand(self.sample_source)

        self.assertIsInstance(result, SourceUnderstanding)
        self.assertEqual(result.title, self.expected_data["title"])
        self.assertEqual(result.important_numbers, ["18 hours"])
        self.assertEqual(result.dates, ["2026"])
        mock_structured_llm.invoke.assert_called_once()

    async def test_05_agent_asynchronous_aunderstand_with_mock_llm(self):
        """Test 5: SourceUnderstandingAgent.aunderstand() works with async structured output."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()

        mock_structured_llm.ainvoke = AsyncMock(
            return_value=SourceUnderstanding(**self.expected_data)
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_manager.get_llm.return_value = mock_llm

        agent = SourceUnderstandingAgent(manager=mock_manager, temperature=0.1)
        result = await agent.aunderstand(self.sample_source)

        self.assertIsInstance(result, SourceUnderstanding)
        self.assertEqual(result.title, self.expected_data["title"])
        mock_structured_llm.ainvoke.assert_called_once()

    def test_06_fastapi_endpoint_success(self):
        """Test 6: POST /api/agents/source-understanding returns 200 with structured JSON."""
        client = TestClient(app)
        mock_result = SourceUnderstanding(**self.expected_data)

        with patch(
            "app.api.source_understanding.source_understanding_agent.aunderstand",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/agents/source-understanding",
                json={"source_text": self.sample_source},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["title"], self.expected_data["title"])
        self.assertEqual(data["important_numbers"], ["18 hours"])
        self.assertEqual(data["dates"], ["2026"])
        self.assertEqual(data["target_audience"], "Not specified")

    def test_07_fastapi_endpoint_empty_input_rejected(self):
        """Test 7: Empty or whitespace input returns 422 Unprocessable Entity."""
        client = TestClient(app)

        # Empty string
        resp1 = client.post(
            "/api/agents/source-understanding",
            json={"source_text": ""},
        )
        self.assertEqual(resp1.status_code, 422)

        # Whitespace string
        resp2 = client.post(
            "/api/agents/source-understanding",
            json={"source_text": "     "},
        )
        self.assertEqual(resp2.status_code, 422)

    def test_08_fastapi_endpoint_configuration_error(self):
        """Test 8: GroqConfigurationError returns 503 without exposing secrets."""
        client = TestClient(app)

        with patch(
            "app.api.source_understanding.source_understanding_agent.aunderstand",
            new_callable=AsyncMock,
            side_effect=GroqConfigurationError("No Groq API keys configured."),
        ):
            resp = client.post(
                "/api/agents/source-understanding",
                json={"source_text": self.sample_source},
            )

        self.assertEqual(resp.status_code, 503)
        self.assertIn("No Groq API keys configured", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
