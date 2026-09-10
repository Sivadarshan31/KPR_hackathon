import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_generation import GeneratedContent
from app.schemas.validation import ValidationResult, ValidationRequest
from app.agents.validation import ValidationAgent, SYSTEM_PROMPT
from app.main import app


class TestValidationAgent(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for Agent 4 (Validation Agent).
    """

    def setUp(self):
        self.sample_su = SourceUnderstanding(
            title="Marine Robotics 2026",
            summary="A university developed an autonomous vehicle operating for 18 hours in 2026.",
            main_topic="Marine monitoring",
            key_points=["18-hour mission", "Operated in 2026"],
            facts=["Operated for 18 hours", "Launched in 2026"],
            entities=["University lab"],
            important_numbers=["18 hours", "2026"],
            dates=["2026"],
            claims=[],
            terminology=["autonomous vehicle"],
            target_audience="Engineers",
            tone="Technical",
            source_type="Report",
        )

        self.sample_gc = GeneratedContent(
            linkedin="Our team launched an 18-hour autonomous marine monitoring vehicle in 2026.\n\n#MarineTech #Robotics",
        )

        self.sample_val_pass = ValidationResult(
            passed=True,
            score=0.95,
            issues=[],
            suggestions=[],
            checks={
                "grounding_verified": True,
                "no_hallucinations": True,
                "numbers_accurate": True,
                "format_compliance": True,
            },
        )

        self.sample_val_fail = ValidationResult(
            passed=False,
            score=0.4,
            issues=["Unsupported claim: claimed $10M revenue not found in source text."],
            suggestions=["Remove all references to revenue."],
            checks={
                "grounding_verified": False,
                "no_hallucinations": False,
                "numbers_accurate": False,
                "format_compliance": True,
            },
        )

    def test_01_schema_validation(self):
        """Test ValidationResult fields and constraints."""
        res = self.sample_val_pass
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.95)
        self.assertEqual(len(res.issues), 0)
        self.assertTrue(res.checks["no_hallucinations"])

        fail_res = self.sample_val_fail
        self.assertFalse(fail_res.passed)
        self.assertEqual(len(fail_res.issues), 1)

    def test_02_system_prompt_constraints(self):
        """Verify prompt enforces strict factual auditing and zero hallucination tolerance."""
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("validation agent", prompt_lower)
        self.assertIn("anti-hallucination", prompt_lower)
        self.assertIn("grounding", prompt_lower)
        self.assertIn("critical validator", prompt_lower)

    def test_03_agent_sync_execution(self):
        """Test synchronous validate method with mock LLM."""
        mock_manager = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = self.sample_val_pass.model_dump_json()
        mock_manager.invoke.return_value = mock_resp

        agent = ValidationAgent(manager=mock_manager)
        result = agent.validate(self.sample_su, self.sample_gc)

        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.passed)
        mock_manager.invoke.assert_called_once()

    async def test_04_agent_async_execution(self):
        """Test asynchronous avalidate method with mock LLM."""
        mock_manager = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = self.sample_val_pass.model_dump_json()
        mock_manager.ainvoke = AsyncMock(return_value=mock_resp)

        agent = ValidationAgent(manager=mock_manager)
        result = await agent.avalidate(self.sample_su, self.sample_gc)

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.score, 0.95)
        mock_manager.ainvoke.assert_called_once()

    def test_05_api_endpoint_success(self):
        """Test POST /api/agents/validation endpoint."""
        client = TestClient(app)
        with patch(
            "app.api.validation.validation_agent.avalidate",
            new_callable=AsyncMock,
        ) as mock_avalidate:
            mock_avalidate.return_value = self.sample_val_pass
            payload = {
                "source_understanding": self.sample_su.model_dump(),
                "generated_content": self.sample_gc.model_dump(),
            }
            resp = client.post("/api/agents/validation", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["passed"])
            self.assertEqual(data["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
