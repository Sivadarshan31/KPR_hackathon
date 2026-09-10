import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import (
    ContentStrategy,
    ContentStrategyRequest,
    LinkedInStrategy,
    InstagramStrategy,
    AdvisoryStrategy,
)
from app.agents.content_strategy import (
    ContentStrategyAgent,
    SYSTEM_PROMPT,
)
from app.llm.exceptions import GroqConfigurationError
from app.main import app


class TestContentStrategy(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for Agent 2 (Content Strategy Agent):
    - Input & output schema structure
    - Platform strategy fields (LinkedIn, Instagram, Advisory)
    - Anti-hallucination prompt verification
    - Agent execution (sync and async)
    - FastAPI endpoint integration
    """

    def setUp(self):
        self.sample_source_understanding = SourceUnderstanding(
            title="Autonomous Underwater Vehicle for Marine Pollution Monitoring",
            summary="NIOT developed an autonomous underwater vehicle to monitor marine pollution.",
            main_topic="Autonomous underwater vehicle for marine pollution monitoring",
            key_points=[
                "Autonomous underwater vehicle was developed",
                "Designed to monitor marine pollution",
                "Operates for up to 18 hours",
                "Project began in 2026",
            ],
            facts=[
                "The National Institute of Ocean Technology developed the vehicle",
                "The vehicle operates for up to 18 hours",
                "The project began in 2026",
            ],
            entities=["National Institute of Ocean Technology"],
            important_numbers=["18 hours"],
            dates=["2026"],
            claims=[],
            terminology=["autonomous underwater vehicle", "marine pollution"],
            target_audience="Environmental scientists and marine researchers",
            tone="Technical",
            source_type="Technical document",
        )

        self.sample_strategy = ContentStrategy(
            summary="Multi-platform campaign highlighting robotics innovation in marine conservation.",
            overall_angle="How autonomous robotics can safeguard ocean ecosystems for up to 18 hours per mission.",
            target_audience="Environmental professionals, marine engineers, and sustainability advocates.",
            key_takeaway="Autonomous underwater vehicles provide high-endurance marine monitoring starting in 2026.",
            linkedin=LinkedInStrategy(
                objective="Highlight engineering leadership in ocean preservation",
                audience="Environmental engineers and sustainability directors",
                angle="Autonomous technology as a scalable solution for marine pollution",
                key_message="NIOT's 18-hour autonomous vehicle proves how robotics accelerates marine cleanup.",
                tone="Authoritative",
                cta="Share your thoughts on robotics in environmental monitoring in the comments.",
                recommended_structure=["Hook", "Context", "Key 18-hr metric", "Strategic impact", "Discussion CTA"],
            ),
            instagram=InstagramStrategy(
                objective="Showcase ocean exploration tech visually",
                audience="Tech enthusiasts and ocean lovers",
                angle="Under the sea with autonomous pollution monitors",
                carousel_direction=["Slide 1: Hook", "Slide 2: The 18-hour mission", "Slide 3: Water sensors"],
                visual_direction="High-contrast underwater visuals with sensor data overlays",
                tone="Inspiring",
                cta="Save this post for your daily dose of clean ocean innovation.",
            ),
            advisory=AdvisoryStrategy(
                objective="Brief leadership on new ocean monitoring capabilities",
                audience="Environmental regulatory bodies and agency leadership",
                key_information=["Autonomous operation up to 18 hours", "Deployment begins 2026", "Sensors monitor water quality"],
                priority="High",
                tone="Objective",
                recommended_structure=["Executive Summary", "Capabilities", "Deployment Timeline", "Recommendations"],
            ),
        )

    def test_01_schema_validation(self):
        """Test schema validation for ContentStrategy and platform models."""
        strat = self.sample_strategy
        self.assertEqual(strat.linkedin.tone, "Authoritative")
        self.assertEqual(strat.instagram.tone, "Inspiring")
        self.assertEqual(strat.advisory.priority, "High")
        self.assertIn("18-hour", strat.linkedin.key_message)

    def test_02_system_prompt_constraints(self):
        """Verify that the system prompt strictly enforces grounding and anti-hallucination."""
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("anti-hallucination", prompt_lower)
        self.assertIn("linkedin", prompt_lower)
        self.assertIn("instagram", prompt_lower)
        self.assertIn("advisory", prompt_lower)
        self.assertIn("50,000", prompt_lower)

    def test_03_agent_sync_execution(self):
        """Verify synchronous strategize method with mock LLM."""
        mock_manager = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self.sample_strategy.model_dump_json()
        mock_manager.invoke.return_value = mock_response

        agent = ContentStrategyAgent(manager=mock_manager)
        result = agent.strategize(self.sample_source_understanding)

        self.assertIsInstance(result, ContentStrategy)
        self.assertEqual(result.advisory.priority, "High")
        mock_manager.invoke.assert_called_once()

    async def test_04_agent_async_execution(self):
        """Verify asynchronous astrategize method with mock LLM."""
        mock_manager = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self.sample_strategy.model_dump_json()
        mock_manager.ainvoke = AsyncMock(return_value=mock_response)

        agent = ContentStrategyAgent(manager=mock_manager)
        result = await agent.astrategize(self.sample_source_understanding)

        self.assertIsInstance(result, ContentStrategy)
        self.assertEqual(result.linkedin.objective, "Highlight engineering leadership in ocean preservation")
        mock_manager.ainvoke.assert_called_once()

    def test_05_api_endpoint_success(self):
        """Verify POST /api/agents/content-strategy returns 200 OK with valid strategy."""
        client = TestClient(app)
        with patch(
            "app.api.content_strategy.content_strategy_agent.astrategize",
            new_callable=AsyncMock,
        ) as mock_astrategize:
            mock_astrategize.return_value = self.sample_strategy
            payload = {"source_understanding": self.sample_source_understanding.model_dump()}
            response = client.post("/api/agents/content-strategy", json=payload)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("linkedin", data)
            self.assertIn("instagram", data)
            self.assertIn("advisory", data)
            self.assertEqual(data["advisory"]["priority"], "High")


if __name__ == "__main__":
    unittest.main()
