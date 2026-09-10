import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import (
    ContentStrategy,
    LinkedInStrategy,
    InstagramStrategy,
    AdvisoryStrategy,
)
from app.schemas.content_generation import (
    InstagramContent,
    GeneratedContent,
)
from app.schemas.pipeline import PipelineGenerationResponse
from app.services.content_pipeline import ContentPipeline
from app.llm.exceptions import GroqRequestError
from app.main import app


class TestPipelineGeneration(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for Agent 1 -> Agent 2 -> Agent 3 full pipeline:
    - Service sync run_full()
    - Service async arun_full()
    - Failure propagation and halting if an agent fails
    - FastAPI endpoint POST /api/pipeline/content-generation
    """

    def setUp(self):
        self.sample_source_text = (
            "A university technology club organized a two-day artificial intelligence workshop "
            "in 2026 for undergraduate students. More than 150 students participated."
        )

        self.sample_source_understanding = SourceUnderstanding(
            title="University AI Workshop 2026",
            summary="A university technology club organized a two-day AI workshop in 2026 for over 150 students.",
            main_topic="AI Workshop",
            key_points=["Two-day workshop in 2026", "150+ students participated"],
            facts=["Organized in 2026", "150+ students"],
            entities=["University technology club"],
            important_numbers=["150", "two-day", "2026"],
            dates=["2026"],
            claims=[],
            terminology=["artificial intelligence"],
            target_audience="Undergraduates",
            tone="Informative",
            source_type="Report",
        )

        self.sample_strategy = ContentStrategy(
            summary="Campaign on practical student AI training.",
            overall_angle="Empowering students with hands-on AI tools.",
            target_audience="Students and technology leaders.",
            key_takeaway="Practical workshops rapidly upskill students.",
            linkedin=LinkedInStrategy(
                objective="Highlight workshop success",
                audience="Tech industry professionals",
                angle="Building the AI workforce of tomorrow",
                key_message="Over 150 students trained in 2026",
                tone="Professional",
                cta="Connect to learn more",
                recommended_structure=["Hook", "Details", "CTA"],
            ),
            instagram=InstagramStrategy(
                objective="Event recap",
                audience="University students",
                angle="Highlights from the 2026 AI workshop",
                carousel_direction=["Slide 1: Recap", "Slide 2: Numbers", "Slide 3: Next steps"],
                visual_direction="Workshop photos",
                tone="Dynamic",
                cta="Follow for next events",
            ),
            advisory=AdvisoryStrategy(
                objective="Department briefing",
                audience="University faculty",
                key_information=["150 students participated in 2026"],
                priority="Medium",
                tone="Executive",
                recommended_structure=["Summary", "Outcomes"],
            ),
        )

        self.sample_generated_content = GeneratedContent(
            linkedin="150+ students attended our 2026 AI workshop.\n\n#AI #Education",
            instagram=InstagramContent(
                caption="Recap of our 2026 AI workshop!",
                slides=["Slide 1: Recap", "Slide 2: 150+ students", "Slide 3: Next steps"],
                hashtags=["#AI", "#Tech"],
                call_to_action="Follow for more!",
            ),
            advisory="EXECUTIVE SUMMARY: 150 students completed the 2026 AI workshop successfully.",
        )

    def test_01_pipeline_sync_run_full(self):
        """Test synchronous full pipeline execution."""
        mock_source_agent = MagicMock()
        mock_source_agent.understand.return_value = self.sample_source_understanding

        mock_strategy_agent = MagicMock()
        mock_strategy_agent.strategize.return_value = self.sample_strategy

        mock_generation_agent = MagicMock()
        mock_generation_agent.generate.return_value = self.sample_generated_content

        pipeline = ContentPipeline(
            source_agent=mock_source_agent,
            strategy_agent=mock_strategy_agent,
            generation_agent=mock_generation_agent,
        )

        result = pipeline.run_full(self.sample_source_text)

        self.assertIsInstance(result, PipelineGenerationResponse)
        self.assertEqual(result.source_understanding.title, "University AI Workshop 2026")
        self.assertEqual(result.content_strategy.linkedin.objective, "Highlight workshop success")
        self.assertIn("150", result.generated_content.linkedin)

        mock_source_agent.understand.assert_called_once_with(self.sample_source_text)
        mock_strategy_agent.strategize.assert_called_once_with(self.sample_source_understanding)
        mock_generation_agent.generate.assert_called_once_with(
            source_understanding=self.sample_source_understanding,
            content_strategy=self.sample_strategy,
        )

    async def test_02_pipeline_async_arun_full(self):
        """Test asynchronous full pipeline execution."""
        mock_source_agent = MagicMock()
        mock_source_agent.aunderstand = AsyncMock(return_value=self.sample_source_understanding)

        mock_strategy_agent = MagicMock()
        mock_strategy_agent.astrategize = AsyncMock(return_value=self.sample_strategy)

        mock_generation_agent = MagicMock()
        mock_generation_agent.agenerate = AsyncMock(return_value=self.sample_generated_content)

        pipeline = ContentPipeline(
            source_agent=mock_source_agent,
            strategy_agent=mock_strategy_agent,
            generation_agent=mock_generation_agent,
        )

        result = await pipeline.arun_full(self.sample_source_text)

        self.assertIsInstance(result, PipelineGenerationResponse)
        self.assertEqual(result.source_understanding.title, "University AI Workshop 2026")
        self.assertEqual(result.content_strategy.linkedin.objective, "Highlight workshop success")
        self.assertIn("150", result.generated_content.linkedin)

        mock_source_agent.aunderstand.assert_called_once_with(self.sample_source_text)
        mock_strategy_agent.astrategize.assert_called_once_with(self.sample_source_understanding)
        mock_generation_agent.agenerate.assert_called_once_with(
            source_understanding=self.sample_source_understanding,
            content_strategy=self.sample_strategy,
        )

    async def test_03_pipeline_halt_on_agent1_failure(self):
        """Test pipeline halts immediately if Agent 1 fails, without calling Agent 2 or Agent 3."""
        mock_source_agent = MagicMock()
        mock_source_agent.aunderstand = AsyncMock(side_effect=GroqRequestError("Agent 1 network error"))

        mock_strategy_agent = MagicMock()
        mock_strategy_agent.astrategize = AsyncMock()

        mock_generation_agent = MagicMock()
        mock_generation_agent.agenerate = AsyncMock()

        pipeline = ContentPipeline(
            source_agent=mock_source_agent,
            strategy_agent=mock_strategy_agent,
            generation_agent=mock_generation_agent,
        )

        with self.assertRaises(GroqRequestError):
            await pipeline.arun_full(self.sample_source_text)

        mock_strategy_agent.astrategize.assert_not_called()
        mock_generation_agent.agenerate.assert_not_called()

    async def test_04_pipeline_halt_on_agent2_failure(self):
        """Test pipeline halts immediately if Agent 2 fails, without calling Agent 3."""
        mock_source_agent = MagicMock()
        mock_source_agent.aunderstand = AsyncMock(return_value=self.sample_source_understanding)

        mock_strategy_agent = MagicMock()
        mock_strategy_agent.astrategize = AsyncMock(side_effect=GroqRequestError("Agent 2 rate limit"))

        mock_generation_agent = MagicMock()
        mock_generation_agent.agenerate = AsyncMock()

        pipeline = ContentPipeline(
            source_agent=mock_source_agent,
            strategy_agent=mock_strategy_agent,
            generation_agent=mock_generation_agent,
        )

        with self.assertRaises(GroqRequestError):
            await pipeline.arun_full(self.sample_source_text)

        mock_generation_agent.agenerate.assert_not_called()

    def test_05_api_endpoint_success(self):
        """Verify POST /api/pipeline/content-generation returns 200 OK with full pipeline output."""
        client = TestClient(app)
        expected_response = PipelineGenerationResponse(
            source_understanding=self.sample_source_understanding,
            content_strategy=self.sample_strategy,
            generated_content=self.sample_generated_content,
        )
        with patch(
            "app.api.pipeline.content_pipeline.arun_full",
            new_callable=AsyncMock,
        ) as mock_arun_full:
            mock_arun_full.return_value = expected_response
            payload = {"source_text": self.sample_source_text}
            response = client.post("/api/pipeline/content-generation", json=payload)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("source_understanding", data)
            self.assertIn("content_strategy", data)
            self.assertIn("generated_content", data)
            self.assertIn("linkedin", data["generated_content"])
            self.assertIn("instagram", data["generated_content"])
            self.assertIn("advisory", data["generated_content"])

    def test_06_api_endpoint_empty_source(self):
        """Verify POST /api/pipeline/content-generation returns 422 on empty or whitespace source."""
        client = TestClient(app)
        res_empty = client.post("/api/pipeline/content-generation", json={"source_text": ""})
        self.assertEqual(res_empty.status_code, 422)

        res_ws = client.post("/api/pipeline/content-generation", json={"source_text": "   "})
        self.assertEqual(res_ws.status_code, 422)


if __name__ == "__main__":
    unittest.main()
