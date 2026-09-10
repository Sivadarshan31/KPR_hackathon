import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.schemas.source_understanding import SourceUnderstanding
from app.schemas.content_strategy import (
    ContentStrategy,
    LinkedInStrategy,
    InstagramStrategy,
    AdvisoryStrategy,
)
from app.schemas.pipeline import PipelineRequest, PipelineResponse
from app.services.content_pipeline import ContentPipeline
from app.llm.exceptions import GroqRequestError
from app.main import app


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for the connected pipeline (Agent 1 -> Agent 2):
    - Input validation (rejection of empty/whitespace input)
    - Sequential execution flow
    - Error propagation and safety (halting on Agent 1 failure)
    - Combined FastAPI endpoint integration
    """

    def setUp(self):
        self.sample_text = (
            "Artificial intelligence is transforming healthcare. A recent pilot program "
            "analyzed 50,000 medical images over six months to assist doctors with diagnosis."
        )

        self.mock_su = SourceUnderstanding(
            title="AI Healthcare Diagnostic Pilot",
            summary="A pilot analyzed 50,000 medical images over six months.",
            main_topic="AI in medical imaging",
            key_points=["AI assists doctors", "50,000 images analyzed"],
            facts=["50,000 medical images analyzed over six months"],
            entities=["Healthcare AI pilot"],
            important_numbers=["50,000", "six months"],
            dates=["six months"],
            claims=[],
            terminology=["medical images", "diagnostic assistance"],
            target_audience="Healthcare professionals",
            tone="Informative",
            source_type="Report",
        )

        self.mock_cs = ContentStrategy(
            summary="Strategic communication plan for healthcare AI trial.",
            overall_angle="Assisting doctors with 50,000 image analysis.",
            target_audience="Medical clinicians, hospital leadership, health-tech researchers.",
            key_takeaway="AI diagnostic assistance validated over 50,000 medical images.",
            linkedin=LinkedInStrategy(
                objective="Professional awareness of medical AI impact",
                audience="Healthcare administrators and clinicians",
                angle="Real-world results from 50,000 medical images",
                key_message="AI provides proven diagnostic assistance over 50,000 images.",
                tone="Professional",
                cta="Join the discussion on clinical AI adoption.",
                recommended_structure=["Hook", "Evidence", "Impact", "CTA"],
            ),
            instagram=InstagramStrategy(
                objective="Visual overview of AI in medicine",
                audience="General public and tech enthusiasts",
                angle="How AI analyzed 50,000 images",
                carousel_direction=["Slide 1", "Slide 2", "Slide 3"],
                visual_direction="Medical tech infographics with scan overlays",
                tone="Educational",
                cta="Save this post to follow AI healthcare breakthroughs.",
            ),
            advisory=AdvisoryStrategy(
                objective="Brief clinical directors on pilot findings",
                audience="Hospital CIOs and Medical Directors",
                key_information=["50,000 images analyzed", "6-month duration", "Diagnostic assistance"],
                priority="High",
                tone="Executive",
                recommended_structure=["Briefing", "Metrics", "Action items"],
            ),
        )

    def test_01_pipeline_request_validation(self):
        """Test input schema validation: accepts non-empty, rejects empty and whitespace."""
        req = PipelineRequest(source_text="Valid source text")
        self.assertEqual(req.source_text, "Valid source text")

        with self.assertRaises(ValidationError):
            PipelineRequest(source_text="")

        with self.assertRaises(ValidationError):
            PipelineRequest(source_text="   \n\t  ")

    async def test_02_pipeline_async_flow_success(self):
        """Test happy path: Agent 1 executes, passes to Agent 2, returns combined response."""
        mock_source_agent = MagicMock()
        mock_source_agent.aunderstand = AsyncMock(return_value=self.mock_su)

        mock_strategy_agent = MagicMock()
        mock_strategy_agent.astrategize = AsyncMock(return_value=self.mock_cs)

        pipeline = ContentPipeline(
            source_agent=mock_source_agent,
            strategy_agent=mock_strategy_agent,
        )

        response = await pipeline.arun(self.sample_text)

        self.assertIsInstance(response, PipelineResponse)
        self.assertEqual(response.source_understanding.title, "AI Healthcare Diagnostic Pilot")
        self.assertEqual(response.content_strategy.advisory.priority, "High")
        mock_source_agent.aunderstand.assert_called_once_with(self.sample_text)
        mock_strategy_agent.astrategize.assert_called_once_with(self.mock_su)

    async def test_03_pipeline_halts_on_agent1_failure(self):
        """Test failure path: If Agent 1 fails, pipeline halts and Agent 2 is never invoked."""
        mock_source_agent = MagicMock()
        mock_source_agent.aunderstand = AsyncMock(side_effect=GroqRequestError("Agent 1 provider error"))

        mock_strategy_agent = MagicMock()
        mock_strategy_agent.astrategize = AsyncMock()

        pipeline = ContentPipeline(
            source_agent=mock_source_agent,
            strategy_agent=mock_strategy_agent,
        )

        with self.assertRaises(GroqRequestError):
            await pipeline.arun(self.sample_text)

        mock_strategy_agent.astrategize.assert_not_called()

    def test_04_combined_api_endpoint(self):
        """Test POST /api/pipeline/source-to-strategy returns 200 with both agent responses."""
        client = TestClient(app)
        mock_resp = PipelineResponse(
            source_understanding=self.mock_su,
            content_strategy=self.mock_cs,
        )
        with patch(
            "app.api.pipeline.content_pipeline.arun",
            new_callable=AsyncMock,
        ) as mock_arun:
            mock_arun.return_value = mock_resp
            payload = {"source_text": self.sample_text}
            res = client.post("/api/pipeline/source-to-strategy", json=payload)

            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("source_understanding", data)
            self.assertIn("content_strategy", data)
            self.assertEqual(data["source_understanding"]["title"], "AI Healthcare Diagnostic Pilot")
            self.assertIn("linkedin", data["content_strategy"])
            self.assertIn("instagram", data["content_strategy"])
            self.assertIn("advisory", data["content_strategy"])

    def test_05_combined_api_rejects_empty_input(self):
        """Test POST /api/pipeline/source-to-strategy rejects empty source_text with 422."""
        client = TestClient(app)
        res = client.post("/api/pipeline/source-to-strategy", json={"source_text": ""})
        self.assertEqual(res.status_code, 422)

        res_ws = client.post("/api/pipeline/source-to-strategy", json={"source_text": "   "})
        self.assertEqual(res_ws.status_code, 422)


if __name__ == "__main__":
    unittest.main()
