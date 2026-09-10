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
from app.schemas.content_generation import (
    InstagramContent,
    GeneratedContent,
    ContentGenerationRequest,
)
from app.agents.content_generation import (
    ContentGenerationAgent,
    SYSTEM_PROMPT,
)
from app.main import app


class TestContentGeneration(unittest.IsolatedAsyncioTestCase):
    """
    Unit test suite for Agent 3 (Content Generation Agent):
    - Input & output schema structure
    - Platform content fields (LinkedIn, Instagram, Advisory)
    - Anti-hallucination prompt verification
    - Agent execution (sync and async)
    - Platform selectivity
    - FastAPI endpoint integration
    """

    def setUp(self):
        self.sample_source_understanding = SourceUnderstanding(
            title="University AI Workshop 2026",
            summary="A university technology club organized a two-day AI workshop in 2026 for over 150 students.",
            main_topic="Artificial Intelligence Workshop",
            key_points=[
                "Two-day artificial intelligence workshop organized in 2026",
                "Covered machine learning fundamentals and prompt engineering",
                "More than 150 undergraduate students participated",
                "Hands-on activities building small AI-powered applications",
            ],
            facts=[
                "Organized by a university technology club",
                "Held in 2026",
                "Duration was two days",
                "More than 150 students participated",
            ],
            entities=["University technology club"],
            important_numbers=["two-day", "150", "2026"],
            dates=["2026"],
            claims=[],
            terminology=["machine learning", "prompt engineering", "AI-powered applications"],
            target_audience="Undergraduate students",
            tone="Informative",
            source_type="Report",
        )

        self.sample_strategy = ContentStrategy(
            summary="Multi-platform campaign highlighting hands-on student AI education.",
            overall_angle="Empowering over 150 undergraduates with practical AI skills in 2026.",
            target_audience="Students, academic faculty, and technology recruiters.",
            key_takeaway="Hands-on workshops bridge the gap between AI theory and real-world application.",
            linkedin=LinkedInStrategy(
                objective="Showcase student-led innovation and tech education leadership",
                audience="Technology professionals and university alumni",
                angle="Why practical AI education matters for the next generation of engineers",
                key_message="Over 150 students built working AI applications at the 2026 workshop.",
                tone="Professional",
                cta="How is your organization supporting early AI talent? Share below.",
                recommended_structure=["Hook", "Workshop Overview", "Student Achievement", "CTA"],
            ),
            instagram=InstagramStrategy(
                objective="Engage students with dynamic visual event highlights",
                audience="Undergraduate students and campus community",
                angle="Two days of building AI: inside the 2026 workshop",
                carousel_direction=[
                    "Slide 1: Two days, 150+ students, unlimited AI potential",
                    "Slide 2: Day 1 - Machine Learning & Prompt Engineering",
                    "Slide 3: Day 2 - Building AI applications hands-on",
                    "Slide 4: What the students built",
                    "Slide 5: Join the next workshop",
                ],
                visual_direction="Bright workshop photos with code snippet overlays and bold numbers",
                tone="Dynamic",
                cta="Drop a comment if you want to join our next AI hackathon!",
            ),
            advisory=AdvisoryStrategy(
                objective="Brief university department leadership on student AI engagement",
                audience="Academic Deans and Department Chairs",
                key_information=[
                    "150+ undergraduate participants",
                    "High interest in machine learning and prompt engineering",
                    "Successful hands-on application development",
                ],
                priority="Medium",
                tone="Executive",
                recommended_structure=["Executive Summary", "Student Participation", "Recommendations"],
            ),
        )

        self.sample_generated_content = GeneratedContent(
            linkedin=(
                "Over 150 undergraduate students joined us for an intensive two-day artificial intelligence "
                "workshop in 2026.\n\nFrom machine learning fundamentals to hands-on prompt engineering, "
                "students built real AI-powered applications from the ground up.\n\n"
                "How is your organization fostering early AI talent? Share below.\n\n"
                "#ArtificialIntelligence #MachineLearning #TechEducation #StudentsInTech"
            ),
            instagram=InstagramContent(
                caption="Two days. 150+ students. Real AI applications built from scratch! Check out our workshop recap.",
                slides=[
                    "Slide 1: Two days, 150+ students, unlimited AI potential",
                    "Slide 2: Day 1 - Machine Learning & Prompt Engineering",
                    "Slide 3: Day 2 - Building AI applications hands-on",
                    "Slide 4: Hands-on student applications",
                    "Slide 5: Join our next session",
                ],
                hashtags=["#AIWorkshop", "#StudentTech", "#MachineLearning", "#UniversityLife"],
                call_to_action="Drop a comment if you want to join our next AI hackathon!",
            ),
            advisory=(
                "EXECUTIVE BRIEFING: AI WORKSHOP OUTCOMES (2026)\n\n"
                "Summary: The university technology club successfully hosted a two-day artificial intelligence "
                "workshop engaging more than 150 undergraduate participants.\n\n"
                "Key Findings:\n"
                "- Strong student engagement in machine learning and prompt engineering.\n"
                "- Practical hands-on development enabled students to construct working AI-powered applications.\n\n"
                "Recommendation: Continue institutional support for student-led technical workshops."
            ),
        )

    def test_01_schema_validation(self):
        """Test schema validation for GeneratedContent and InstagramContent."""
        content = self.sample_generated_content
        self.assertIsNotNone(content.linkedin)
        self.assertIsNotNone(content.instagram)
        self.assertIsNotNone(content.advisory)
        self.assertEqual(len(content.instagram.slides), 5)
        self.assertEqual(len(content.instagram.hashtags), 4)
        self.assertIn("150", content.linkedin)

    def test_02_platform_selectivity_schema(self):
        """Test that GeneratedContent supports optional fields when only specific platforms are requested."""
        linkedin_only = GeneratedContent(
            linkedin="Professional post text",
            instagram=None,
            advisory=None,
        )
        self.assertIsNotNone(linkedin_only.linkedin)
        self.assertIsNone(linkedin_only.instagram)
        self.assertIsNone(linkedin_only.advisory)

    def test_03_system_prompt_constraints(self):
        """Verify that the system prompt strictly enforces grounding and anti-hallucination."""
        prompt_lower = SYSTEM_PROMPT.lower()
        self.assertIn("content generation agent", prompt_lower)
        self.assertIn("anti-hallucination", prompt_lower)
        self.assertIn("linkedin", prompt_lower)
        self.assertIn("instagram", prompt_lower)
        self.assertIn("advisory", prompt_lower)
        self.assertIn("never invent", prompt_lower)

    def test_04_agent_sync_execution(self):
        """Verify synchronous generate method with mock LLM."""
        mock_manager = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self.sample_generated_content.model_dump_json()
        mock_manager.invoke.return_value = mock_response

        agent = ContentGenerationAgent(manager=mock_manager)
        result = agent.generate(
            source_understanding=self.sample_source_understanding,
            content_strategy=self.sample_strategy,
        )

        self.assertIsInstance(result, GeneratedContent)
        self.assertIn("150", result.linkedin)
        self.assertEqual(len(result.instagram.slides), 5)
        mock_manager.invoke.assert_called_once()

    async def test_05_agent_async_execution(self):
        """Verify asynchronous agenerate method with mock LLM."""
        mock_manager = MagicMock()
        mock_response = MagicMock()
        mock_response.content = self.sample_generated_content.model_dump_json()
        mock_manager.ainvoke = AsyncMock(return_value=mock_response)

        agent = ContentGenerationAgent(manager=mock_manager)
        result = await agent.agenerate(
            source_understanding=self.sample_source_understanding,
            content_strategy=self.sample_strategy,
        )

        self.assertIsInstance(result, GeneratedContent)
        self.assertIn("150", result.linkedin)
        mock_manager.ainvoke.assert_called_once()

    def test_06_api_endpoint_success(self):
        """Verify POST /api/agents/content-generation returns 200 OK with valid inputs."""
        client = TestClient(app)
        with patch(
            "app.api.content_generation.content_generation_agent.agenerate",
            new_callable=AsyncMock,
        ) as mock_agenerate:
            mock_agenerate.return_value = self.sample_generated_content
            payload = {
                "source_understanding": self.sample_source_understanding.model_dump(),
                "content_strategy": self.sample_strategy.model_dump(),
            }
            response = client.post("/api/agents/content-generation", json=payload)

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("linkedin", data)
            self.assertIn("instagram", data)
            self.assertIn("advisory", data)
            self.assertEqual(len(data["instagram"]["slides"]), 5)

    def test_07_api_endpoint_validation_error(self):
        """Verify POST /api/agents/content-generation returns 422 on invalid payload."""
        client = TestClient(app)
        response = client.post("/api/agents/content-generation", json={})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
